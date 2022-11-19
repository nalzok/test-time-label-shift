from hashlib import sha256
import re

import numpy as np
from scipy.special import softmax
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from PIL import Image
import torchvision.transforms as T

from tta.datasets import MultipleDomainDataset


class MultipleDomainCheXpert(MultipleDomainDataset):

    def __init__(self, root, train_domains, generator, Y_column: str, Z_column: str, use_embedding: bool, target_domain_count: int):
        if len(train_domains) != 1:
            raise NotImplementedError(
                "Training on multiple source distributions is not supported yet."
            )
        train_domain = next(iter(train_domains))

        if use_embedding:
            input_shape = (1, 1376)
        else:
            input_shape = (1, 224, 224, 3)
        C = 2
        K = 2
        confounder_strength = np.linspace(0, 1, 21)

        m = sha256()
        m.update(self.__class__.__name__.encode())
        m.update(str(sorted(train_domains)).encode())
        m.update(generator.get_state().numpy().data.hex().encode())
        m.update(Y_column.encode())
        m.update(Z_column.encode())
        m.update(str(use_embedding).encode())
        m.update(str(target_domain_count).encode())

        m.update(str(input_shape).encode())
        m.update(str(C).encode())
        m.update(str(K).encode())
        m.update(confounder_strength.data.hex().encode())
        m.update(str(train_domain).encode())
        hexdigest = m.hexdigest()

        super().__init__(input_shape, C, K, confounder_strength, train_domain, hexdigest)

        cache_key = f'{train_domain}_{Y_column}_{Z_column}_{use_embedding}_{target_domain_count}_{hexdigest}'
        cache_file = root / 'cached' / f'{cache_key}.pt'
        if cache_file.is_file():
            # NOTE: The torch.Generator state won't be the same if we load from cache
            print(f'Loading cached datasets from {cache_file}')
            self.domains = torch.load(cache_file)
            return

        print('Building datasets... (this may take a while)')
        if root is None:
            raise ValueError('Data directory not specified!')

        self.generator = generator
        self.use_embedding = use_embedding
        self.train_domains = train_domains

        labels: pd.DataFrame = pd.read_csv(root / "labels.csv", index_col="image_id")
        if use_embedding:
            datastore = np.load(root / "embeddings.npz")
        else:
            datastore = CheXpertImages(root)

        #   PNEUMONIA
        # 0 = no mention    - 15933
        # 1 = positive      - 4657
        # 2 = uncertain     - 2054
        # 3 = negative      - 167855
        #
        #   EFFUSION
        # 0 = no mention    - 9527
        # 1 = positive      - 76726
        # 2 = uncertain     - 25371
        # 3 = negative      - 78875
        relevant_columns = {Y_column, Z_column}
        if "PNEUMONIA" in relevant_columns:
            labels = labels.loc[labels["PNEUMONIA"].isin({1, 3})]
        if "EFFUSION" in relevant_columns:
            labels = labels.loc[labels["EFFUSION"].isin({1, 3})]
        if "GENDER" in relevant_columns:
            labels = labels.loc[labels["GENDER"] != "Unknown"]

        self.Y_column = Y_column
        self.Z_column = Z_column
        for column in (Y_column, Z_column):
            code, uniques = pd.factorize(labels[column], sort=True)
            if len(uniques) != 2:
                raise NotImplementedError(f"Column {column} has {len(uniques)} != 2 levels.")
            labels[column] = code

        # PNEUMONIA/EFFUSION:   0 = Positive, 1 = Negative
        # GENDER:               0 = Female, 1 = Male
        labels["M"] = 2 * labels[Y_column] + labels[Z_column]
        print(Y_column, Z_column, labels["M"].value_counts().sort_index().values)
        mask = np.ones(len(labels.index), dtype=bool)

        # joint distribution of Y and Z
        if Y_column == "PNEUMONIA" and Z_column == "EFFUSION":
            # PNEUMONIA EFFUSION [ 1395  2702 69011 66799]
            anchor1 = np.array([[0.025, 0.0], [0.0, 0.975]])
            anchor2 = np.array([[0.0, 0.025], [0.975, 0.0]])
        elif Y_column == "PNEUMONIA" and Z_column == "GENDER":
            # PNEUMONIA GENDER [ 1939  2718 69209 98645]
            anchor1 = np.array([[0.025, 0.0], [0.0, 0.975]])
            anchor2 = np.array([[0.0, 0.025], [0.975, 0.0]])
        elif Y_column == "EFFUSION" and Z_column == "GENDER":
            # EFFUSION GENDER [31900 44826 32053 46822]
            anchor1 = np.array([[0.5, 0.0], [0.0, 0.5]])
            anchor2 = np.array([[0.0, 0.5], [0.5, 0.0]])
        elif Y_column == "GENDER" and Z_column == "EFFUSION":
            # GENDER EFFUSION [31900 32053 44826 46822]
            anchor1 = np.array([[0.5, 0.0], [0.0, 0.5]])
            anchor2 = np.array([[0.0, 0.5], [0.5, 0.0]])
        else:
            raise NotImplementedError(f"Please specify confounders for (Y, Z) = ({Y_column}, {Z_column})")

        domains = [None for _ in confounder_strength]

        # Sample source domains
        for i, strength in enumerate(self.confounder_strength):
            if i not in train_domains:
                continue

            quota = labels["M"].loc[mask].value_counts(ascending=True).to_numpy() - target_domain_count
            quota = torch.from_numpy(quota)
            joint_M = torch.from_numpy(strength * anchor1 + (1-strength) * anchor2)
            joint_M_flatten = joint_M.flatten()
            count = torch.round(torch.min(quota/joint_M_flatten)*joint_M_flatten).long()
            count = count.reshape((2, 2))
            joint_M = count / torch.sum(count)

            print(f"histogram(M) = {count.flatten()}")
            reservation = np.ceil(target_domain_count * np.maximum(anchor1, anchor2).flatten())
            domain, in_sample_patients = self.sample(datastore, labels, mask, count, reservation)
            mask &= ~labels["patient_id"].isin(in_sample_patients)
            domains[i] = (joint_M, domain)

        remainder = np.sum(mask)
        if remainder < target_domain_count:
            raise ValueError(f"Not enough data for target domains: {remainder} < {target_domain_count}")

        # Sample target domains
        for i, strength in enumerate(self.confounder_strength):
            if i in train_domains:
                continue

            joint_M = torch.from_numpy(strength * anchor1 + (1-strength) * anchor2)
            joint_M_flatten = joint_M.flatten()
            count = torch.round(target_domain_count * joint_M_flatten).long()

            l1, l2, l3 = torch.topk(count, 3).indices
            if torch.sum(count) > target_domain_count:
                count[l1] -= 1
            if torch.sum(count) > target_domain_count:
                count[l2] -= 1
            if torch.sum(count) > target_domain_count:
                count[l3] -= 1

            s1, s2, s3 = torch.topk(count, 3, largest=False).indices
            if torch.sum(count) < target_domain_count:
                count[s1] += 1
            if torch.sum(count) < target_domain_count:
                count[s2] += 1
            if torch.sum(count) < target_domain_count:
                count[s3] += 1

            total_count = torch.sum(count)
            assert total_count == target_domain_count, f"Incorrect total count: {total_count} != {target_domain_count}"

            count = count.reshape((2, 2))
            joint_M = count / torch.sum(count)

            print(f"histogram(M) = {count.flatten()}")
            domain, _ = self.sample(datastore, labels, mask, count, None)
            domains[i] = (joint_M, domain)

        self.domains = domains

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print(f'Saving cached datasets to {cache_file}')
        torch.save(self.domains, cache_file)


    def sample(self, datastore, labels, mask, count, reservation):
        random_state = 0
        while True:
            in_sample = set()
            for Y in range(2):
                for Z in range(2):
                    masked = labels.loc[mask & (labels["M"] == 2 * Y + Z)]
                    image_per_patient = masked.groupby("patient_id").size()
                    weights = softmax(image_per_patient.loc[masked["patient_id"]].values)
                    indices = masked.sample(int(count[Y, Z]), weights=weights, random_state=random_state)
                    in_sample.update(indices.index)

            in_sample_patients = { fname.split("/")[2] for fname in in_sample }
            remainder = np.bincount(labels["M"], weights=mask & ~labels["patient_id"].isin(in_sample_patients))
            if reservation is None or np.all(remainder >= reservation):
                print(f"  remainder = {remainder} >= {reservation} = target_domain_count")
                break

            random_state += 1
            print(f"  remainder = {remainder} < {reservation} = target_domain_count")

        N = int(torch.sum(count))
        assert len(in_sample) == N, f"Incorrect number of elements: {len(in_sample)} != {N}"

        x = torch.empty((N, *self.input_shape[1:]))
        y_tilde = torch.empty(N, dtype=torch.long)
        y = torch.empty(N, dtype=torch.long)
        z_flattened = torch.empty(N, dtype=torch.long)

        perm = torch.randperm(N, generator=self.generator)
        for i, key in enumerate(in_sample):
            x[perm[i]] = torch.Tensor(datastore[key])
            row = labels.loc[key]
            y[perm[i]] = y_tilde[perm[i]] = row[self.Y_column]
            z_flattened[perm[i]] = row[self.Z_column]

        return TensorDataset(x, y_tilde, y, z_flattened), in_sample_patients


class CheXpertImages:
    def __init__(self, root):
        self.root = root
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Lambda(lambda x: x.permute(1, 2, 0)), # (C, H, W) -> (H, W, C)
        ])
        self.pattern = re.compile("^CheXpert-v1.0/")

    def __getitem__(self, key):
        key = re.sub(self.pattern, "CheXpert-v1.0-small/", key)
        image = Image.open(self.root / key)
        return self.transform(image)
