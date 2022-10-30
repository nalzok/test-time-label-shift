from hashlib import sha256

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from . import MultipleDomainDataset


class MultipleDomainCheXpert(MultipleDomainDataset):

    def __init__(self, root, generator, train_domains, Y_column, Z_column, target_domain_count):
        input_shape = (1, 1376)
        C = 2
        K = 2
        confounder_strength = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        super().__init__(input_shape, C, K, confounder_strength)

        if Y_column != "PNEUMONIA" or Z_column != "GENDER":
            raise NotImplementedError(f"(Y, Z) must be (PNEUMONIA, GENDER)")

        if len(train_domains) != 1:
            raise NotImplementedError(
                "Training on multiple source distributions is not supported yet."
            )

        m = sha256()
        m.update(str(train_domains).encode())
        m.update(Y_column.encode())
        m.update(Z_column.encode())
        m.update(str(target_domain_count).encode())
        cache_key = m.hexdigest()
        cache_file = root / 'cached' / f'{cache_key}.pt'
        if cache_file.is_file():
            # NOTE: The torch.Generator state won't be the same if we load from cache
            print(f'Loading cached datasets from {cache_file}')
            self.domains = torch.load(cache_file)
            return

        if root is None:
            raise ValueError('Data directory not specified!')

        self.generator = generator
        self.train_domains = train_domains

        labels: pd.DataFrame = pd.read_csv(root / "labels.csv", index_col="image_id")
        embeddings = np.load(root / "embeddings.npz")

        # 0 = no mention    - 15933
        # 1 = positive      - 4657
        # 2 = uncertain     - 2054
        # 3 = negative      - 167855
        labels = labels.loc[labels["PNEUMONIA"].isin({1, 3})]
        labels = labels.loc[labels["GENDER"] != "Unknown"]

        self.Y_column = Y_column
        self.Z_column = Z_column
        for column in (Y_column, Z_column):
            code, uniques = pd.factorize(labels[column], sort=True)
            if len(uniques) != 2:
                raise NotImplementedError(f"Column {column} has {len(uniques)} != 2 levels.")
            labels[column] = code

        # Y: 0 = Positive, 1 = Negative
        # Z: 0 = Female, 1 = Male

        # joint distribution of Y and Z
        confounder1 = np.array([[0.0, 0.025], [0.0, 0.975]])
        confounder2 = np.array([[0.025, 0.0], [0.975, 0.0]])

        domains = [None for _ in confounder_strength]

        # 0 = Positive, Female  - 1939
        # 1 = Positive, Male    - 2718
        # 2 = Negative, Female  - 69209
        # 3 = Negative, Male    - 98645
        labels["M"] = 2 * labels[Y_column] + labels[Z_column]
        mask = np.ones(len(labels.index), dtype=bool)

        # Sample source domains
        for i, strength in enumerate(self.confounder_strength):
            if i not in train_domains:
                continue

            quota = labels["M"].loc[mask].value_counts(ascending=True).to_numpy() - target_domain_count
            quota = torch.from_numpy(quota)
            joint_M = torch.from_numpy(strength * confounder1 + (1-strength) * confounder2)
            joint_M_flatten = joint_M.flatten()
            count = torch.round(torch.min(quota/joint_M_flatten)*joint_M_flatten).long()
            count = count.reshape((2, 2))
            joint_M = count / torch.sum(count)

            domain, in_sample = self.sample(embeddings, labels, mask, count)
            mask &= ~labels.index.isin(in_sample)
            domains[i] = (joint_M, domain)

        # Sample target domains
        for i, strength in enumerate(self.confounder_strength):
            if i in train_domains:
                continue

            joint_M = torch.from_numpy(strength * confounder1 + (1-strength) * confounder2)
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
            assert total_count == target_domain_count, f"Incorrect total count {total_count} != {target_domain_count}"

            count = count.reshape((2, 2))
            joint_M = count / torch.sum(count)

            domain, _ = self.sample(embeddings, labels, mask, count)
            domains[i] = (joint_M, domain)

        self.domains = domains

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print(f'Saving cached datasets to {cache_file}')
        torch.save(self.domains, cache_file)


    def sample(self, embeddings, labels, mask, count):
        in_sample = set()

        for Y in range(2):
            for Z in range(2):
                masked = labels.loc[mask & (labels["M"] == 2 * Y + Z)]
                indices = masked.sample(int(count[Y, Z]), random_state=42)
                in_sample.update(indices.index)

        N = int(torch.sum(count))
        assert len(in_sample) == N, f"Incorrect number of elements {len(in_sample)} != {N}"

        x = torch.empty((N, *self.input_shape[1:]))
        y_tilde = torch.empty(N, dtype=torch.long)
        y = torch.empty(N, dtype=torch.long)
        z_flattened = torch.empty(N, dtype=torch.long)

        perm = torch.randperm(N, generator=self.generator)
        for i, key in enumerate(in_sample):
            x[perm[i]] = torch.Tensor(embeddings[key])
            row = labels.loc[key]
            y[perm[i]] = y_tilde[perm[i]] = row[self.Y_column]
            z_flattened[perm[i]] = row[self.Z_column]

        return TensorDataset(x, y_tilde, y, z_flattened), in_sample