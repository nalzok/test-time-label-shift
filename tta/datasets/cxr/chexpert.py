from hashlib import sha256
import re

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import torch
from PIL import Image
import torchvision.transforms as T

from tta.datasets.cxr import MultipleDomainCXR


class MultipleDomainCheXpert(MultipleDomainCXR):

    def __init__(self, root, train_domains, generator, Y_col: str, Z_col: str, use_embedding: bool, target_domain_count: int):
        if len(train_domains) != 1:
            raise NotImplementedError(
                "Training on multiple source distributions is not supported yet."
            )
        train_domain = next(iter(train_domains))
        patient_col = "patient_id"

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
        m.update(Y_col.encode())
        m.update(Z_col.encode())
        m.update(str(use_embedding).encode())
        m.update(str(target_domain_count).encode())

        m.update(str(input_shape).encode())
        m.update(str(C).encode())
        m.update(str(K).encode())
        m.update(confounder_strength.data.hex().encode())
        m.update(str(train_domain).encode())
        hexdigest = m.hexdigest()

        super().__init__(input_shape, C, K, confounder_strength, train_domain, hexdigest)

        cache_key = f'{train_domain}_{Y_col}_{Z_col}_{use_embedding}_{target_domain_count}_{hexdigest}'
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
        relevant_columns = {Y_col, Z_col}
        pathology_dtype = CategoricalDtype(categories=(3, 1))
        gender_dtype = CategoricalDtype(categories=("Female", "Male"))
        for column in ("PNEUMONIA", "EFFUSION", "GENDER"):
            if column not in relevant_columns:
                continue

            if column in {"PNEUMONIA", "EFFUSION"}:
                labels[column] = labels[column].astype(pathology_dtype)
            elif column in "GENDER":
                labels[column] = labels[column].astype(gender_dtype)

            nlevels = len(labels[column].dtype.categories)
            if nlevels != 2:
                raise NotImplementedError(f"Column {column} has {nlevels} != 2 levels.")

            labels = labels.loc[~labels[column].isna()]
            labels[column] = labels[column].cat.codes

        self.domains = self.build(generator, datastore, labels, Y_col, Z_col, patient_col, target_domain_count)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print(f'Saving cached datasets to {cache_file}')
        torch.save(self.domains, cache_file)


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
