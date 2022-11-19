from hashlib import sha256

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import torch

from tta.datasets.cxr import MultipleDomainCXR


class MultipleDomainMIMIC(MultipleDomainCXR):

    def __init__(self, root, train_domains, generator, Y_col: str, Z_col: str, use_embedding: bool, target_domain_count: int):
        if len(train_domains) != 1:
            raise NotImplementedError(
                "Training on multiple source distributions is not supported yet."
            )
        if not use_embedding:
            raise NotImplementedError(
                "Using raw images for MIMIC is not supported yet."
            )
        train_domain = next(iter(train_domains))
        patient_col = "subject_id"

        input_shape = (1, 1376)
        C = 2
        K = 2
        confounder_strength = np.linspace(0, 1, 21)

        m = sha256()
        m.update(self.__class__.__name__.encode())
        m.update(str(sorted(train_domains)).encode())
        m.update(generator.get_state().numpy().data.hex().encode())
        m.update(Y_col.encode())
        m.update(Z_col.encode())
        m.update(str(target_domain_count).encode())

        m.update(str(input_shape).encode())
        m.update(str(C).encode())
        m.update(str(K).encode())
        m.update(confounder_strength.data.hex().encode())
        m.update(str(train_domain).encode())
        hexdigest = m.hexdigest()

        super().__init__(input_shape, C, K, confounder_strength, train_domain, hexdigest)

        cache_key = f'{train_domain}_{Y_col}_{Z_col}_{target_domain_count}_{hexdigest}'
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
        self.train_domains = train_domains

        labels: pd.DataFrame = pd.read_csv(root / "mimic_labels_raw.csv", index_col="dicom_id")
        datastore = np.load(root / "mimic.npz")

        #   Pneumonia
        #  0 = negative     - 24303
        # -1 = uncertain    - 19441
        #  1 = positive     - 17222
        #
        #   Pleural Effusion
        #  0 = negative     - 27645
        # -1 = uncertain    -  6202
        #  1 = positive     - 57721
        #
        #   Edema
        #  0 = negative     - 25991
        # -1 = uncertain    - 14244
        #  1 = positive     - 29331
        relevant_columns = {Y_col, Z_col}
        pathology_dtype = CategoricalDtype(categories=(0.0, 1.0))
        for column in ("Pneumonia", "Pleural Effusion", "Edema"):
            if column not in relevant_columns:
                continue

            labels[column] = labels[column].astype(pathology_dtype)
            nlevels = len(labels[column].dtype.categories)
            if nlevels != 2:
                raise NotImplementedError(f"Column {column} has {nlevels} != 2 levels.")

            labels = labels.loc[~labels[column].isna()]
            labels[column] = labels[column].cat.codes

        # Pathology:    0 = Negative, 1 = Positive
        labels["M"] = 2 * labels[Y_col] + labels[Z_col]
        print(f"histogram({Y_col}, {Z_col}) =", labels["M"].value_counts().sort_index().values)

        # joint distribution of Y and Z
        if Y_col == "Pleural Effusion" and Z_col == "Edema":
            # hist(Pleural Effusion, Edema) = [12077  2436  5784 15440]
            anchor1 = np.array([[0.4, 0.0], [0.0, 0.6]])
            anchor2 = np.array([[0.0, 0.3], [0.7, 0.0]])
        elif Y_col == "Pneumonia" and Z_col == "Edema":
            # hist(Pneumonia, Edema) = [11242  1518  1450  3294]
            anchor1 = np.array([[0.75, 0.0], [0.0, 0.25]])
            anchor2 = np.array([[0.0, 0.5], [0.5, 0.0]])
        else:
            raise NotImplementedError(f"Please specify confounders for (Y, Z) = ({Y_col}, {Z_col})")

        self.domains = self.build(generator, datastore, labels, Y_col, Z_col, patient_col, anchor1, anchor2, target_domain_count)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print(f'Saving cached datasets to {cache_file}')
        torch.save(self.domains, cache_file)
