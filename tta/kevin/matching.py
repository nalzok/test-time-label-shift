from typing import Dict
from pathlib import Path

import pandas as pd
import numpy as np


def match(labels: pd.DataFrame, datastore: Dict[str, np.ndarray], output: Path):
    labels = labels.drop(columns=["Unnamed: 0", "patient_id"])

    uniques = {}
    for col in ("split", "GENDER", "PRIMARY_RACE", "ETHNICITY"):
        labels[col], uniques[col] = pd.factorize(labels[col], sort=True)

    N = len(labels.index)
    features = np.empty((N, 1376), dtype=float)
    attributes = np.empty((N, len(labels.columns)), dtype=int)
    for i, (image_id, *image_attributes) in enumerate(labels.itertuples()):
        features[i] = datastore[image_id]
        attributes[i] = image_attributes

    np.savez(
        output,
        features=features,
        attributes=attributes,
        columns=labels.columns,
        uniques=uniques,
    )


if __name__ == "__main__":
    root = Path("data/CheXpert")
    labels = pd.read_csv(root / "labels.csv", index_col="image_id")
    datastore = np.load(root / "embeddings.npz")
    output = Path("data/CheXpert/data_matrix.npz")
    match(labels, datastore, output)
