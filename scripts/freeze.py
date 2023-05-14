from typing import Tuple, Set
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

from tta.utils import Dataset
from tta.datasets.mnist import MultipleDomainMNIST
from tta.datasets.cxr.chexpert import MultipleDomainCheXpert
from tta.datasets import MultipleDomainDataset, split


@click.command()
@click.option("--seed", type=int, required=True)
def main(seed: int):
    for getter in [get_mnist, get_chexpert]:
        name, train_domains_set, dataset = getter(seed)

        train_fraction = 0.9
        train_calibration_fraction = 0.1
        calibration_domains_set = set()
        calibration_fraction = 0.0

        (train, _), (calibration, _), test_splits = split(
            dataset,
            train_domains_set,
            train_fraction,
            train_calibration_fraction,
            calibration_domains_set,
            calibration_fraction,
        )

        splits = [(0, train), (1, calibration)]
        for i, (test, _) in enumerate(test_splits):
            splits.append((i + 2, test))

        metadata_split = []
        for split_id, ds in splits:
            X, Y, _, Z = dataset2np(ds)
            filenames = []
            for i, embedding in enumerate(X):
                path = Path("frozen") / name / f"{i}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                filenames.append(path.name)
                np.save(path, embedding)

            split_ = np.ones_like(Y) * split_id
            meta = pd.DataFrame.from_dict({
                "filename": filenames,
                "split": split_,
                "y": Y,
                "a": Z
            })
            metadata_split.append(meta)
            
        metadata = pd.concat(metadata_split)
        metadata.index.name = "id"

        metadata_path = Path("frozen") / f"{name}.csv"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(metadata_path)



def get_mnist(seed) -> Tuple[str, Set[int], MultipleDomainDataset]:
    generator = torch.Generator().manual_seed(seed)

    train_domain = 1
    train_domains_set = {train_domain}
    dataset_apply_rotation = False
    dataset_feature_noise = 0
    dataset_label_noise = 0

    root = Path("data/mnist")
    dataset = MultipleDomainMNIST(
        root,
        train_domains_set,
        generator,
        dataset_apply_rotation,
        dataset_feature_noise,
        dataset_label_noise,
    )

    name = f"mnist_rot{dataset_apply_rotation}_noise{dataset_label_noise}_domain{train_domain}_seed{seed}"

    return name, train_domains_set, dataset


def get_chexpert(seed) -> Tuple[str, Set[int], MultipleDomainDataset]:
    generator = torch.Generator().manual_seed(seed)

    train_domain = 1
    train_domains_set = {train_domain}
    dataset_y_column = "EFFUSION"
    dataset_z_column = "GENDER"
    dataset_target_domain_count = 512
    dataset_source_domain_count = 65536
    dataset_use_embedding = True

    root = Path("data/CheXpert")
    dataset = MultipleDomainCheXpert(
        root,
        train_domains_set,
        generator,
        dataset_y_column,
        dataset_z_column,
        dataset_use_embedding,
        dataset_target_domain_count,
        dataset_source_domain_count,
    )

    name = f"chexpert-embedding_{dataset_y_column}_{dataset_z_column}_domain{train_domain}_size{dataset_source_domain_count}_seed{seed}"

    return name, train_domains_set, dataset


def dataset2np(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, Y, Y_tilde, Z = [], [], [], []
    for x, y_tilde, y, z_flattened in dataset:
        X.append(x)
        Y.append(y)
        Y_tilde.append(y_tilde)
        Z.append(z_flattened)

    X = np.stack(X)
    Y = np.stack(Y)
    Y_tilde = np.stack(Y_tilde)
    Z = np.stack(Z)

    return X, Y, Y_tilde, Z


if __name__ == "__main__":
    main()
