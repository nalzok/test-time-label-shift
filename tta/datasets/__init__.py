# Forked from https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py

from typing import Set, Tuple, List

import numpy as np
from torch.utils.data import ConcatDataset

from ..utils import Dataset, split_dataset


class MultipleDomainDataset:
    def __init__(self, input_shape) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.domains = []


def split(dataset: MultipleDomainDataset, train_domains: Set[int],
          train_fraction: float, calibration_fraction: float,
          rng: np.random.Generator) -> Tuple[ConcatDataset, ConcatDataset, List[Dataset]]:
    train_splits = []
    calibrate_splits = []
    test_splits = []

    for i, (joint, domain) in enumerate(dataset.domains):
        train, test = split_dataset(domain, int(len(domain)*train_fraction), rng)

        if i in train_domains:
            calibrate, train = split_dataset(train, int(len(domain)*calibration_fraction), rng)
            train_splits.append(train)
            calibrate_splits.append(calibrate)

        test_splits.append((joint, test))

    train = ConcatDataset(train_splits)
    calibrate = ConcatDataset(calibrate_splits)

    return train, calibrate, test_splits
