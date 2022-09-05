# Forked from https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py

from typing import Set, Tuple, List

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from ..utils import Dataset, split_dataset


class MultipleDomainDataset:
    def __init__(self, input_shape, C, K) -> None:
        super().__init__()

        self.input_shape: Tuple[int] = input_shape
        self.C: int = C
        self.K: int = K
        self.domains: List[Tuple[torch.Tensor, Dataset]] = []


def split(dataset: MultipleDomainDataset, train_domains: Set[int],
          train_fraction: float, calibration_fraction: float,
          rng: np.random.Generator) -> Tuple[ConcatDataset, ConcatDataset, List[Tuple[torch.Tensor, Dataset]]]:
    train_splits = []
    calibrate_splits = []
    test_splits = []

    for i, (joint, domain) in enumerate(dataset.domains):
        if i in train_domains:
            # For source domains, we split it into train + calibrate + test
            train, test = split_dataset(domain, int(len(domain)*train_fraction), rng)
            calibrate, train = split_dataset(train, int(len(domain)*calibration_fraction), rng)
            train_splits.append(train)
            calibrate_splits.append(calibrate)
            test_splits.append((joint, test))
        else:
            # For target domains, all samples are used as test
            test_splits.append((joint, domain))

    train = ConcatDataset(train_splits)
    calibrate = ConcatDataset(calibrate_splits)

    return train, calibrate, test_splits
