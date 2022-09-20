# Forked from https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py

from typing import Set, Tuple, List

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from ..utils import Dataset, split_dataset


class MultipleDomainDataset:
    def __init__(self, input_shape, C, K, environments) -> None:
        super().__init__()

        self.input_shape: Tuple[int] = input_shape
        self.C: int = C
        self.K: int = K
        self.environments: np.ndarray = environments
        self.domains: List[Tuple[torch.Tensor, Dataset]] = []


def split(dataset: MultipleDomainDataset, train_domains: Set[int], train_fraction: float, train_calibration_fraction: float,
          calibration_domains: Set[int], calibration_fraction: float) \
                  -> Tuple[ConcatDataset, ConcatDataset, List[Tuple[torch.Tensor, Dataset]]]:
    train_splits = []
    calibration_splits = []
    test_splits = []

    for i, (joint, domain) in enumerate(dataset.domains):
        if i in train_domains:
            # For source domains, we split it into train + calibrate + test
            train, test = split_dataset(domain, int(len(domain)*train_fraction))
            calibration, train = split_dataset(train, int(len(domain)*train_calibration_fraction))

            train_splits.append(train)
            calibration_splits.append(calibration)
            test_splits.append((joint, test))
        elif i in calibration_domains:
            # For calibration domains, we split it into calibrate + test
            calibration, test = split_dataset(domain, int(len(domain)*calibration_fraction))

            calibration_splits.append(calibration)
            test_splits.append((joint, test))
        else:
            # For target domains, all samples are used as test
            test_splits.append((joint, domain))

    train = ConcatDataset(train_splits)
    calibrate = ConcatDataset(calibration_splits)

    return train, calibrate, test_splits
