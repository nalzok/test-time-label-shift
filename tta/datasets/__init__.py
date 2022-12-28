# Forked from https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py

from typing import Set, Tuple, List

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset

from tta.utils import Dataset, split_dataset


class MultipleDomainDataset:
    def __init__(self, input_shape, C, K, confounder_strength, train_domain, hexdigest) -> None:
        super().__init__()

        self.input_shape: Tuple[int] = input_shape
        self.C: int = C
        self.K: int = K
        self.confounder_strength: np.ndarray = confounder_strength
        self.train_domain: int = train_domain
        self.hexdigest: str = hexdigest
        self.domains: List[Tuple[Dataset, torch.Tensor]] = []


def split(dataset: MultipleDomainDataset, train_domains: Set[int], train_fraction: float, train_calibration_fraction: float,
          calibration_domains: Set[int], calibration_fraction: float) \
                  -> Tuple[
                          Tuple[Dataset, torch.Tensor],
                          Tuple[Dataset, torch.Tensor],
                          List[
                              Tuple[Dataset, torch.Tensor]
                          ]
                    ]:
    train_splits = []
    calibration_splits = []
    test_splits = []

    for i, (domain, joint_M) in enumerate(dataset.domains):
        if i in train_domains:
            # For source domains, we split it into train + calibration + test
            train, test = split_dataset(domain, int(len(domain)*train_fraction))
            calibration, train = split_dataset(train, int(len(domain)*train_calibration_fraction))

            train_splits.append(train)
            calibration_splits.append(calibration)
            test_splits.append((test, joint_M))
        elif i in calibration_domains:
            # For calibration domains, we split it into calibration + test
            calibration, test = split_dataset(domain, int(len(domain)*calibration_fraction))

            calibration_splits.append(calibration)
            test_splits.append((test, joint_M))
        else:
            # For target domains, all samples are used as test
            test_splits.append((domain, joint_M))

    joint_shape = dataset.domains[0][1].shape
    if joint_shape != (2, 2):
        raise NotImplementedError(f"(C, K) = {joint_shape} != (2, 2)")

    train = ConcatDataset(train_splits)
    joint_M_train = torch.zeros_like(dataset.domains[0][1])
    for _, _, y, z in train:
        joint_M_train[y][z] += 1
    joint_M_train /= torch.sum(joint_M_train)

    calibration = ConcatDataset(calibration_splits)
    joint_M_calibration = torch.zeros_like(dataset.domains[0][1])
    for _, _, y, z in calibration:
        joint_M_calibration[y][z] += 1
    joint_M_calibration /= torch.sum(joint_M_calibration)

    return (train, joint_M_train), (calibration, joint_M_calibration), test_splits


def subsample(dataset: Dataset, joint_M: torch.Tensor, subsample_what: str, generator: torch.Generator) -> Tuple[Dataset, torch.Tensor]:
    joint_M_count = torch.zeros_like(joint_M, dtype=torch.long)
    M = []
    for _, _, y, z in dataset:
        joint_M_count[y][z] += 1
        m = y * joint_M.shape[-1] + z
        M.append(m)
    M = torch.ByteTensor(M)

    count_per_group = torch.min(joint_M_count).item()
    Y = M // joint_M.shape[-1]
    joint_Y_count = torch.sum(joint_M_count, dim=1)
    count_per_class = torch.min(joint_Y_count).item()

    if subsample_what == "groups":
        indices_list = []
        for m in range(np.prod(joint_M_count.shape)):
            weights = (M == m).float()
            indices_m = torch.multinomial(weights, count_per_group, replacement=False, generator=generator)
            indices_list.extend(indices_m)

    elif subsample_what == "classes":
        indices_list = []
        for y in range(np.prod(joint_Y_count.shape)):
            weights = (Y == y).float()
            indices_y = torch.multinomial(weights, count_per_class, replacement=False, generator=generator)
            indices_list.extend(indices_y)

    else:
        raise ValueError(f"Unknown setting {subsample_what = }")

    subset = Subset(dataset, indices_list)

    joint_M_actual = torch.zeros_like(joint_M_count)
    for _, _, y, z in subset:
        joint_M_actual[y][z] += 1
    joint_Y_actual = torch.sum(joint_M_actual, dim=1)

    # Sanity check
    if subsample_what == "groups":
        joint_M_expected = count_per_group * torch.ones_like(joint_M_count)
        if not torch.allclose(joint_M_actual, joint_M_expected):
            raise ValueError(f"{joint_M_actual = }, {joint_M_expected = }")

    elif subsample_what == "classes":
        joint_Y_expected = count_per_class * torch.ones_like(joint_Y_count)
        if not torch.allclose(joint_Y_actual, joint_Y_expected):
            raise ValueError(f"{joint_Y_actual = }, {joint_Y_expected = }")

    else:
        raise ValueError(f"Unknown setting {subsample_what = }")

    joint_M_actual = joint_M_actual.float() / torch.sum(joint_M_actual)

    return subset, joint_M_actual
