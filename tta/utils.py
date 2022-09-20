from typing import Tuple
import sys

import torch
from torch.utils.data import Dataset, random_split


class Dataset(Dataset):
    def __len__(self):
        raise NotImplementedError


class Tee:
    def __init__(self, fname, mode="w"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def split_dataset(dataset: Dataset, n: int) -> Tuple[Dataset, Dataset]:
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert 0 <= n <= len(dataset)

    generator = torch.Generator().manual_seed(2022)
    first, second = random_split(dataset, (n, len(dataset) - n), generator)

    return first, second
