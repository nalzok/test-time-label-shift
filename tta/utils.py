from typing import Tuple
import sys

import numpy as np
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __len__(self):
        raise NotImplementedError


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


class _SplitDataset(Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys

    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)

    def __iter__(self):
        return self.underlying_dataset[self.keys]


def split_dataset(dataset: Dataset, n: int, rng: np.random.Generator) -> Tuple[Dataset, Dataset]:
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert 0 <= n <= len(dataset)
    keys = rng.permutation(len(dataset))
    return _SplitDataset(dataset, keys[:n]), _SplitDataset(dataset, keys[n:])
