# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import random

import numpy as np
import torch
from torch.utils.data import Sampler, RandomSampler, BatchSampler, DataLoader


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class _InfiniteSampler(Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, batch_size: int, num_workers: int, generator: torch.Generator):
        sampler = RandomSampler(dataset, replacement=True, generator=generator)

        batch_sampler = BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True
        )

        self._infinite_iterator = iter(DataLoader(
            dataset,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError('Infinite DataLoaders are infinite')


class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""
    def __init__(self, dataset, batch_size: int, num_workers: int, generator: torch.Generator):
        sampler = RandomSampler(dataset, replacement=True, generator=generator)

        batch_sampler = BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=False
        )

        self._infinite_iterator = iter(DataLoader(
            dataset,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length

