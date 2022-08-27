from pathlib import Path
import sys
import random
from itertools import islice

import jax
import jax.numpy as jnp
from jax.experimental.compilation_cache.compilation_cache import initialize_cache
import flax
from flax.jax_utils import replicate, unreplicate
import numpy as np
import torch
from torch.utils.data import Dataset
import click

from .datasets import split
from .datasets.mnist import ColoredMNIST, RotatedMNIST
from .datasets.coco import ColoredCOCO
from .fast_data_loader import InfiniteDataLoader, FastDataLoader 
from .train import (
    TrainState,
    create_train_state,
    train_step,
    calibration_step,
    cross_replica_mean,
    induce_step,
    adapt_step,
    test_step
)
from .utils import Tee


@click.command()
@click.option('--dataset_name', type=click.Choice(['CMNIST', 'RMNIST', 'CCOCO']), required=True)
@click.option('--train_domains', type=str, required=True)
@click.option('--train_batch_size', type=int, required=True)
@click.option('--train_fraction', type=float, required=True)
@click.option('--train_steps', type=int, required=True)
@click.option('--train_lr', type=float, required=True)
@click.option('--source_prior_estimation', type=click.Choice(['count', 'induce', 'average']), required=True)
@click.option('--calibration_batch_size', type=int, required=True)
@click.option('--calibration_fraction', type=float, required=True)
@click.option('--calibration_temperature', type=float, required=True)
@click.option('--calibration_steps', type=int, required=True)
@click.option('--calibration_multiplier', type=float, required=True)
@click.option('--test_batch_size', type=int, required=True)
@click.option('--seed', type=int, required=True)
@click.option('--num_workers', type=int, required=True)
@click.option('--log_dir', type=click.Path(path_type=Path), required=True)
def cli(dataset_name: str,
        train_domains: str, train_batch_size: int, train_fraction: float, train_steps: int, train_lr: float,
        source_prior_estimation: str, calibration_batch_size: int, calibration_fraction: float,
        calibration_temperature: float, calibration_steps: int, calibration_multiplier: float,
        test_batch_size: int, seed: int, num_workers: int, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(log_dir / 'out.txt')
    sys.stderr = Tee(log_dir / 'err.txt')

    device_count = jax.local_device_count()
    assert train_batch_size % device_count == 0, f'train_batch_size should be divisible by {device_count}'
    assert calibration_batch_size % device_count == 0, f'calibration_batch_size should be divisible by {device_count}'
    assert test_batch_size % device_count == 0, f'test_batch_size should be divisible by {device_count}'

    train_domains_set = set(int(env) for env in train_domains.split(','))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)

    if dataset_name == 'CMNIST':
        C = 2
        K = 2
        root = Path('data/')
        dataset = ColoredMNIST(root, generator)
    elif dataset_name == 'RMNIST':
        C = 10
        K = 6
        root = Path('data/')
        dataset = RotatedMNIST(root, generator)
    elif dataset_name == 'CCOCO':
        C = 9
        K = 9
        root = Path('data/COCO/train2017')
        annFile = Path('data/COCO/annotations/instances_train2017.json')
        dataset = ColoredCOCO(root, annFile, generator)
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

    train, calibration, test_splits = split(dataset, train_domains_set, train_fraction, calibration_fraction, rng)
    print('train', len(train))
    print('calibration', len(calibration))
    print('test_splits', len(test_splits[0]))

    key_init, key = jax.random.split(key)
    specimen = jnp.empty(dataset.input_shape)
    state = create_train_state(key_init, C, K, calibration_temperature, train_lr, specimen)
    state = replicate(state)


    print('===> Training')
    inf_train_loader = InfiniteDataLoader(train, train_batch_size, num_workers, generator)
    for step, (X, Y, Z) in enumerate(islice(inf_train_loader, train_steps)):
        X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
        Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])
        Z = jnp.array(Z).reshape(device_count, -1, *Z.shape[1:])
        M = Y * K + Z

        state, loss = train_step(state, X, M)
        if step % (train_steps // 20) == 0:
            with jnp.printoptions(precision=3):
                print(f'Train step {step + 1}, loss: {unreplicate(loss)}')


    print('===> Calibrating')
    inf_calibration_loader = InfiniteDataLoader(calibration, calibration_batch_size, num_workers, generator)
    for step, (X, Y, Z) in enumerate(islice(inf_calibration_loader, calibration_steps)):
        X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
        Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])
        Z = jnp.array(Z).reshape(device_count, -1, *Z.shape[1:])
        M = Y * K + Z

        state, loss = calibration_step(state, X, M, calibration_multiplier)
        if step % (calibration_steps // 20) == 0:
            with jnp.printoptions(precision=3):
                print(f'Calibration step {step + 1}, loss: {unreplicate(loss)}')

    # Sync the batch statistics across replicas so that evaluation is deterministic.
    state = state.replace(batch_stats=cross_replica_mean(state.batch_stats))


    print('===> Estimating Source Label Prior')
    if source_prior_estimation == 'average':
        source_prior_count = estimate_source_prior(train, train_batch_size, num_workers, generator,
                              C, K, device_count, state, 'count')
        source_prior_induce = estimate_source_prior(train, train_batch_size, num_workers, generator,
                              C, K, device_count, state, 'induce')
        source_prior = (source_prior_count + source_prior_induce) / 2
    else:
        source_prior = estimate_source_prior(train, train_batch_size, num_workers, generator,
                              C, K, device_count, state, source_prior_estimation)

    prior = state.prior.unfreeze()
    prior['source'] = source_prior
    state = state.replace(prior=flax.core.frozen_dict.freeze(prior))


    print('===> Adapting & Evaluating')
    for i, test in enumerate(test_splits):
        source_hits = 0
        indep_hits = 0
        uniform_hits = 0
        adapted_hits = 0
        test_loader = FastDataLoader(test, test_batch_size, num_workers, generator)
        for X, Y, _ in test_loader:
            remainder = X.shape[0] % device_count
            if remainder != 0:
                X = X[:-remainder]
                Y = Y[:-remainder]

            X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
            Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])

            # Source
            prior = state.prior.unfreeze()
            prior['target'] = prior['source']
            state = state.replace(prior=flax.core.frozen_dict.freeze(prior))

            source_hits += unreplicate(test_step(state, X, Y))

            # Independent
            prior = state.prior.unfreeze()
            joint_source = unreplicate(prior['source']).reshape((C, K))
            marginal_y = jnp.sum(joint_source, axis=1)
            marginal_z = jnp.sum(joint_source, axis=0)
            joint_target = jnp.outer(marginal_y, marginal_z)
            prior['target'] = replicate(joint_target.flatten())
            state = state.replace(prior=flax.core.frozen_dict.freeze(prior))

            indep_hits += unreplicate(test_step(state, X, Y))

            # Uniform
            prior = state.prior.unfreeze()
            prior['target'] = jnp.ones_like(prior['target']) / K / C
            state = state.replace(prior=flax.core.frozen_dict.freeze(prior))

            uniform_hits += unreplicate(test_step(state, X, Y))

            # Adaptation
            state = adapt_step(state, X)

            adapted_hits += unreplicate(test_step(state, X, Y))

        source_accuracy = source_hits/len(test)
        indep_accuracy = indep_hits/len(test)
        uniform_accuracy = uniform_hits/len(test)
        adapted_accuracy = adapted_hits/len(test)
        with jnp.printoptions(precision=3):
            print(f'Environment {i}: test accuracy {source_accuracy:.4f} (source), {indep_accuracy:.4f} (independent), '
                  f'{uniform_accuracy:.4f} (uniform), {adapted_accuracy:.4f} (adapted)')


def estimate_source_prior(train: Dataset, train_batch_size: int, num_workers: int, generator: torch.Generator,
                          C: int, K: int, device_count: int, state: TrainState, method: str) -> jnp.ndarray:
    train_loader = FastDataLoader(train, train_batch_size, num_workers, generator)
    if method == 'count':
        source_prior = np.zeros((C * K))
        I = np.identity(C * K)
        for _, Y, Z in train_loader:
            M = Y * K + Z
            source_prior += np.sum(I[M], axis=0)

        source_prior = replicate(jnp.array(source_prior / np.sum(source_prior)))

    elif method == 'induce':
        N = 0
        source_prior = jnp.zeros((device_count, C * K))
        for X, _, _ in train_loader:
            remainder = X.shape[0] % device_count
            if remainder != 0:
                X = X[:-remainder]

            N += X.shape[0]
            X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
            source_prior = source_prior + induce_step(state, X)

        source_prior = source_prior / N

    else:
        raise ValueError(f'Unknown source label prior estimation method {method}')

    return source_prior


if __name__ == '__main__':
    initialize_cache('jit_cache')
    cli()
