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
import click

from .datasets import ColoredMNIST, RotatedMNIST, split
from .fast_data_loader import InfiniteDataLoader, FastDataLoader 
from .train import create_train_state, train_step, calibration_step, cross_replica_mean, induce_step, adapt_step, test_step
from .utils import Tee


@click.command()
@click.option('--dataset_name', type=click.Choice(['CMNIST', 'RMNIST']), required=True)
@click.option('--train_domains', type=str, required=True)
@click.option('--train_batch_size', type=int, required=True)
@click.option('--train_fraction', type=float, required=True)
@click.option('--train_steps', type=int, required=True)
@click.option('--train_lr', type=float, required=True)
@click.option('--calibration_fraction', type=float, required=True)
@click.option('--calibration_temperature', type=float, required=True)
@click.option('--calibration_steps', type=int, required=True)
@click.option('--test_batch_size', type=int, required=True)
@click.option('--seed', type=int, required=True)
@click.option('--num_workers', type=int, required=True)
@click.option('--log_dir', type=click.Path(path_type=Path), required=True)
def cli(dataset_name: str,
        train_domains: str, train_batch_size: int, train_fraction: float, train_steps: int, train_lr: float,
        calibration_fraction: float, calibration_temperature: float, calibration_steps: int,
        test_batch_size: int,
        seed: int, num_workers: int, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(log_dir / 'out.txt')
    sys.stderr = Tee(log_dir / 'err.txt')

    device_count = jax.local_device_count()
    assert train_batch_size % device_count == 0, f'train_batch_size should be divisible by {device_count}'
    assert test_batch_size % device_count == 0, f'test_batch_size should be divisible by {device_count}'

    train_domains_set = set(int(env) for env in train_domains.split(','))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)

    root = 'data/'
    if dataset_name == 'CMNIST':
        C = 2
        K = 3
        dataset = ColoredMNIST(root)
    elif dataset_name == 'RMNIST':
        C = 10
        K = 6
        dataset = RotatedMNIST(root)
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

    train, calibration, test_splits = split(dataset, train_domains_set, train_fraction, calibration_fraction, rng)

    key_init, key = jax.random.split(key)
    specimen = jnp.empty(dataset.input_shape)
    state = create_train_state(key_init, C, K, calibration_temperature, train_lr, specimen)
    state = replicate(state)


    print('===> Training')
    train_loader = InfiniteDataLoader(train, train_batch_size, num_workers, generator)
    for step, (X, Y, Z) in enumerate(islice(train_loader, train_steps)):
        X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
        Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])
        Z = jnp.array(Z).reshape(device_count, -1, *Z.shape[1:])
        M = Y * K + Z

        state, loss = train_step(state, X, M)
        if step % 100 == 0:
            with jnp.printoptions(precision=3):
                print(f'Train step {step + 1}, loss: {unreplicate(loss)}')


    print('===> Calibrating')
    calibration_loader = InfiniteDataLoader(calibration, train_batch_size, num_workers, generator)
    for step, (X, Y, Z) in enumerate(islice(calibration_loader, calibration_steps)):
        X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
        Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])
        Z = jnp.array(Z).reshape(device_count, -1, *Z.shape[1:])
        M = Y * K + Z

        state, loss = calibration_step(state, X, M)
        if step % 100 == 0:
            with jnp.printoptions(precision=3):
                print(f'Calibration step {step + 1}, loss: {unreplicate(loss)}')

    # Sync the batch statistics across replicas so that evaluation is deterministic.
    state = state.replace(batch_stats=cross_replica_mean(state.batch_stats))


    print('===> Inducing Source Label Prior')
    N = 0
    source_prior = jnp.zeros((device_count, C * K))
    calibration_loader = FastDataLoader(calibration, train_batch_size, num_workers, generator)
    for X, _, _ in calibration_loader:
        remainder = X.shape[0] % device_count
        if remainder != 0:
            X = X[:-remainder]

        N += X.shape[0]
        X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
        source_prior = source_prior + induce_step(state, X)

    params = state.params.unfreeze()
    params['source_prior'] = source_prior / N
    state = state.replace(params=flax.core.frozen_dict.freeze(params))


    print('===> Adapting & Evaluating')
    for i, test in enumerate(test_splits):
        hits = 0
        test_loader = FastDataLoader(test, test_batch_size, num_workers, generator)
        for X, Y, _ in test_loader:
            remainder = X.shape[0] % device_count
            if remainder != 0:
                X = X[:-remainder]
                Y = Y[:-remainder]

            X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
            Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])

            # Adapation (online)
            state = adapt_step(state, X)

            # Evaluation
            hits += unreplicate(test_step(state, X, Y))

        accuracy = hits/len(test)
        with jnp.printoptions(precision=3):
            print(f'Environment {i}: test accuracy {accuracy*100}%')


if __name__ == '__main__':
    initialize_cache('jit_cache')
    cli()
