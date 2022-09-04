from typing import Any, Sequence, List, Tuple, Set, Dict
from pathlib import Path
import sys
import random
from itertools import islice, product
from pprint import pprint

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
from .datasets.mnist import MultipleDomainMNIST
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
@click.option('--dataset_name', type=click.Choice(['MNIST', 'COCO']), required=True)
@click.option('--train_domains', type=str, required=True)
@click.option('--train_batch_size', type=int, required=True)
@click.option('--train_fraction', type=float, required=True)
@click.option('--train_num_layers', type=int, required=True)
@click.option('--train_steps', type=int, required=True)
@click.option('--train_lr', type=float, required=True)
@click.option('--source_prior_estimation', type=click.Choice(['count', 'induce', 'average']), required=True)
@click.option('--calibration_batch_size', type=int, required=True)
@click.option('--calibration_fraction', type=float, required=True)
@click.option('--calibration_temperature', type=float, required=True)
@click.option('--calibration_steps', type=int, required=True)
@click.option('--calibration_multiplier', type=float, required=True)
@click.option('--test_batch_size', type=int, required=True, multiple=True)
@click.option('--test_symmetric_dirichlet', type=bool, required=True, multiple=True)
@click.option('--test_pseudocount_factor', type=float, required=True, multiple=True)
@click.option('--test_fix_marginal', type=bool, required=True, multiple=True)
@click.option('--seed', type=int, required=True)
@click.option('--num_workers', type=int, required=True)
@click.option('--log_dir', type=click.Path(path_type=Path), required=True)
def cli(dataset_name: str, train_domains: str,
        train_batch_size: int, train_fraction: float, train_num_layers: int, train_steps: int, train_lr: float,
        source_prior_estimation: str, calibration_batch_size: int, calibration_fraction: float,
        calibration_temperature: float, calibration_steps: int, calibration_multiplier: float,
        test_batch_size: Sequence[int], test_symmetric_dirichlet: Sequence[bool],
        test_pseudocount_factor: Sequence[float], test_fix_marginal: Sequence[bool],
        seed: int, num_workers: int, log_dir: Path) -> None:
    main(dataset_name, train_domains,
            train_batch_size, train_fraction, train_num_layers, train_steps, train_lr,
            source_prior_estimation, calibration_batch_size, calibration_fraction,
            calibration_temperature, calibration_steps, calibration_multiplier,
            test_batch_size, test_symmetric_dirichlet, test_pseudocount_factor, test_fix_marginal,
            seed, num_workers, log_dir)


def main(dataset_name: str, train_domains: str,
        train_batch_size: int, train_fraction: float, train_num_layers: int, train_steps: int, train_lr: float,
        source_prior_estimation: str, calibration_batch_size: int, calibration_fraction: float,
        calibration_temperature: float, calibration_steps: int, calibration_multiplier: float,
        test_batch_size: Sequence[int], test_symmetric_dirichlet: Sequence[bool],
        test_pseudocount_factor: Sequence[float], test_fix_marginal: Sequence[bool],
        seed: int, num_workers: int, log_dir: Path) -> Dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(log_dir / 'out.txt')
    sys.stderr = Tee(log_dir / 'err.txt')

    device_count = jax.local_device_count()
    assert train_batch_size % device_count == 0, f'train_batch_size should be divisible by {device_count}'
    assert calibration_batch_size % device_count == 0, f'calibration_batch_size should be divisible by {device_count}'
    for batch_size in test_batch_size:
        assert batch_size % device_count == 0, f'test_batch_size should be divisible by {device_count}'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)

    train_domains_set = set(int(env) for env in train_domains.split(','))

    state, C, K, test_splits, unadapted_accuracy_curve, unadapted_norm_curve = train(dataset_name, train_domains_set,
            train_batch_size, train_fraction, train_num_layers, train_steps, train_lr,
            source_prior_estimation, calibration_batch_size, calibration_fraction,
            calibration_temperature, calibration_steps, calibration_multiplier,
            device_count, key, rng, generator, num_workers)
    results = {None: (unadapted_accuracy_curve, unadapted_norm_curve)}

    for batch_size, symmetric_dirichlet, pseudocount_factor, fix_marginal \
            in product(test_batch_size, test_symmetric_dirichlet, test_pseudocount_factor, test_fix_marginal):
        accuracy_curve, norm_curve = test(state, C, K, train_domains_set, test_splits,
                batch_size, symmetric_dirichlet, pseudocount_factor, fix_marginal,
                device_count, generator, num_workers)
        results[batch_size, symmetric_dirichlet, pseudocount_factor, fix_marginal] = accuracy_curve, norm_curve

    pprint(results)

    return results


def train(dataset_name: str, train_domains_set: Set[int],
        train_batch_size: int, train_fraction: float, train_num_layers: int, train_steps: int, train_lr: float,
        source_prior_estimation: str, calibration_batch_size: int, calibration_fraction: float,
        calibration_temperature: float, calibration_steps: int, calibration_multiplier: float,
        device_count: int, key: Any, rng: np.random.Generator, generator: torch.Generator, num_workers: int) \
                -> Tuple[TrainState, int, int, List[Tuple[torch.Tensor, Dataset]], jnp.ndarray, jnp.ndarray]:
    if dataset_name == 'MNIST':
        root = Path('data/')
        dataset = MultipleDomainMNIST(root, generator)
    elif dataset_name == 'COCO':
        root = Path('data/COCO/train2017')
        annFile = Path('data/COCO/annotations/instances_train2017.json')
        dataset = ColoredCOCO(root, annFile, generator)
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

    C, K = dataset.C, dataset.K

    train, calibration, test_splits = split(dataset, train_domains_set, train_fraction, calibration_fraction, rng)
    print('train', len(train))
    print('calibration', len(calibration))
    print('test_splits[0][1]', len(test_splits[0][1]))

    key_init, key = jax.random.split(key)
    specimen = jnp.empty(dataset.input_shape)
    state = create_train_state(key_init, C, K, calibration_temperature, train_num_layers, train_lr, specimen)
    state: TrainState = replicate(state)


    print('===> Training')
    inf_train_loader = InfiniteDataLoader(train, train_batch_size, num_workers, generator)
    for step, (X, Y, Z) in enumerate(islice(inf_train_loader, train_steps)):
        X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
        Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])
        Z = jnp.array(Z).reshape(device_count, -1, *Z.shape[1:])
        M = Y * K + Z

        state, loss = train_step(state, X, M)
        if step % (train_steps // 20 + 1) == 0:
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
        if step % (calibration_steps // 20 + 1) == 0:
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

    print('---> Unadapted')
    prior = state.prior.unfreeze()
    prior['target'] = prior['source']
    state = state.replace(prior=flax.core.frozen_dict.freeze(prior))

    accuracy_curve = jnp.empty(len(test_splits))
    norm_curve = jnp.empty(len(test_splits))
    for i, (joint, test) in enumerate(test_splits):
        seen = '  seen' if i in train_domains_set else 'unseen'

        hits = norm = 0
        joint = jnp.array(joint)
        # batch size should not matter since we are not doing adaptation
        test_loader = FastDataLoader(test, train_batch_size, num_workers, generator)
        for X, Y, _ in test_loader:
            remainder = X.shape[0] % device_count
            if remainder != 0:
                X = X[:-remainder]
                Y = Y[:-remainder]

            X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
            Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])

            hits += unreplicate(test_step(state, X, Y))
            prior = unreplicate(state.prior['target']).reshape((C, K))
            norm += jnp.linalg.norm(prior - joint)

        accuracy = hits/len(test)
        accuracy_curve = accuracy_curve.at[i].set(accuracy)
        norm = norm/len(test)
        norm_curve = norm_curve.at[i].set(norm)

        with jnp.printoptions(precision=4):
            print(f'Environment {i:>2} ({seen}) Accuracy {accuracy}, Norm {norm}')

    return state, C, K, test_splits, accuracy_curve, norm_curve


def test(state: TrainState, C: int, K: int, train_domains_set: Set[int],
        test_splits: Sequence[Tuple[torch.Tensor, Dataset]],
        batch_size: int, symmetric_dirichlet: bool, pseudocount_factor: float, fix_marginal: bool,
        device_count: int, generator: torch.Generator, num_workers: int) -> Tuple[jnp.ndarray, jnp.ndarray]:

    print(f'---> {batch_size=}, {symmetric_dirichlet=}, {pseudocount_factor=}, {fix_marginal=}')
    accuracy_curve = jnp.empty(len(test_splits))
    norm_curve = jnp.empty(len(test_splits))
    for i, (joint, test) in enumerate(test_splits):
        seen = '  seen' if i in train_domains_set else 'unseen'

        hits = norm = 0
        joint = jnp.array(joint)
        test_loader = FastDataLoader(test, batch_size, num_workers, generator)
        for X, Y, _ in test_loader:
            remainder = X.shape[0] % device_count
            if remainder != 0:
                X = X[:-remainder]
                Y = Y[:-remainder]

            X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
            Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])

            # Adaptation
            state = adapt_step(state, X, C, K, symmetric_dirichlet, pseudocount_factor, fix_marginal)

            hits += unreplicate(test_step(state, X, Y))
            prior = unreplicate(state.prior['target']).reshape((C, K))
            norm += jnp.linalg.norm(prior - joint)

        accuracy = hits/len(test)
        accuracy_curve = accuracy_curve.at[i].set(accuracy)
        norm = norm/len(test)
        norm_curve = norm_curve.at[i].set(norm)

        with jnp.printoptions(precision=4):
            print(f'Environment {i:>2} ({seen}) Accuracy {accuracy}, Norm {norm}')

    return accuracy_curve, norm_curve


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
