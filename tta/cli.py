from typing import Any, Sequence, List, Tuple, Set, Dict, Optional
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
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from .datasets import split
from .datasets.mnist import MultipleDomainMNIST
from .datasets.waterbirds import MultipleDomainWaterbirds
from .datasets.coco import ColoredCOCO
from .datasets.fast_data_loader import InfiniteDataLoader, FastDataLoader
from .train import (
    TrainState,
    create_train_state,
    restore_train_state,
    train_step,
    calibration_step,
    cross_replica_mean,
    induce_step,
    adapt_step,
    test_step
)
from .utils import Tee


Curves = Dict[Tuple[str, Tuple[Optional[int], Optional[bool], Optional[float], Optional[bool]]], jnp.ndarray]


@click.command()
@click.option('--config_name', type=str, required=True)
@click.option('--dataset_name', type=click.Choice(['MNIST', 'COCO', 'Waterbirds']), required=True)
@click.option('--train_domains', type=str, required=True)
@click.option('--train_apply_rotation', type=bool, required=True)
@click.option('--train_label_noise', type=float, required=True)
@click.option('--train_batch_size', type=int, required=True)
@click.option('--train_fraction', type=float, required=True)
@click.option('--train_model', type=str, required=True)
@click.option('--train_checkpoint_path', type=click.Path(path_type=Path), required=False)
@click.option('--train_steps', type=int, required=True)
@click.option('--train_lr', type=float, required=True)
@click.option('--source_prior_estimation', type=click.Choice(['count', 'induce', 'average']), required=True)
@click.option('--calibration_batch_size', type=int, required=True)
@click.option('--calibration_fraction', type=float, required=True)
@click.option('--calibration_temperature', type=float, required=True)
@click.option('--calibration_steps', type=int, required=True)
@click.option('--calibration_lr', type=float, required=True)
@click.option('--test_batch_size', type=int, required=True, multiple=True)
@click.option('--test_symmetric_dirichlet', type=bool, required=True, multiple=True)
@click.option('--test_prior_strength', type=float, required=True, multiple=True)
@click.option('--test_fix_marginal', type=bool, required=True, multiple=True)
@click.option('--plot_title', type=str, required=False, default='Performance on Each Domain')
@click.option('--seed', type=int, required=True)
@click.option('--num_workers', type=int, required=True)
def cli(config_name: str, dataset_name: str, train_domains: str, train_apply_rotation: bool,
        train_label_noise: float, train_batch_size: int, train_fraction: float, train_model: str,
        train_checkpoint_path: Optional[Path], train_steps: int, train_lr: float,
        source_prior_estimation: str, calibration_batch_size: int, calibration_fraction: float,
        calibration_temperature: float, calibration_steps: int, calibration_lr: float,
        test_batch_size: Sequence[int], test_symmetric_dirichlet: Sequence[bool],
        test_prior_strength: Sequence[float], test_fix_marginal: Sequence[bool],
        plot_title: str, seed: int, num_workers: int) -> None:
    main(config_name, dataset_name, train_domains, train_apply_rotation, train_label_noise,
            train_batch_size, train_fraction, train_model, train_checkpoint_path, train_steps, train_lr,
            source_prior_estimation, calibration_batch_size, calibration_fraction,
            calibration_temperature, calibration_steps, calibration_lr,
            test_batch_size, test_symmetric_dirichlet, test_prior_strength, test_fix_marginal,
            plot_title, seed, num_workers)


def main(config_name: str, dataset_name: str, train_domains: str, train_apply_rotation: bool,
        train_label_noise: float, train_batch_size: int, train_fraction: float, train_model: str,
        train_checkpoint_path: Optional[Path], train_steps: int, train_lr: float,
        source_prior_estimation: str, calibration_batch_size: int, calibration_fraction: float,
        calibration_temperature: float, calibration_steps: int, calibration_lr: float,
        test_batch_size: Sequence[int], test_symmetric_dirichlet: Sequence[bool],
        test_prior_strength: Sequence[float], test_fix_marginal: Sequence[bool],
        plot_title: str, seed: int, num_workers: int) -> Tuple[Curves, Curves, Curves]:
    log_root = Path('logs/')
    plot_root = Path('plots/')
    npz_root = Path('npz/')

    log_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)
    npz_root.mkdir(parents=True, exist_ok=True)

    log_path = log_root / f'{config_name}.txt'
    sys.stdout = Tee(log_path)

    auc_plot_path = plot_root / f'{config_name}_auc.png'
    accuracy_plot_path = plot_root / f'{config_name}_accuracy.png'
    norm_plot_path = plot_root / f'{config_name}_norm.png'
    auc_npz_path = npz_root / f'{config_name}_auc.npz'
    accuracy_npz_path = npz_root / f'{config_name}_accuracy.npz'
    norm_npz_path = npz_root / f'{config_name}_norm.npz'

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
    if len(train_domains_set) != 1:
        raise NotImplementedError('Training on multiple source distributions is not supported yet.')

    state, C, K, environments, test_splits, unadapted_auc_sweep, unadapted_accuracy_sweep, unadapted_norm_sweep \
            = train(dataset_name, train_domains_set, train_apply_rotation, train_label_noise,
                    train_batch_size, train_fraction, train_model, train_checkpoint_path, train_steps, train_lr,
                    source_prior_estimation, calibration_batch_size, calibration_fraction,
                    calibration_temperature, calibration_steps, calibration_lr,
                    device_count, key, rng, generator, num_workers)
    auc_sweeps: Curves = {('Unadapted', (None, None, None, None)): unadapted_auc_sweep}
    accuracy_sweeps: Curves = {('Unadapted', (None, None, None, None)): unadapted_accuracy_sweep}
    norm_sweeps: Curves = {('Unadapted', (None, None, None, None)): unadapted_norm_sweep}

    for batch_size, symmetric_dirichlet, prior_strength, fix_marginal \
            in product(test_batch_size, test_symmetric_dirichlet, test_prior_strength, test_fix_marginal):
        label = f'{batch_size=}, {symmetric_dirichlet=}, {prior_strength=}, {fix_marginal=}'
        state, auc_sweep, accuracy_sweep, norm_sweep = adapt(state, C, K, train_domains_set, test_splits,
                batch_size, symmetric_dirichlet, prior_strength, fix_marginal, label,
                device_count, generator, num_workers)
        key = label, (batch_size, symmetric_dirichlet, prior_strength, fix_marginal)
        auc_sweeps[key] = auc_sweep
        accuracy_sweeps[key] = accuracy_sweep
        norm_sweeps[key] = norm_sweep

    print('===> auc_sweeps')
    pprint(auc_sweeps)
    print('===> accuracy_sweeps')
    pprint(accuracy_sweeps)
    print('===> norm_sweeps')
    pprint(norm_sweeps)

    jnp.savez(auc_npz_path, **{k: v for (k, _), v in auc_sweeps.items()})
    jnp.savez(accuracy_npz_path, **{k: v for (k, _), v in accuracy_sweeps.items()})
    jnp.savez(norm_npz_path, **{k: v for (k, _), v in norm_sweeps.items()})

    plot_auc(auc_sweeps, environments, train_domains_set, plot_title, auc_plot_path)
    plot_accuracy(accuracy_sweeps, environments, train_domains_set, train_label_noise, plot_title, accuracy_plot_path)
    plot_norm(norm_sweeps, environments, train_domains_set, plot_title, norm_plot_path)

    return auc_sweeps, accuracy_sweeps, norm_sweeps


def train(dataset_name: str, train_domains_set: Set[int], train_apply_rotation: bool,
        train_label_noise: float, train_batch_size: int, train_fraction: float, train_model: str,
        train_checkpoint_path: Optional[Path], train_steps: int, train_lr: float,
        source_prior_estimation: str, calibration_batch_size: int, calibration_fraction: float,
        calibration_temperature: float, calibration_steps: int, calibration_lr: float,
        device_count: int, key: Any, rng: np.random.Generator, generator: torch.Generator, num_workers: int) \
                -> Tuple[TrainState, int, int, np.ndarray, List[Tuple[torch.Tensor, Dataset]], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if dataset_name == 'MNIST':
        root = Path('data/')
        dataset = MultipleDomainMNIST(root, generator, train_domains_set, train_apply_rotation, train_label_noise)
    elif dataset_name == 'COCO':
        root = Path('data/COCO/train2017')
        annFile = Path('data/COCO/annotations/instances_train2017.json')
        dataset = ColoredCOCO(root, annFile, generator)
    elif dataset_name == 'Waterbirds':
        root = Path('data/')
        dataset = MultipleDomainWaterbirds(root, generator, train_domains_set)
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

    C, K = dataset.C, dataset.K
    if C != 2:
        raise NotImplementedError('Multi-label classification is not supported yet.')

    train, calibration, test_splits = split(dataset, train_domains_set, train_fraction, calibration_fraction, rng)
    print('train', len(train))
    print('calibration', len(calibration))
    train_domain = next(iter(train_domains_set))
    print(f'test_splits[{train_domain}][1]', len(test_splits[train_domain][1]))

    key_init, key = jax.random.split(key)
    specimen = jnp.empty(dataset.input_shape)
    state = create_train_state(key_init, C, K, calibration_temperature, train_model, train_lr, specimen, device_count)
    if train_checkpoint_path is not None:
        state = restore_train_state(state, train_checkpoint_path)
    state: TrainState = replicate(state)


    print('===> Training')
    inf_train_loader = InfiniteDataLoader(train, train_batch_size, num_workers, generator)
    for step, (X, Y, Z) in enumerate(islice(inf_train_loader, train_steps)):
        X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
        Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])
        Z = jnp.array(Z).reshape(device_count, -1, *Z.shape[1:])
        M = Y * K + Z

        state, loss = train_step(state, X, M)
        if step % (train_steps // 21 + 1) == 0:
            with jnp.printoptions(precision=3):
                print(f'Train step {step + 1}, loss: {unreplicate(loss)}')

    # Sync the batch statistics across replicas so that evaluation is deterministic.
    state = state.replace(batch_stats=cross_replica_mean(state.batch_stats))


    print('===> Calibrating')
    if len(calibration) or calibration_steps:
        inf_calibration_loader = InfiniteDataLoader(calibration, calibration_batch_size, num_workers, generator)
        for step, (X, Y, Z) in enumerate(islice(inf_calibration_loader, calibration_steps)):
            X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
            Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])
            Z = jnp.array(Z).reshape(device_count, -1, *Z.shape[1:])
            M = Y * K + Z

            state, loss = calibration_step(state, X, M, calibration_lr)
            if step % (calibration_steps // 21 + 1) == 0:
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

    label = 'Unadapted'
    print(f'---> {label}')
    prior = state.prior.unfreeze()
    prior['target'] = prior['source']
    state = state.replace(prior=flax.core.frozen_dict.freeze(prior))

    auc_sweep = jnp.empty(len(test_splits))
    accuracy_sweep = jnp.empty(len(test_splits))
    norm_sweep = jnp.empty(len(test_splits))
    for i, (joint, test) in enumerate(test_splits):
        # happens on the source domain when train_fraction = 1
        if len(test) == 0:
            auc_sweep = auc_sweep.at[i].set(jnp.nan)
            accuracy_sweep = accuracy_sweep.at[i].set(jnp.nan)
            norm_sweep = norm_sweep.at[i].set(jnp.nan)
            continue

        seen = '  (seen)' if i in train_domains_set else '(unseen)'

        auc = hits = norm = 0
        joint = jnp.array(joint)

        # batch size should not matter since we are not doing adaptation
        test_loader = FastDataLoader(test, train_batch_size, num_workers, generator)
        for X, Y, _ in test_loader:
            remainder = X.shape[0] % device_count
            if remainder != 0:
                X = X[:-remainder]
                Y = Y[:-remainder]

            N = X.shape[0]
            X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
            Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])

            log_prob_ratio, hit = test_step(state, X, Y)

            auc += N * roc_auc_score(Y.flatten(), log_prob_ratio.flatten())
            hits += unreplicate(hit)
            prior = unreplicate(state.prior['target']).reshape((C, K))
            norm += N * jnp.linalg.norm(prior - joint)

        auc = auc/len(test)
        auc_sweep = auc_sweep.at[i].set(auc)
        accuracy = hits/len(test)
        accuracy_sweep = accuracy_sweep.at[i].set(accuracy)
        norm = norm/len(test)
        norm_sweep = norm_sweep.at[i].set(norm)

        with jnp.printoptions(precision=4):
            print(f'[{label}] Environment {i:>2} {seen} AUC {auc}, Accuracy {accuracy}, Norm {norm}')

    print(f'[{label}] Average AUC {jnp.nanmean(auc_sweep)}, Accuracy {jnp.nanmean(accuracy_sweep)}, Norm {jnp.nanmean(norm_sweep)}')

    return state, C, K, dataset.environments, test_splits, auc_sweep, accuracy_sweep, norm_sweep


def adapt(state: TrainState, C: int, K: int, train_domains_set: Set[int],
        test_splits: Sequence[Tuple[torch.Tensor, Dataset]],
        batch_size: int, symmetric_dirichlet: bool, prior_strength: float, fix_marginal: bool, label: str,
        device_count: int, generator: torch.Generator, num_workers: int) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    print(f'---> {label}')
    prior_strength = replicate(prior_strength)

    auc_sweep = jnp.empty(len(test_splits))
    accuracy_sweep = jnp.empty(len(test_splits))
    norm_sweep = jnp.empty(len(test_splits))
    for i, (joint, test) in enumerate(test_splits):
        # happens on the source domain when train_fraction = 1
        if len(test) == 0:
            auc_sweep = auc_sweep.at[i].set(jnp.nan)
            accuracy_sweep = accuracy_sweep.at[i].set(jnp.nan)
            norm_sweep = norm_sweep.at[i].set(jnp.nan)
            continue

        seen = '  (seen)' if i in train_domains_set else '(unseen)'

        auc = hits = norm = 0
        joint = jnp.array(joint)

        test_loader = FastDataLoader(test, batch_size, num_workers, generator)
        for X, Y, _ in test_loader:
            remainder = X.shape[0] % device_count
            if remainder != 0:
                X = X[:-remainder]
                Y = Y[:-remainder]

            N = X.shape[0]
            X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
            Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])

            state = adapt_step(state, X, C, K, symmetric_dirichlet, prior_strength, fix_marginal)
            log_prob_ratio, hit = test_step(state, X, Y)

            auc += N * roc_auc_score(Y.flatten(), log_prob_ratio.flatten())
            hits += unreplicate(hit)
            prior = unreplicate(state.prior['target']).reshape((C, K))
            norm += N * jnp.linalg.norm(prior - joint)

        auc = auc/len(test)
        auc_sweep = auc_sweep.at[i].set(auc)
        accuracy = hits/len(test)
        accuracy_sweep = accuracy_sweep.at[i].set(accuracy)
        norm = norm/len(test)
        norm_sweep = norm_sweep.at[i].set(norm)

        with jnp.printoptions(precision=4):
            print(f'[{label}] Environment {i:>2} {seen} AUC {auc}, Accuracy {accuracy}, Norm {norm}')

    print(f'[{label}] Average AUC {jnp.nanmean(auc_sweep)}, Accuracy {jnp.nanmean(accuracy_sweep)}, Norm {jnp.nanmean(norm_sweep)}')

    return state, auc_sweep, accuracy_sweep, norm_sweep


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


def plot_auc(auc_sweeps: Curves, environments: np.ndarray, train_domains_set: Set[int], plot_title: str, plot_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))

    auc_sweep = auc_sweeps.pop(('Unadapted', (None, None, None, None)))
    ax.plot(environments, auc_sweep, linestyle='--', label='Unadapted')

    for (label, _), auc_sweep in auc_sweeps.items():
        ax.plot(environments, auc_sweep, label=label)

    for i in train_domains_set:
        ax.axvline(environments[i], linestyle=':')

    plt.ylim((0, 1))
    plt.xlabel('Shift Parameter')
    plt.ylabel('Average AUC')
    plt.title(plot_title)
    plt.grid(True)
    plt.legend()

    plt.savefig(plot_path)
    plt.close(fig)


def plot_accuracy(accuracy_sweeps: Curves, environments: np.ndarray, train_domains_set: Set[int],
        train_label_noise: float, plot_title: str, plot_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    upper_bound = np.maximum(np.maximum(1-environments, environments), (1-train_label_noise) * np.ones_like(environments))
    ax.plot(environments, upper_bound, linestyle=':', label='Upper bound')

    accuracy_sweep = accuracy_sweeps.pop(('Unadapted', (None, None, None, None)))
    ax.plot(environments, accuracy_sweep, linestyle='--', label='Unadapted')

    for (label, _), accuracy_sweep in accuracy_sweeps.items():
        ax.plot(environments, accuracy_sweep, label=label)

    for i in train_domains_set:
        ax.axvline(environments[i], linestyle=':')

    plt.ylim((0, 1))
    plt.xlabel('Shift Parameter')
    plt.ylabel('Accuracy')
    plt.title(plot_title)
    plt.grid(True)
    plt.legend()

    plt.savefig(plot_path)
    plt.close(fig)


def plot_norm(norm_sweeps: Curves, environments: np.ndarray, train_domains_set: Set[int], plot_title: str, plot_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))

    norm_sweep = norm_sweeps.pop(('Unadapted', (None, None, None, None)))
    ax.plot(environments, norm_sweep, linestyle='--', label='Unadapted')

    for (label, _), norm_sweep in norm_sweeps.items():
        ax.plot(environments, norm_sweep, label=label)

    for i in train_domains_set:
        ax.axvline(environments[i], linestyle=':')

    plt.xlabel('Shift Parameter')
    plt.ylabel('Frobenius norm')
    plt.title(plot_title)
    plt.grid(True)
    plt.legend()

    plt.savefig(plot_path)
    plt.close(fig)


if __name__ == '__main__':
    initialize_cache('jit_cache')
    cli()
