from typing import Any, Sequence, List, Tuple, Set, Optional
from pathlib import Path
import sys
import random
from itertools import product
from pprint import pprint

import jax
import jax.numpy as jnp
from jax.experimental.compilation_cache.compilation_cache import initialize_cache
import flax
from flax.jax_utils import replicate, unreplicate
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import click
from sklearn.metrics import roc_auc_score

from .common import Curves, Sweeps
from .utils import Tee
from .datasets import split
from .datasets.mnist import MultipleDomainMNIST
from .datasets.coco import ColoredCOCO
from .datasets.waterbirds import MultipleDomainWaterbirds
from .datasets.chexpert import MultipleDomainCheXpert
from .train import (
    TrainState,
    create_train_state,
    train_step,
    calibration_step,
    cross_replica_mean,
    induce_step,
    adapt_step,
    test_step,
)
from .restore import restore_train_state
from .visualize import plot_mean, plot_l1, plot_auc, plot_accuracy, plot_norm


@click.command()
@click.option("--config_name", type=str, required=True)
@click.option(
    "--dataset_name", type=click.Choice(["MNIST", "COCO", "Waterbirds", "CheXpert"]), required=True
)
@click.option("--dataset_apply_rotation", type=bool, required=True)
@click.option("--dataset_label_noise", type=float, required=True)
@click.option("--train_model", type=str, required=True)
@click.option(
    "--train_checkpoint_path", type=click.Path(path_type=Path), required=False
)
@click.option("--train_domains", type=str, required=True)
@click.option("--train_fraction", type=float, required=True)
@click.option("--train_calibration_fraction", type=float, required=True)
@click.option("--train_batch_size", type=int, required=True)
@click.option("--train_epochs", type=int, required=True)
@click.option("--train_lr", type=float, required=True)
@click.option("--calibration_domains", type=str, required=False)
@click.option("--calibration_fraction", type=float, required=False)
@click.option("--calibration_batch_size", type=int, required=True)
@click.option("--calibration_epochs", type=int, required=True)
@click.option("--calibration_lr", type=float, required=True)
@click.option("--test_prior_strength", type=float, required=False, multiple=True)
@click.option("--test_symmetric_dirichlet", type=bool, required=False, multiple=True)
@click.option("--test_fix_marginal", type=bool, required=False, multiple=True)
@click.option("--test_batch_size", type=int, required=True, multiple=True)
@click.option(
    "--plot_title", type=str, required=False, default="Performance on Each Domain"
)
@click.option("--seed", type=int, required=True)
@click.option("--num_workers", type=int, required=True)
def cli(
    config_name: str,
    dataset_name: str,
    dataset_apply_rotation: bool,
    dataset_label_noise: float,
    train_model: str,
    train_checkpoint_path: Optional[Path],
    train_domains: str,
    train_fraction: float,
    train_calibration_fraction: float,
    train_batch_size: int,
    train_epochs: int,
    train_lr: float,
    calibration_domains: Optional[str],
    calibration_fraction: Optional[float],
    calibration_batch_size: int,
    calibration_epochs: int,
    calibration_lr: float,
    test_prior_strength: Sequence[float],
    test_symmetric_dirichlet: Sequence[bool],
    test_fix_marginal: Sequence[bool],
    test_batch_size: Sequence[int],
    plot_title: str,
    seed: int,
    num_workers: int,
) -> None:
    main(
        config_name,
        dataset_name,
        dataset_apply_rotation,
        dataset_label_noise,
        train_model,
        train_checkpoint_path,
        train_domains,
        train_fraction,
        train_calibration_fraction,
        train_batch_size,
        train_epochs,
        train_lr,
        calibration_domains,
        calibration_fraction,
        calibration_batch_size,
        calibration_epochs,
        calibration_lr,
        test_prior_strength,
        test_symmetric_dirichlet,
        test_fix_marginal,
        test_batch_size,
        plot_title,
        seed,
        num_workers,
    )


def main(
    config_name: str,
    dataset_name: str,
    dataset_apply_rotation: bool,
    dataset_label_noise: float,
    train_model: str,
    train_checkpoint_path: Optional[Path],
    train_domains: str,
    train_fraction: float,
    train_calibration_fraction: float,
    train_batch_size: int,
    train_epochs: int,
    train_lr: float,
    calibration_domains: Optional[str],
    calibration_fraction: Optional[float],
    calibration_batch_size: int,
    calibration_epochs: int,
    calibration_lr: float,
    test_prior_strength: Sequence[float],
    test_symmetric_dirichlet: Sequence[bool],
    test_fix_marginal: Sequence[bool],
    test_batch_size: Sequence[int],
    plot_title: str,
    seed: int,
    num_workers: int,
) -> Tuple[Curves, Curves, Curves, Curves, Curves]:
    log_root = Path("logs/")
    plot_root = Path("plots/")
    npz_root = Path("npz/")

    log_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)
    npz_root.mkdir(parents=True, exist_ok=True)

    log_path = log_root / f"{config_name}.txt"
    sys.stdout = Tee(log_path)

    mean_plot_path = plot_root / f"{config_name}_mean.png"
    l1_plot_path = plot_root / f"{config_name}_l1.png"
    auc_plot_path = plot_root / f"{config_name}_auc.png"
    accuracy_plot_path = plot_root / f"{config_name}_accuracy.png"
    norm_plot_path = plot_root / f"{config_name}_norm.png"

    mean_npz_path = npz_root / f"{config_name}_mean.npz"
    l1_npz_path = npz_root / f"{config_name}_l1.npz"
    auc_npz_path = npz_root / f"{config_name}_auc.npz"
    accuracy_npz_path = npz_root / f"{config_name}_accuracy.npz"
    norm_npz_path = npz_root / f"{config_name}_norm.npz"

    device_count = jax.local_device_count()
    assert (
        train_batch_size % device_count == 0
    ), f"train_batch_size should be divisible by {device_count}"
    assert (
        calibration_batch_size % device_count == 0
    ), f"calibration_batch_size should be divisible by {device_count}"
    for batch_size in test_batch_size:
        assert (
            batch_size % device_count == 0
        ), f"test_batch_size should be divisible by {device_count}"
    assert (
        (
            calibration_domains is not None
            and (calibration_fraction is None or calibration_fraction > 0)
        )
        or train_calibration_fraction > 0
        or calibration_epochs == 0
    ), "Calibration set may not be empty"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    key = jax.random.PRNGKey(seed)
    generator = torch.Generator().manual_seed(seed)

    train_domains_set = set(int(env) for env in train_domains.split(","))
    if len(train_domains_set) != 1:
        raise NotImplementedError(
            "Training on multiple source distributions is not supported yet."
        )

    if calibration_domains is not None:
        calibration_domains_set = set(
            int(env) for env in calibration_domains.split(",")
        )
    else:
        calibration_domains_set = set()
    if calibration_fraction is None:
        calibration_fraction = 1.0

    (
        state,
        C,
        K,
        confounder_strength,
        eval_splits,
        oracle_sweep,
        unadapted_sweep,
    ) = train(
        dataset_name,
        dataset_apply_rotation,
        dataset_label_noise,
        train_model,
        train_checkpoint_path,
        train_domains_set,
        train_fraction,
        train_calibration_fraction,
        train_batch_size,
        train_epochs,
        train_lr,
        calibration_domains_set,
        calibration_fraction,
        calibration_batch_size,
        calibration_epochs,
        calibration_lr,
        device_count,
        key,
        generator,
        num_workers,
    )

    (
        oracle_mean_sweep,
        oracle_l1_sweep,
        oracle_auc_sweep,
        oracle_accuracy_sweep,
        oracle_norm_sweep,
    ) = oracle_sweep
    (
        unadapted_mean_sweep,
        unadapted_l1_sweep,
        unadapted_auc_sweep,
        unadapted_accuracy_sweep,
        unadapted_norm_sweep,
    ) = unadapted_sweep
    mean_sweeps: Curves = {
        ("Oracle", None, train_batch_size): oracle_mean_sweep,
        ("Unadapted", None, train_batch_size): unadapted_mean_sweep,
    }
    l1_sweeps: Curves = {
        ("Oracle", None, train_batch_size): oracle_l1_sweep,
        ("Unadapted", None, train_batch_size): unadapted_l1_sweep,
    }
    auc_sweeps: Curves = {
        ("Oracle", None, train_batch_size): oracle_auc_sweep,
        ("Unadapted", None, train_batch_size): unadapted_auc_sweep,
    }
    accuracy_sweeps: Curves = {
        ("Oracle", None, train_batch_size): oracle_accuracy_sweep,
        ("Unadapted", None, train_batch_size): unadapted_accuracy_sweep,
    }
    norm_sweeps: Curves = {
        ("Oracle", None, train_batch_size): oracle_norm_sweep,
        ("Unadapted", None, train_batch_size): unadapted_norm_sweep,
    }

    for prior_strength, symmetric_dirichlet, fix_marginal, batch_size in product(
        test_prior_strength,
        test_symmetric_dirichlet,
        test_fix_marginal,
        test_batch_size,
    ):
        label = f"{prior_strength=} | {symmetric_dirichlet=} | {fix_marginal=} | {batch_size=}"
        scheme = (prior_strength, symmetric_dirichlet, fix_marginal)
        state, (mean_sweep, l1_sweep, auc_sweep, accuracy_sweep, norm_sweep) = adapt(
            state,
            C,
            K,
            dataset_label_noise,
            train_domains_set,
            calibration_domains_set,
            eval_splits,
            scheme,
            batch_size,
            label,
            device_count,
            generator,
            num_workers,
        )
        key = label, (prior_strength, symmetric_dirichlet, fix_marginal), batch_size
        mean_sweeps[key] = mean_sweep
        l1_sweeps[key] = l1_sweep
        auc_sweeps[key] = auc_sweep
        accuracy_sweeps[key] = accuracy_sweep
        norm_sweeps[key] = norm_sweep

    print("===> mean_sweeps")
    pprint(mean_sweeps)
    print("===> l1_sweeps")
    pprint(l1_sweeps)
    print("===> auc_sweeps")
    pprint(auc_sweeps)
    print("===> accuracy_sweeps")
    pprint(accuracy_sweeps)
    print("===> norm_sweeps")
    pprint(norm_sweeps)

    jnp.savez(mean_npz_path, **{k: v for (k, _, _), v in mean_sweeps.items()})
    jnp.savez(l1_npz_path, **{k: v for (k, _, _), v in l1_sweeps.items()})
    jnp.savez(auc_npz_path, **{k: v for (k, _, _), v in auc_sweeps.items()})
    jnp.savez(accuracy_npz_path, **{k: v for (k, _, _), v in accuracy_sweeps.items()})
    jnp.savez(norm_npz_path, **{k: v for (k, _, _), v in norm_sweeps.items()})

    plot_mean(
        mean_sweeps,
        train_batch_size,
        confounder_strength,
        train_domains_set,
        plot_title,
        mean_plot_path,
    )
    plot_l1(
        l1_sweeps,
        train_batch_size,
        confounder_strength,
        train_domains_set,
        plot_title,
        l1_plot_path,
    )
    plot_auc(
        auc_sweeps,
        train_batch_size,
        confounder_strength,
        train_domains_set,
        plot_title,
        auc_plot_path,
    )
    plot_accuracy(
        accuracy_sweeps,
        train_batch_size,
        confounder_strength,
        train_domains_set,
        dataset_label_noise,
        plot_title,
        accuracy_plot_path,
    )
    plot_norm(
        norm_sweeps,
        train_batch_size,
        confounder_strength,
        train_domains_set,
        plot_title,
        norm_plot_path,
    )

    return mean_sweeps, l1_sweeps, auc_sweeps, accuracy_sweeps, norm_sweeps


def train(
    dataset_name: str,
    dataset_apply_rotation: bool,
    dataset_label_noise: float,
    train_model: str,
    train_checkpoint_path: Optional[Path],
    train_domains_set: Set[int],
    train_fraction: float,
    train_calibration_fraction: float,
    train_batch_size: int,
    train_epochs: int,
    train_lr: float,
    calibration_domains_set: Set[int],
    calibration_fraction: float,
    calibration_batch_size: int,
    calibration_epochs: int,
    calibration_lr: float,
    device_count: int,
    key: Any,
    generator: torch.Generator,
    num_workers: int,
) -> Tuple[
    TrainState,
    int,
    int,
    np.ndarray,
    List[Tuple[torch.Tensor, Dataset]],
    Sweeps,
    Sweeps,
]:
    if dataset_name == "MNIST":
        root = Path("data/")
        dataset = MultipleDomainMNIST(
            root,
            generator,
            train_domains_set,
            dataset_apply_rotation,
            dataset_label_noise,
        )
    elif dataset_name == "COCO":
        assert (
            dataset_apply_rotation is False
        ), "Parameter dataset_apply_rotation is not supported with COCO"
        assert (
            dataset_label_noise == 0
        ), "Parameter dataset_label_noise is not supported with COCO"

        root = Path("data/COCO/train2017")
        annFile = Path("data/COCO/annotations/instances_train2017.json")
        dataset = ColoredCOCO(root, annFile, generator)
    elif dataset_name == "Waterbirds":
        assert (
            dataset_apply_rotation is False
        ), "Parameter dataset_apply_rotation is not supported with Waterbirds"
        assert (
            dataset_label_noise == 0
        ), "Parameter dataset_label_noise is not supported with Waterbirds"

        root = Path("data/")
        dataset = MultipleDomainWaterbirds(root, generator)
    elif dataset_name == "CheXpert":
        assert (
            dataset_apply_rotation is False
        ), "Parameter dataset_apply_rotation is not supported with CheXpert"
        assert (
            dataset_label_noise == 0
        ), "Parameter dataset_label_noise is not supported with CheXpert"

        root = Path("data/CheXpert")
        dataset = MultipleDomainCheXpert(root, generator, train_domains_set, "PNEUMONIA", "GENDER")
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    C, K = dataset.C, dataset.K
    if C != 2:
        raise NotImplementedError("Multi-label classification is not supported yet.")

    train, calibration, test_splits = split(
        dataset,
        train_domains_set,
        train_fraction,
        train_calibration_fraction,
        calibration_domains_set,
        calibration_fraction,
    )
    print("train", len(train))
    print("calibration", len(calibration))
    train_domain, = train_domains_set
    print(f"test_splits[{train_domain}][-1]", len(test_splits[train_domain][-1]))

    eval_splits = test_splits.copy()
    eval_splits.append((*test_splits[train_domain][:-1], train))

    key_init, key = jax.random.split(key)
    specimen = jnp.empty(dataset.input_shape)
    state = create_train_state(
        key_init,
        C,
        K,
        train_model,
        train_lr,
        specimen,
        device_count,
    )
    if train_checkpoint_path is not None:
        state = restore_train_state(state, train_checkpoint_path)
    state: TrainState = replicate(state)

    print("===> Training")
    train_loader = DataLoader(
        train,
        train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    for epoch in range(train_epochs):
        epoch_loss = 0
        epoch_hit = jnp.zeros(C * K, dtype=int)
        epoch_total = jnp.zeros(C * K, dtype=int)
        for X, _, Y, Z in train_loader:
            remainder = X.shape[0] % device_count
            X = X[remainder:]
            Y = Y[remainder:]
            Z = Z[remainder:]

            X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
            Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])
            Z = jnp.array(Z).reshape(device_count, -1, *Z.shape[1:])
            M = Y * K + Z

            state, (loss, hit, total) = train_step(state, X, M)
            epoch_loss += unreplicate(loss)
            epoch_hit += unreplicate(hit)
            epoch_total += unreplicate(total)

        with jnp.printoptions(precision=3):
            print(f"Train epoch {epoch + 1}, loss: {epoch_loss}, hit: {epoch_hit}, total: {epoch_total}")

    # Sync the batch statistics across replicas so that evaluation is deterministic.
    state = state.replace(batch_stats=cross_replica_mean(state.batch_stats))

    print("===> Calibrating")
    if len(calibration) or calibration_epochs:
        calibration_loader = DataLoader(
            calibration,
            calibration_batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        )
        for epoch in range(calibration_epochs):
            epoch_loss = 0
            epoch_hit = jnp.zeros(C * K, dtype=int)
            epoch_total = jnp.zeros(C * K, dtype=int)
            for X, _, Y, Z in calibration_loader:
                remainder = X.shape[0] % device_count
                X = X[remainder:]
                Y = Y[remainder:]
                Z = Z[remainder:]

                X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
                Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])
                Z = jnp.array(Z).reshape(device_count, -1, *Z.shape[1:])
                M = Y * K + Z

                state, (loss, hit, total) = calibration_step(state, X, M, calibration_lr)
                epoch_loss += unreplicate(loss)
                epoch_hit += unreplicate(hit)
                epoch_total += unreplicate(total)

            with jnp.printoptions(precision=3):
                print(f"Calibration epoch {epoch + 1}, loss: {epoch_loss}, hit: {epoch_hit}, total: {epoch_total}")

    # Sync the batch statistics across replicas so that evaluation is deterministic.
    state = state.replace(batch_stats=cross_replica_mean(state.batch_stats))
    
    print("---> Temperature =", unreplicate(state.params["T"]))
    print("---> Bias =", unreplicate(state.params["b"]))

    print("===> Estimating Source Label Prior")
    source_prior_induced = estimate_source_prior(
        calibration,
        calibration_batch_size,
        num_workers,
        generator,
        C,
        K,
        device_count,
        state,
        "induce"
    )
    source_prior_empirical = estimate_source_prior(
        train,
        train_batch_size,
        num_workers,
        generator,
        C,
        K,
        device_count,
        state,
        "count"
    )
    
    print("---> Induced source label prior =", source_prior_induced)
    print("---> Empirical source label prior =", source_prior_empirical)
    print("---> Total variation distance =", jnp.sum(jnp.abs(source_prior_induced - source_prior_empirical))/2)

    prior = state.prior.unfreeze()
    prior["source"] = replicate(source_prior_induced)
    state = state.replace(prior=flax.core.frozen_dict.freeze(prior))

    print("===> Adapting & Evaluating")

    # batch size does not matter since we are not doing adaptation
    label = "Oracle"
    state, oracle_sweep = adapt(
        state,
        C,
        K,
        dataset_label_noise,
        train_domains_set,
        calibration_domains_set,
        eval_splits,
        None,
        train_batch_size,
        label,
        device_count,
        generator,
        num_workers,
    )
    label = "Unadapted"
    state, unadapted_sweep = adapt(
        state,
        C,
        K,
        dataset_label_noise,
        train_domains_set,
        calibration_domains_set,
        eval_splits,
        None,
        train_batch_size,
        label,
        device_count,
        generator,
        num_workers,
    )

    return (
        state,
        C,
        K,
        dataset.confounder_strength,
        eval_splits,
        oracle_sweep,
        unadapted_sweep,
    )


def adapt(
    state: TrainState,
    C: int,
    K: int,
    dataset_label_noise: float,
    train_domains_set: Set[int],
    calibration_domains_set: Set[int],
    eval_splits: Sequence[Tuple[torch.Tensor, Dataset]],
    scheme: Optional[Tuple[float, bool, bool]],
    batch_size: int,
    label: str,
    device_count: int,
    generator: torch.Generator,
    num_workers: int,
) -> Tuple[TrainState, Sweeps]:
    print(f"---> {label}")

    mean_sweep = jnp.empty(len(eval_splits))
    l1_sweep = jnp.empty(len(eval_splits))
    auc_sweep = jnp.empty(len(eval_splits))
    accuracy_sweep = jnp.empty(len(eval_splits))
    norm_sweep = jnp.empty(len(eval_splits))
    for i, (joint_M, eval_) in enumerate(eval_splits):
        # happens on the source domain when train_fraction = 1.0
        if len(eval_) == 0:
            mean_sweep = mean_sweep.at[i].set(jnp.nan)
            l1_sweep = l1_sweep.at[i].set(jnp.nan)
            auc_sweep = auc_sweep.at[i].set(jnp.nan)
            accuracy_sweep = accuracy_sweep.at[i].set(jnp.nan)
            norm_sweep = norm_sweep.at[i].set(jnp.nan)
            continue

        seen = (
            "  (seen)"
            if i in train_domains_set.union(calibration_domains_set)
            else " (train)"
            if i == len(eval_splits) - 1
            else "(unseen)"
        )

        joint_M = jnp.array(joint_M)
        flip_prob = jnp.array(
            [
                [1 - dataset_label_noise, dataset_label_noise],
                [dataset_label_noise, 1 - dataset_label_noise],
            ]
        )
        joint = flip_prob[:, :, jnp.newaxis] * joint_M  # P(Y_tilde, Y, Z)
        prob = joint / jnp.sum(joint, axis=(0, 2), keepdims=True)
        prob = prob[:, 1, :]  # P(Y=1|Y_tilde, Z)

        # using shuffle=True so that Y contains multiple classes, otherwise AUC is not defined
        mean = l1 = hits = norm = 0
        epoch_Y = jnp.empty(len(eval_) // device_count * device_count)
        epoch_score = jnp.empty(len(eval_) // device_count * device_count)
        offset = 0

        eval_loader = DataLoader(
            eval_,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        )
        for X, Y_tilde, Y, Z in eval_loader:
            remainder = X.shape[0] % device_count
            X = X[remainder:]
            Y_tilde = Y_tilde[remainder:]
            Y = Y[remainder:]
            Z = Z[remainder:]

            N = X.shape[0]
            X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
            Y_tilde = jnp.array(Y_tilde)
            Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])
            Z = jnp.array(Z).reshape(device_count, -1, *Z.shape[1:])

            if isinstance(scheme, tuple):
                prior_strength, symmetric_dirichlet, fix_marginal = scheme
                state = adapt_step(
                    state,
                    X,
                    replicate(prior_strength),
                    symmetric_dirichlet,
                    fix_marginal,
                    C,
                    K,
                )
            elif label == "Oracle":
                prior = state.prior.unfreeze()
                prior["target"] = replicate(joint_M.flatten())
                state = state.replace(prior=flax.core.frozen_dict.freeze(prior))
            elif label == "Unadapted":
                prior = state.prior.unfreeze()
                prior["target"] = prior["source"]
                state = state.replace(prior=flax.core.frozen_dict.freeze(prior))
            else:
                raise ValueError(f"Unknown adaptation scheme {label}")

            (score, hit), (_, _) = test_step(state, X, Y, Z)

            mean += jnp.sum(score)
            l1 += jnp.sum(jnp.abs(score.flatten() - prob[Y_tilde, Z.flatten()]))
            epoch_Y = epoch_Y.at[offset:offset+N].set(Y.flatten())
            epoch_score = epoch_score.at[offset:offset+N].set(score.flatten())
            hits += unreplicate(hit)
            prior = unreplicate(state.prior["target"]).reshape((C, K))
            norm += N * jnp.linalg.norm(prior - joint_M)

            offset += N

        mean = mean / len(eval_)
        l1 = l1 / len(eval_)
        auc = roc_auc_score(epoch_Y, epoch_score)
        accuracy = hits / len(eval_)
        norm = norm / len(eval_)

        with jnp.printoptions(precision=4):
            print(
                f"[{label}] Environment {i:>2} {seen} mean {mean}, L1 {l1}, AUC {auc}, Accuracy {accuracy}, Norm {norm}"
            )

        # note that foo_sweep.at[-1] is the training foo
        mean_sweep = mean_sweep.at[i].set(mean)
        l1_sweep = l1_sweep.at[i].set(l1)
        auc_sweep = auc_sweep.at[i].set(auc)
        accuracy_sweep = accuracy_sweep.at[i].set(accuracy)
        norm_sweep = norm_sweep.at[i].set(norm)

    print(
        f"[{label}] Average response {jnp.nanmean(mean_sweep[:-1])}, "
        f"Average L1 {jnp.nanmean(l1_sweep[:-1])}, "
        f"Average AUC {jnp.nanmean(auc_sweep[:-1])}, "
        f"Accuracy {jnp.nanmean(accuracy_sweep[:-1])}, "
        f"Norm {jnp.nanmean(norm_sweep[:-1])}"
    )

    return state, (mean_sweep, l1_sweep, auc_sweep, accuracy_sweep, norm_sweep)


def estimate_source_prior(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    generator: torch.Generator,
    C: int,
    K: int,
    device_count: int,
    state: TrainState,
    method: str,
) -> jnp.ndarray:
    loader = DataLoader(
        dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        generator=generator,
    )
    if method == "count":
        source_prior = np.zeros((C * K))
        I = np.identity(C * K)
        for _, _, Y, Z in loader:
            M = Y * K + Z
            source_prior += np.sum(I[M], axis=0)

        source_prior = jnp.array(source_prior / np.sum(source_prior))

    elif method == "induce":
        N = 0
        source_prior = jnp.zeros(C * K)
        for X, _, _, _ in loader:
            remainder = X.shape[0] % device_count
            X = X[remainder:]

            N += X.shape[0]
            X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
            source_prior = source_prior + unreplicate(induce_step(state, X))

        source_prior = source_prior / N

    else:
        raise ValueError(f"Unknown source label prior estimation method {method}")

    return source_prior


if __name__ == "__main__":
    initialize_cache("jit_cache")
    cli()
