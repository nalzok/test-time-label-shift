from types import SimpleNamespace
from typing import Any, Sequence, List, Tuple, Set, Optional, Dict
from pathlib import Path
from hashlib import sha256
import sys
import random
from itertools import product
from pprint import pprint

import jax
import jax.numpy as jnp
from jax.experimental.compilation_cache.compilation_cache import initialize_cache
import flax
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
from flax.jax_utils import replicate, unreplicate
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import click
from sklearn.metrics import roc_auc_score

from tta.common import Adaptation, Curves, Sweeps
from tta.utils import Tee
from tta.datasets import MultipleDomainDataset, split, subsample
from tta.datasets.mnist import MultipleDomainMNIST
from tta.datasets.coco import ColoredCOCO
from tta.datasets.waterbirds import MultipleDomainWaterbirds
from tta.datasets.cxr.chexpert import MultipleDomainCheXpert
from tta.datasets.cxr.mimic import MultipleDomainMIMIC
from tta.train import (
    TrainState,
    create_train_state,
    train_step,
    validation_step,
    calibration_step,
    cross_replica_mean,
    induce_step,
    adapt_step,
    test_step,
)
from tta.restore import restore_train_state
from tta.visualize import latexify, plot


@click.command()
@click.option("--config_name", type=str, required=True)
@click.option(
    "--dataset_name",
    type=click.Choice(["MNIST", "COCO", "Waterbirds", "CheXpert", "MIMIC"]),
    required=True,
)
@click.option("--dataset_Y_column", type=str, required=False)
@click.option("--dataset_Z_column", type=str, required=False)
@click.option("--dataset_target_domain_count", type=int, required=False)
@click.option("--dataset_source_domain_count", type=int, required=False)
@click.option("--dataset_subsample_what", type=str, required=True)
@click.option("--dataset_use_embedding", type=bool, required=False)
@click.option("--dataset_apply_rotation", type=bool, required=False)
@click.option("--dataset_feature_noise", type=float, required=True)
@click.option("--dataset_label_noise", type=float, required=True)
@click.option("--train_fit_joint", type=bool, required=True)
@click.option("--train_model", type=str, required=True)
@click.option(
    "--train_pretrained_path", type=click.Path(path_type=Path), required=False
)
@click.option("--train_domains", type=str, required=True)
@click.option("--train_fraction", type=float, required=True)
@click.option("--train_calibration_fraction", type=float, required=True)
@click.option("--train_batch_size", type=int, required=True)
@click.option("--train_epochs", type=int, required=True)
@click.option("--train_decay", type=float, required=True)
@click.option("--train_patience", type=int, required=True)
@click.option("--train_tau", type=float, required=True)
@click.option("--train_lr", type=float, required=True)
@click.option("--calibration_domains", type=str, required=False)
@click.option("--calibration_fraction", type=float, required=False)
@click.option("--calibration_batch_size", type=int, required=True)
@click.option("--calibration_epochs", type=int, required=True)
@click.option("--calibration_decay", type=float, required=True)
@click.option("--calibration_patience", type=int, required=True)
@click.option("--calibration_tau", type=float, required=True)
@click.option("--calibration_lr", type=float, required=True)
@click.option("--adapt_skip_null_oracle", is_flag=True)
@click.option("--adapt_gmtl_alpha", type=float, required=False, multiple=True)
@click.option("--adapt_prior_strength", type=float, required=False, multiple=True)
@click.option("--adapt_symmetric_dirichlet", type=bool, required=False, multiple=True)
@click.option("--adapt_fix_marginal", type=bool, required=False, multiple=True)
@click.option("--test_argmax_joint", type=bool, required=True, multiple=True)
@click.option("--test_batch_size", type=int, required=True, multiple=True)
@click.option("--seed", type=int, required=True)
@click.option("--num_workers", type=int, required=True)
@click.option(
    "--plot_title", type=str, required=False, default="Performance on Each Domain"
)
@click.option("--plot_only", type=bool, required=True)
def cli(
    config_name: str,
    dataset_name: str,
    dataset_y_column: Optional[str],
    dataset_z_column: Optional[str],
    dataset_target_domain_count: Optional[int],
    dataset_source_domain_count: Optional[int],
    dataset_subsample_what: str,
    dataset_use_embedding: Optional[bool],
    dataset_apply_rotation: Optional[bool],
    dataset_feature_noise: float,
    dataset_label_noise: float,
    train_fit_joint: bool,
    train_model: str,
    train_pretrained_path: Optional[Path],
    train_domains: str,
    train_fraction: float,
    train_calibration_fraction: float,
    train_batch_size: int,
    train_epochs: int,
    train_decay: float,
    train_patience: int,
    train_tau: float,
    train_lr: float,
    calibration_domains: Optional[str],
    calibration_fraction: Optional[float],
    calibration_batch_size: int,
    calibration_epochs: int,
    calibration_decay: float,
    calibration_patience: int,
    calibration_tau: float,
    calibration_lr: float,
    adapt_skip_null_oracle: bool,
    adapt_gmtl_alpha: Sequence[float],
    adapt_prior_strength: Sequence[float],
    adapt_symmetric_dirichlet: Sequence[bool],
    adapt_fix_marginal: Sequence[bool],
    test_argmax_joint: Sequence[bool],
    test_batch_size: Sequence[int],
    seed: int,
    num_workers: int,
    plot_title: str,
    plot_only: bool,
) -> None:
    log_root = Path("logs/")
    npz_root = Path("npz/")
    plot_root = Path("plots/")

    log_root.mkdir(parents=True, exist_ok=True)
    npz_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)

    log_path = log_root / f"{config_name}.txt"
    if not plot_only:
        sys.stdout = Tee(log_path)
    npz_path = npz_root / f"{config_name}.npz"

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

    if calibration_domains is None:
        calibration_domains_set = set()
    else:
        calibration_domains_set = set(
            int(env) for env in calibration_domains.split(",")
        )

    if calibration_fraction is None:
        calibration_fraction = 1.0


    if plot_only:
        # Ugly hack
        confounder_strength = np.linspace(0, 1, 21)
        dataset = SimpleNamespace(confounder_strength=confounder_strength)
    else:
        (
            dataset,
            (train, joint_train),
            (calibration, joint_calibration),
            eval_splits,
        ) = prepare_dataset(
            dataset_name,
            dataset_y_column,
            dataset_z_column,
            dataset_target_domain_count,
            dataset_source_domain_count,
            dataset_subsample_what,
            dataset_use_embedding,
            dataset_apply_rotation,
            dataset_feature_noise,
            dataset_label_noise,
            train_domains_set,
            train_fraction,
            train_calibration_fraction,
            calibration_domains_set,
            calibration_fraction,
            generator,
        )

        main(
            npz_path,
            dataset,
            train,
            joint_train,
            calibration,
            joint_calibration,
            eval_splits,
            train_domains_set,
            calibration_domains_set,
            dataset_label_noise,
            train_fit_joint,
            train_model,
            train_pretrained_path,
            train_batch_size,
            train_epochs,
            train_decay,
            train_patience,
            train_tau,
            train_lr,
            calibration_batch_size,
            calibration_epochs,
            calibration_decay,
            calibration_patience,
            calibration_tau,
            calibration_lr,
            adapt_skip_null_oracle,
            adapt_gmtl_alpha,
            adapt_prior_strength,
            adapt_symmetric_dirichlet,
            adapt_fix_marginal,
            test_argmax_joint,
            test_batch_size,
            key,
            generator,
            num_workers,
        )

    plot(
        npz_path,
        dataset.confounder_strength,
        train_domains_set,
        dataset_label_noise,
        plot_title,
        plot_root,
        config_name,
    )


def prepare_dataset(
    dataset_name: str,
    dataset_y_column: Optional[str],
    dataset_z_column: Optional[str],
    dataset_target_domain_count: Optional[int],
    dataset_source_domain_count: Optional[int],
    dataset_subsample_what: str,
    dataset_use_embedding: Optional[bool],
    dataset_apply_rotation: Optional[bool],
    dataset_feature_noise: float,
    dataset_label_noise: float,
    train_domains_set: Set[int],
    train_fraction: float,
    train_calibration_fraction: float,
    calibration_domains_set: Set[int],
    calibration_fraction: float,
    generator: torch.Generator,
) -> Tuple[
    MultipleDomainDataset,
    Tuple[Dataset, torch.Tensor],
    Tuple[Dataset, torch.Tensor],
    List[Tuple[Dataset, torch.Tensor]],
]:
    if dataset_name == "MNIST":
        assert dataset_y_column is None
        assert dataset_z_column is None
        assert dataset_target_domain_count is None
        assert dataset_source_domain_count is None
        assert dataset_use_embedding is None
        assert dataset_apply_rotation is not None

        root = Path("data/mnist")
        dataset = MultipleDomainMNIST(
            root,
            train_domains_set,
            generator,
            dataset_apply_rotation,
            dataset_feature_noise,
            dataset_label_noise,
        )
    elif dataset_name == "COCO":
        assert dataset_y_column is None
        assert dataset_z_column is None
        assert dataset_target_domain_count is None
        assert dataset_source_domain_count is None
        assert dataset_use_embedding is None
        assert dataset_apply_rotation is None
        assert dataset_feature_noise == 0
        assert dataset_label_noise == 0

        root = Path("data/COCO/train2017")
        annFile = Path("data/COCO/annotations/instances_train2017.json")
        dataset = ColoredCOCO(root, annFile, generator)
    elif dataset_name == "Waterbirds":
        assert dataset_y_column is None
        assert dataset_z_column is None
        assert dataset_target_domain_count is None
        assert dataset_source_domain_count is None
        assert dataset_use_embedding is None
        assert dataset_apply_rotation is None
        assert dataset_feature_noise == 0
        assert dataset_label_noise == 0

        root = Path("data/")
        dataset = MultipleDomainWaterbirds(root, generator)
    elif dataset_name == "CheXpert":
        assert dataset_y_column is not None
        assert dataset_z_column is not None
        assert dataset_target_domain_count is not None
        assert dataset_use_embedding is not None
        assert dataset_apply_rotation is None
        assert dataset_feature_noise == 0
        assert dataset_label_noise == 0

        root = Path("data/CheXpert")
        dataset = MultipleDomainCheXpert(
            root,
            train_domains_set,
            generator,
            dataset_y_column,
            dataset_z_column,
            dataset_use_embedding,
            dataset_target_domain_count,
            dataset_source_domain_count,
        )
    elif dataset_name == "MIMIC":
        assert dataset_y_column is not None
        assert dataset_z_column is not None
        assert dataset_target_domain_count is not None
        assert dataset_use_embedding is True
        assert dataset_apply_rotation is None
        assert dataset_feature_noise == 0
        assert dataset_label_noise == 0

        root = Path("data/MIMIC")
        dataset = MultipleDomainMIMIC(
            root,
            train_domains_set,
            generator,
            dataset_y_column,
            dataset_z_column,
            dataset_use_embedding,
            dataset_target_domain_count,
            dataset_source_domain_count,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    C, K = dataset.C, dataset.K
    if C != 2 or K != 2:
        raise NotImplementedError("Multi-label classification is not supported yet.")

    m = sha256()
    m.update(dataset.hexdigest.encode())
    m.update(dataset_subsample_what.encode())
    dataset.hexdigest = m.hexdigest()

    print("domains:", [len(domain) for domain, _ in dataset.domains])
    (train, joint_train), (calibration, joint_calibration), test_splits = split(
        dataset,
        train_domains_set,
        train_fraction,
        train_calibration_fraction,
        calibration_domains_set,
        calibration_fraction,
    )
    print("train (before subsampling):", len(train))
    print(joint_train)
    print("calibration (before subsampling):", len(calibration))
    print(joint_calibration)

    if dataset_subsample_what != "none":
        train, joint_train = subsample(train, joint_train, dataset_subsample_what, generator)
        calibration, joint_calibration = subsample(calibration, joint_calibration, dataset_subsample_what, generator)
        print("train (after subsampling):", len(train))
        print(joint_train)
        print("calibration (after subsampling):", len(calibration))
        print(joint_calibration)

    (train_domain,) = train_domains_set
    test_split_train, joint_M_train = test_splits[train_domain]
    print(f"test_split_train:", len(test_split_train))
    print(joint_M_train)

    eval_splits = test_splits.copy()
    eval_splits.append((train, joint_train))

    return (
        dataset,
        (train, joint_train),
        (calibration, joint_calibration),
        eval_splits,
    )


def main(
    npz_path: Path,
    dataset: MultipleDomainDataset,
    train: ConcatDataset,
    joint_train: torch.Tensor,
    calibration: ConcatDataset,
    joint_calibration: torch.Tensor,
    eval_splits: List[Tuple[Dataset, torch.Tensor]],
    train_domains_set: Set[int],
    calibration_domains_set: Set[int],
    dataset_label_noise: float,
    train_fit_joint: bool,
    train_model: str,
    train_pretrained_path: Optional[Path],
    train_batch_size: int,
    train_epochs: int,
    train_decay: float,
    train_patience: int,
    train_tau: float,
    train_lr: float,
    calibration_batch_size: int,
    calibration_epochs: int,
    calibration_decay: float,
    calibration_patience: int,
    calibration_tau: float,
    calibration_lr: float,
    adapt_skip_null_oracle: bool,
    adapt_gmtl_alpha: Sequence[float],
    adapt_prior_strength: Sequence[float],
    adapt_symmetric_dirichlet: Sequence[bool],
    adapt_fix_marginal: Sequence[bool],
    test_argmax_joint: Sequence[bool],
    test_batch_size: Sequence[int],
    key: Any,
    generator: torch.Generator,
    num_workers: int,
) -> Dict[str, Curves]:
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

    state = train_fn(
        dataset,
        train,
        joint_train,
        calibration,
        joint_calibration,
        train_fit_joint,
        train_model,
        train_pretrained_path,
        train_batch_size,
        train_epochs,
        train_decay,
        train_patience,
        train_tau,
        train_lr,
        calibration_batch_size,
        calibration_epochs,
        calibration_decay,
        calibration_patience,
        calibration_tau,
        calibration_lr,
        key,
        generator,
        device_count,
        num_workers,
    )

    mean_sweeps, l1_sweeps, auc_sweeps, auc_Z_sweeps, accuracy_sweeps, accuracy_Z_sweeps, norm_sweeps = baseline_fn(
        state,
        dataset,
        eval_splits,
        dataset_label_noise,
        train_domains_set,
        train_batch_size,
        calibration_domains_set,
        adapt_skip_null_oracle,
        adapt_gmtl_alpha,
        generator,
        device_count,
        num_workers,
    )

    for (
        prior_strength,
        symmetric_dirichlet,
        fix_marginal,
        argmax_joint,
        batch_size,
    ) in product(
        adapt_prior_strength,
        adapt_symmetric_dirichlet,
        adapt_fix_marginal,
        test_argmax_joint,
        test_batch_size,
    ):
        adaptation = ("EM", prior_strength, symmetric_dirichlet, fix_marginal)
        state, (
            mean_sweep,
            l1_sweep,
            auc_sweep,
            auc_Z_sweep,
            accuracy_sweep,
            accuracy_Z_sweep,
            norm_sweep,
        ) = adapt_fn(
            state,
            dataset.C,
            dataset.K,
            dataset_label_noise,
            train_domains_set,
            calibration_domains_set,
            eval_splits,
            adaptation,
            argmax_joint,
            batch_size,
            device_count,
            generator,
            num_workers,
        )
        k = adaptation, argmax_joint, batch_size
        mean_sweeps[k] = mean_sweep
        l1_sweeps[k] = l1_sweep
        auc_sweeps[k] = auc_sweep
        auc_Z_sweeps[k] = auc_Z_sweep
        accuracy_sweeps[k] = accuracy_sweep
        accuracy_Z_sweeps[k] = accuracy_Z_sweep
        norm_sweeps[k] = norm_sweep

    all_sweeps = {
        "mean": (mean_sweeps, "Average probability of class 1"),
        "l1": (l1_sweeps, "Average L1 error of class 1"),
        "auc": (auc_sweeps, "AUC"),
        "auc_Z": (auc_Z_sweeps, "AUC (Z)"),
        "accuracy": (accuracy_sweeps, "Accuracy"),
        "accuracy_Z": (accuracy_Z_sweeps, "Accuracy (Z)"),
        "norm": (norm_sweeps, "Euclidean distance"),
    }
    pprint(all_sweeps)

    if npz_path.exists():
        all_existing_sweeps = dict(**np.load(npz_path, allow_pickle=True))
        for sweep_type in all_sweeps.keys():
            sweeps, ylabel = all_sweeps[sweep_type]
            existing_sweeps, existing_ylabel = all_existing_sweeps[sweep_type]
            assert ylabel == existing_ylabel

            existing_sweeps.update(sweeps)
            all_existing_sweeps[sweep_type] = existing_sweeps, existing_ylabel

        jnp.savez(npz_path, **all_existing_sweeps)

    else:
        jnp.savez(npz_path, **all_sweeps)

    return all_sweeps


def train_fn(
    dataset: MultipleDomainDataset,
    train: ConcatDataset,
    joint_train: torch.Tensor,
    calibration: ConcatDataset,
    joint_calibration: torch.Tensor,
    train_fit_joint: bool,
    train_model: str,
    train_pretrained_path: Optional[Path],
    train_batch_size: int,
    train_epochs: int,
    train_decay: float,
    train_patience: int,
    train_tau: float,
    train_lr: float,
    calibration_batch_size: int,
    calibration_epochs: int,
    calibration_decay: float,
    calibration_patience: int,
    calibration_tau: float,
    calibration_lr: float,
    key: Any,
    generator: torch.Generator,
    device_count: int,
    num_workers: int,
) -> TrainState:
    if len(calibration) == 0 and calibration_epochs > 0:
        raise ValueError("Calibration set may not be empty")

    C, K = dataset.C, dataset.K
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
    if train_pretrained_path is not None:
        state = restore_train_state(state, train_pretrained_path)

    m = sha256()
    m.update(dataset.hexdigest.encode())
    m.update(str(train_fit_joint).encode())
    m.update(train_model.encode())
    m.update(str(train_pretrained_path).encode())
    train_key = (train_batch_size, train_epochs, train_decay, train_patience, train_tau, train_lr)
    m.update(str(train_key).encode())
    calibration_key = (calibration_batch_size, calibration_epochs, calibration_decay, calibration_patience, calibration_tau, calibration_lr)
    m.update(str(calibration_key).encode())
    m.update(str(key).encode())
    hexdigest = m.hexdigest()

    prefix = f"{dataset.__class__.__name__}_{dataset.train_domain}_{train_model}_{train_tau}_{calibration_tau}_{hexdigest}_"
    restored = restore_checkpoint("checkpoints/", state, prefix=prefix)
    if restored is not state:
        print(f"Restoring checkpoint with {prefix = }")

        # HACK: backward compatibility for legacy checkpoints
        # prior = restored.prior.unfreeze()
        # print('prior["source"]', prior["source"])
        # prior["source"] = jnp.ones_like(prior["source"])
        # prior["source"] = prior["source"] / jnp.sum(prior["source"])
        # print('prior["source"]', prior["source"])
        # restored = restored.replace(prior=flax.core.frozen_dict.freeze(prior))

        return replicate(restored)
    else:
        print(f"Cannot find checkpoint with {prefix = }")

    state: TrainState = replicate(state)

    train_loader = DataLoader(
        train,
        train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    if len(calibration) or calibration_epochs:
        calibration_loader = DataLoader(
            calibration,
            calibration_batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        )
    else:
        calibration_loader = None

    print("===> Training")
    joint_train_jnp = replicate(jnp.asarray(joint_train.flatten().numpy()))
    epoch_loss_valid_ema = None
    min_epoch_loss_valid_ema = float('inf')
    wait = 0
    for epoch in range(train_epochs):
        epoch_loss = 0
        epoch_hit = jnp.zeros(C * K, dtype=int)
        epoch_total = jnp.zeros(C * K, dtype=int)
        for X, _, Y, Z in train_loader:
            if X.shape[0] < device_count:
                continue

            remainder = X.shape[0] % device_count
            X = X[remainder:]
            Y = Y[remainder:]
            Z = Z[remainder:]

            X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
            Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])
            Z = jnp.array(Z).reshape(device_count, -1, *Z.shape[1:])
            M = Y * K + Z

            state, (loss, hit, total) = train_step(state, X, M, K, train_fit_joint, train_tau, joint_train_jnp)
            epoch_loss += unreplicate(loss)
            epoch_hit += unreplicate(hit)
            epoch_total += unreplicate(total)

        if calibration_loader is None:
            with jnp.printoptions(precision=3):
                print(
                    f"Train epoch {epoch + 1}, loss: {epoch_loss}, hit: {epoch_hit}, total: {epoch_total}"
                )

            continue

        epoch_loss_valid = 0
        for X, _, Y, Z in calibration_loader:
            if X.shape[0] < device_count:
                continue

            remainder = X.shape[0] % device_count
            X = X[remainder:]
            Y = Y[remainder:]
            Z = Z[remainder:]

            X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
            Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])
            Z = jnp.array(Z).reshape(device_count, -1, *Z.shape[1:])
            M = Y * K + Z

            loss_valid = validation_step(state, X, M, K, train_fit_joint, train_tau, joint_train_jnp)
            epoch_loss_valid += unreplicate(loss_valid)

        if epoch_loss_valid_ema is None:
            epoch_loss_valid_ema = epoch_loss_valid
        else:
            epoch_loss_valid_ema = (1 - train_decay) * epoch_loss_valid_ema + train_decay * epoch_loss_valid

        with jnp.printoptions(precision=3):
            print(
                f"Train epoch {epoch + 1}, loss: {epoch_loss:.2f} (val: {epoch_loss_valid:.2f}, ema: {epoch_loss_valid_ema:.2f}), hit: {epoch_hit}, total: {epoch_total}"
            )

        if epoch_loss_valid >= min_epoch_loss_valid_ema:
            wait += 1
        else:
            wait = 0
            min_epoch_loss_valid_ema = epoch_loss_valid_ema

        if wait > train_patience:
            print(f"Early stopping! {train_decay = }, {train_patience = }, {min_epoch_loss_valid_ema = }")
            break

    # Sync the batch statistics across replicas so that evaluation is deterministic.
    state = state.replace(batch_stats=cross_replica_mean(state.batch_stats))

    print("===> Calibrating")
    joint_calibration_jnp = replicate(jnp.asarray(joint_calibration.flatten().numpy()))
    epoch_loss_ema = None
    min_epoch_loss_ema = float('inf')
    wait = 0
    if calibration_loader is not None:
        for epoch in range(calibration_epochs):
            epoch_loss = 0
            epoch_hit = jnp.zeros(C * K, dtype=int)
            epoch_total = jnp.zeros(C * K, dtype=int)
            for X, _, Y, Z in calibration_loader:
                if X.shape[0] < device_count:
                    continue

                remainder = X.shape[0] % device_count
                X = X[remainder:]
                Y = Y[remainder:]
                Z = Z[remainder:]

                X = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
                Y = jnp.array(Y).reshape(device_count, -1, *Y.shape[1:])
                Z = jnp.array(Z).reshape(device_count, -1, *Z.shape[1:])
                M = Y * K + Z

                state, (loss, hit, total) = calibration_step(
                    state, X, M, K, train_fit_joint, calibration_tau, calibration_lr, joint_calibration_jnp
                )
                epoch_loss += unreplicate(loss)
                epoch_hit += unreplicate(hit)
                epoch_total += unreplicate(total)

            if epoch_loss_ema is None:
                epoch_loss_ema = epoch_loss
            else:
                epoch_loss_ema = (1 - calibration_decay) * epoch_loss_ema + calibration_decay * epoch_loss

            with jnp.printoptions(precision=3):
                print(
                    f"Calibration epoch {epoch + 1}, loss: {epoch_loss:.2f} (ema: {epoch_loss_ema:.2f}), hit: {epoch_hit}, total: {epoch_total}"
                )

            if epoch_loss_ema >= min_epoch_loss_ema:
                wait += 1
            else:
                wait = 0
                min_epoch_loss_ema = epoch_loss_ema

            if wait > calibration_patience:
                print(f"Early stopping! {calibration_decay = }, {calibration_patience = }, {min_epoch_loss_ema = }")
                break

    # Sync the batch statistics across replicas so that evaluation is deterministic.
    state = state.replace(batch_stats=cross_replica_mean(state.batch_stats))

    print("---> Temperature =", unreplicate(state.params["T"]))
    print("---> Bias =", unreplicate(state.params["b"]))

    if train_tau == 0 or calibration_tau == 0:
        # When doing logit adjustment, the source label distribution should be
        # uniform, as we effectively trained on an invariant domain. Since
        # "source" defaults to a uniform distribution, we only need to update
        # it when tau == 0.
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
            "induce",
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
            "count",
        )

        print("---> Induced source label prior =", source_prior_induced)
        print("---> Empirical source label prior =", source_prior_empirical)
        tvd = jnp.sum(jnp.abs(source_prior_induced - source_prior_empirical)) / 2
        print("---> Total variation distance =", tvd)

        prior = state.prior.unfreeze()
        prior["source"] = replicate(source_prior_induced)
        state = state.replace(prior=flax.core.frozen_dict.freeze(prior))

        save_checkpoint("checkpoints/", unreplicate(state), -tvd, prefix)
    else:
        save_checkpoint("checkpoints/", unreplicate(state), 0, prefix)

    return state


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


def baseline_fn(
    state: TrainState,
    dataset: MultipleDomainDataset,
    eval_splits: List[Tuple[Dataset, torch.Tensor]],
    dataset_label_noise: float,
    train_domains_set: Set[int],
    train_batch_size: int,
    calibration_domains_set: Set[int],
    adapt_skip_null_oracle: bool,
    adapt_gmtl_alpha: Sequence[float],
    generator: torch.Generator,
    device_count: int,
    num_workers: int,
):
    print("===> Adapting & Evaluating")

    mean_sweeps = {}
    l1_sweeps = {}
    auc_sweeps = {}
    auc_Z_sweeps = {}
    accuracy_sweeps = {}
    accuracy_Z_sweeps = {}
    norm_sweeps = {}

    adaptations: List[Adaptation]
    if adapt_skip_null_oracle:
        adaptations = []
    else:
        adaptations = [("Null",), ("Oracle",)]

    adaptations.extend(("GMTL", alpha) for alpha in adapt_gmtl_alpha)
    for adaptation in adaptations:
        argmax_joint = False
        batch_size = train_batch_size   # batch size does not matter since we are not adapting on data
        state, (mean, l1, auc, auc_Z, accuracy, accuracy_Z, norm) = adapt_fn(
            state,
            dataset.C,
            dataset.K,
            dataset_label_noise,
            train_domains_set,
            calibration_domains_set,
            eval_splits,
            adaptation,
            argmax_joint,
            batch_size,
            device_count,
            generator,
            num_workers,
        )
        mean_sweeps[adaptation, argmax_joint, batch_size] = mean
        l1_sweeps[adaptation, argmax_joint, batch_size] = l1
        auc_sweeps[adaptation, argmax_joint, batch_size] = auc
        auc_Z_sweeps[adaptation, argmax_joint, batch_size] = auc_Z
        accuracy_sweeps[adaptation, argmax_joint, batch_size] = accuracy
        accuracy_Z_sweeps[adaptation, argmax_joint, batch_size] = accuracy_Z
        norm_sweeps[adaptation, argmax_joint, batch_size] = norm

    return mean_sweeps, l1_sweeps, auc_sweeps, auc_Z_sweeps, accuracy_sweeps, accuracy_Z_sweeps, norm_sweeps


def adapt_fn(
    state: TrainState,
    C: int,
    K: int,
    dataset_label_noise: float,
    train_domains_set: Set[int],
    calibration_domains_set: Set[int],
    eval_splits: Sequence[Tuple[Dataset, torch.Tensor]],
    adaptation: Adaptation,
    argmax_joint: bool,
    batch_size: int,
    device_count: int,
    generator: torch.Generator,
    num_workers: int,
) -> Tuple[TrainState, Sweeps]:
    label = f"{adaptation = }, {argmax_joint = }, {batch_size = }"
    print(f"---> {label}")

    mean_sweep = jnp.empty(len(eval_splits))
    l1_sweep = jnp.empty(len(eval_splits))
    auc_sweep = jnp.empty(len(eval_splits))
    auc_Z_sweep = jnp.empty(len(eval_splits))
    accuracy_sweep = jnp.empty(len(eval_splits))
    accuracy_Z_sweep = jnp.empty(len(eval_splits))
    norm_sweep = jnp.empty(len(eval_splits))
    for i, (eval_, joint_M) in enumerate(eval_splits):
        # happens on the source domain when train_fraction = 1.0
        if len(eval_) == 0:
            mean_sweep = mean_sweep.at[i].set(jnp.nan)
            l1_sweep = l1_sweep.at[i].set(jnp.nan)
            auc_sweep = auc_sweep.at[i].set(jnp.nan)
            auc_Z_sweep = auc_Z_sweep.at[i].set(jnp.nan)
            accuracy_sweep = accuracy_sweep.at[i].set(jnp.nan)
            accuracy_Z_sweep = accuracy_Z_sweep.at[i].set(jnp.nan)
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
        prob = joint / jnp.sum(joint, axis=1, keepdims=True)
        prob = prob[:, 1, :]  # P(Y=1|Y_tilde, Z)

        # using shuffle=True so that Y contains multiple classes, otherwise AUC is not defined
        mean = l1 = hits = hits_Z = norm = 0
        epoch_Y = jnp.empty(len(eval_) // device_count * device_count)
        epoch_score = jnp.empty(len(eval_) // device_count * device_count)
        epoch_Z = jnp.empty(len(eval_) // device_count * device_count)
        epoch_score_Z = jnp.empty(len(eval_) // device_count * device_count)
        offset = 0

        eval_loader = DataLoader(
            eval_,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        )
        for X, Y_tilde, Y, Z in eval_loader:
            if X.shape[0] < device_count:
                continue

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

            epoch_Y = epoch_Y.at[offset : offset + N].set(Y.flatten())
            epoch_Z = epoch_Z.at[offset : offset + N].set(Z.flatten())

            if adaptation[0] == "Null":
                prior = state.prior.unfreeze()
                prior["target"] = prior["source"]
                state = state.replace(prior=flax.core.frozen_dict.freeze(prior))
            elif adaptation[0] == "Oracle":
                prior = state.prior.unfreeze()
                prior["target"] = replicate(joint_M.flatten())
                state = state.replace(prior=flax.core.frozen_dict.freeze(prior))
            elif adaptation[0] == "GMTL":
                _, alpha = adaptation
                prior = state.prior.unfreeze()
                target = prior["source"]**(1-alpha)
                target = target / jnp.sum(-1, keepdims=True)
                prior["target"] = target
                state = state.replace(prior=flax.core.frozen_dict.freeze(prior))
            elif adaptation[0] == "EM":
                _, prior_strength, symmetric_dirichlet, fix_marginal = adaptation
                state = adapt_step(
                    state,
                    X,
                    replicate(prior_strength),
                    symmetric_dirichlet,
                    fix_marginal,
                    C,
                    K,
                )
                prior = unreplicate(state.prior["target"]).reshape((C, K))
                print("prior", prior)
            else:
                raise ValueError(f"Unknown adaptation scheme {adaptation}")

            (score, hit), (score_Z, hit_Z) = test_step(state, X, Y, Z, argmax_joint)
            print("score", score.flatten())

            mean += jnp.sum(score)
            l1 += jnp.sum(jnp.abs(score.flatten() - prob[Y_tilde, Z.flatten()]))
            epoch_score = epoch_score.at[offset : offset + N].set(score.flatten())
            epoch_score_Z = epoch_score_Z.at[offset : offset + N].set(score_Z.flatten())
            hits += unreplicate(hit)
            hits_Z += unreplicate(hit_Z)
            prior = unreplicate(state.prior["target"]).reshape((C, K))
            norm += N * jnp.linalg.norm(prior - joint_M)

            offset += N

        mean = mean / len(eval_)
        l1 = l1 / len(eval_)
        auc = roc_auc_score(epoch_Y, epoch_score)
        auc_Z = roc_auc_score(epoch_Z, epoch_score_Z)
        accuracy = hits / len(eval_)
        accuracy_Z = hits_Z / len(eval_)
        norm = norm / len(eval_)

        with jnp.printoptions(precision=4):
            print(
                f"[{label}] Environment {i:>2} {seen} mean {mean}, L1 {l1}, AUC {auc} ({auc_Z}), Accuracy {accuracy} ({accuracy_Z}), Norm {norm}"
            )

        # note that foo_sweep.at[-1] is the training foo
        mean_sweep = mean_sweep.at[i].set(mean)
        l1_sweep = l1_sweep.at[i].set(l1)
        auc_sweep = auc_sweep.at[i].set(auc)
        auc_Z_sweep = auc_Z_sweep.at[i].set(auc_Z)
        accuracy_sweep = accuracy_sweep.at[i].set(accuracy)
        accuracy_Z_sweep = accuracy_Z_sweep.at[i].set(accuracy_Z)
        norm_sweep = norm_sweep.at[i].set(norm)

    print(
        f"[{label}] Average response {jnp.nanmean(mean_sweep[:-1])}, "
        f"Average L1 {jnp.nanmean(l1_sweep[:-1])}, "
        f"Average AUC {jnp.nanmean(auc_sweep[:-1])} ({jnp.nanmean(auc_Z_sweep[:-1])}), "
        f"Accuracy {jnp.nanmean(accuracy_sweep[:-1])} ({jnp.nanmean(accuracy_Z_sweep[:-1])}), "
        f"Norm {jnp.nanmean(norm_sweep[:-1])}"
    )

    return state, (
        mean_sweep,
        l1_sweep,
        auc_sweep,
        auc_Z_sweep,
        accuracy_sweep,
        accuracy_Z_sweep,
        norm_sweep,
    )


if __name__ == "__main__":
    initialize_cache("jit_cache")
    latexify(width_scale_factor=2, fig_height=2)
    cli()
