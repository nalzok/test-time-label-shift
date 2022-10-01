from typing import Set, Union

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .common import Curves


def plot_mean(
    mean_sweeps: Curves,
    train_batch_size: int,
    confounder_strength: np.ndarray,
    train_domains_set: Set[int],
    plot_title: str,
    plot_path: Path,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    oracle_mean_sweep = mean_sweeps.pop(("Oracle", None, train_batch_size))
    ax.plot(confounder_strength, oracle_mean_sweep[:-1], linestyle="--", label="Oracle")
    unadapted_mean_sweep = mean_sweeps.pop(("Unadapted", None, train_batch_size))
    ax.plot(
        confounder_strength, unadapted_mean_sweep[:-1], linestyle="--", label="Unadapted"
    )

    for (label, _, _), mean_sweep in mean_sweeps.items():
        ax.plot(confounder_strength, mean_sweep[:-1], label=label)

    for i in train_domains_set:
        ax.axvline(confounder_strength[i], linestyle=":")

    plt.ylim((0, 1))
    plt.xlabel("Shift parameter")
    plt.ylabel("Average probability of class 1")
    plt.title(plot_title)
    plt.grid(True)
    plt.legend()

    plt.savefig(plot_path, dpi=300)
    plt.close(fig)


def plot_l1(
    l1_sweeps: Curves,
    train_batch_size: int,
    confounder_strength: np.ndarray,
    train_domains_set: Set[int],
    plot_title: str,
    plot_path: Path,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    oracle_l1_sweep = l1_sweeps.pop(("Oracle", None, train_batch_size))
    ax.plot(confounder_strength, oracle_l1_sweep[:-1], linestyle="--", label="Oracle")
    unadapted_l1_sweep = l1_sweeps.pop(("Unadapted", None, train_batch_size))
    ax.plot(
        confounder_strength, unadapted_l1_sweep[:-1], linestyle="--", label="Unadapted"
    )

    for (label, _, _), l1_sweep in l1_sweeps.items():
        ax.plot(confounder_strength, l1_sweep[:-1], label=label)

    for i in train_domains_set:
        ax.axvline(confounder_strength[i], linestyle=":")

    plt.ylim((0, 1))
    plt.xlabel("Shift parameter")
    plt.ylabel("Average L1 error of class 1")
    plt.title(plot_title)
    plt.grid(True)
    plt.legend()

    plt.savefig(plot_path, dpi=300)
    plt.close(fig)


def plot_auc(
    auc_sweeps: Curves,
    train_batch_size: int,
    confounder_strength: np.ndarray,
    train_domains_set: Set[int],
    plot_title: str,
    plot_path: Path,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    oracle_auc_sweep = auc_sweeps.pop(("Oracle", None, train_batch_size))
    ax.plot(confounder_strength, oracle_auc_sweep[:-1], linestyle="--", label="Oracle")
    unadapted_auc_sweep = auc_sweeps.pop(("Unadapted", None, train_batch_size))
    ax.plot(
        confounder_strength, unadapted_auc_sweep[:-1], linestyle="--", label="Unadapted"
    )

    for (label, _, _), auc_sweep in auc_sweeps.items():
        ax.plot(confounder_strength, auc_sweep[:-1], label=label)

    for i in train_domains_set:
        ax.axvline(confounder_strength[i], linestyle=":")

    plt.ylim((0, 1))
    plt.xlabel("Shift parameter")
    plt.ylabel("Average AUC")
    plt.title(plot_title)
    plt.grid(True)
    plt.legend()

    plt.savefig(plot_path, dpi=300)
    plt.close(fig)


def plot_accuracy(
    accuracy_sweeps: Curves,
    train_batch_size: int,
    confounder_strength: np.ndarray,
    train_domains_set: Set[int],
    dataset_label_noise: float,
    plot_title: str,
    plot_path: Path,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    if dataset_label_noise > 0:
        upper_bound = bayes_accuracy(dataset_label_noise, confounder_strength)
        ax.plot(
            confounder_strength,
            upper_bound,
            color="grey",
            linestyle=":",
            label="Upper bound",
        )

    oracle_accuracy_sweep = accuracy_sweeps.pop(("Oracle", None, train_batch_size))
    ax.plot(
        confounder_strength, oracle_accuracy_sweep[:-1], linestyle="--", label="Oracle"
    )
    unadapted_accuracy_sweep = accuracy_sweeps.pop(
        ("Unadapted", None, train_batch_size)
    )
    ax.plot(
        confounder_strength,
        unadapted_accuracy_sweep[:-1],
        linestyle="--",
        label="Unadapted",
    )

    for (label, _, _), accuracy_sweep in accuracy_sweeps.items():
        ax.plot(confounder_strength, accuracy_sweep[:-1], label=label)

    for i in train_domains_set:
        ax.axvline(confounder_strength[i], linestyle=":")

    plt.ylim((0, 1))
    plt.xlabel("Shift parameter")
    plt.ylabel("Accuracy")
    plt.title(plot_title)
    plt.grid(True)
    plt.legend()

    plt.savefig(plot_path, dpi=300)
    plt.close(fig)


def plot_norm(
    norm_sweeps: Curves,
    train_batch_size: int,
    confounder_strength: np.ndarray,
    train_domains_set: Set[int],
    plot_title: str,
    plot_path: Path,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    oracle_norm_sweep = norm_sweeps.pop(("Oracle", None, train_batch_size))
    ax.plot(confounder_strength, oracle_norm_sweep[:-1], linestyle="--", label="Oracle")
    unadapted_norm_sweep = norm_sweeps.pop(("Unadapted", None, train_batch_size))
    ax.plot(
        confounder_strength,
        unadapted_norm_sweep[:-1],
        linestyle="--",
        label="Unadapted",
    )

    for (label, _, _), norm_sweep in norm_sweeps.items():
        ax.plot(confounder_strength, norm_sweep[:-1], label=label)

    for i in train_domains_set:
        ax.axvline(confounder_strength[i], linestyle=":")

    plt.ylim((0, 1))
    plt.xlabel("Shift parameter")
    plt.ylabel("Euclidean distance")
    plt.title(plot_title)
    plt.grid(True)
    plt.legend()

    plt.savefig(plot_path, dpi=300)
    plt.close(fig)


def bayes_accuracy(
    dataset_label_noise: float, confounder_strength: Union[float, np.ndarray]
) -> np.ndarray:
    upper_bound = np.maximum(
        np.maximum(1 - confounder_strength, confounder_strength),
        (1 - dataset_label_noise) * np.ones_like(confounder_strength),
    )
    return upper_bound
