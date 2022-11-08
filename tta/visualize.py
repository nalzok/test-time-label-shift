from typing import Set, Union, Dict, Tuple

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .common import Curves


def plot(
    all_sweeps: Dict[str, Tuple[Curves, str]],
    train_batch_size: int,
    confounder_strength: np.ndarray,
    train_domains_set: Set[int],
    dataset_label_noise: float,
    plot_title: str,
    plot_root: Path,
    config_name: str,
):
    for sweep_type, (sweeps, ylabel) in all_sweeps.items():
        fig, ax = plt.subplots(figsize=(12, 6))

        if sweep_type == "accuracy":
            if dataset_label_noise > 0:
                upper_bound = bayes_accuracy(dataset_label_noise, confounder_strength)
                ax.plot(
                    confounder_strength,
                    upper_bound,
                    color="grey",
                    linestyle=":",
                    label="Upper bound",
                )

        oracle_sweep = sweeps.pop(("Oracle", None, train_batch_size))
        ax.plot(confounder_strength, oracle_sweep[:-1], linestyle="--", label="Oracle")
        unadapted_sweep = sweeps.pop(("Unadapted", None, train_batch_size))
        ax.plot(
            confounder_strength, unadapted_sweep[:-1], linestyle="--", label="Unadapted"
        )

        for (label, _, _), sweep in sweeps.items():
            ax.plot(confounder_strength, sweep[:-1], label=label)

        for i in train_domains_set:
            ax.axvline(confounder_strength[i], linestyle=":")

        plt.ylim((0, 1))
        plt.xlabel("Shift parameter")
        plt.ylabel(ylabel)
        plt.title(plot_title)
        plt.grid(True)
        plt.legend()

        for suffix in ("png", "pdf"):
            plt.savefig(plot_root / f"{config_name}_{sweep_type}.{suffix}", dpi=300)

        plt.close(fig)


def bayes_accuracy(
    dataset_label_noise: float, confounder_strength: Union[float, np.ndarray]
) -> np.ndarray:
    upper_bound = np.maximum(
        np.maximum(1 - confounder_strength, confounder_strength),
        (1 - dataset_label_noise) * np.ones_like(confounder_strength),
    )
    return upper_bound
