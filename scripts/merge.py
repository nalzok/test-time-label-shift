from typing import Literal, Dict, Tuple, Union, List
from pathlib import Path
from collections import defaultdict
import re

import click
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from tta.common import Adaptation, Curves
from tta.visualize import latexify, format_axes


Dataset = Union[Literal["mnist"], Literal["chexpert"], Literal["mimic"]]
ConfigKey = Tuple[Dataset, int, str, float]
AdaptKey = Tuple[Adaptation, bool, int]


@click.command()
@click.option("--npz_pattern", type=str, required=True)
@click.option("--merged_title", type=str, required=True)
@click.option("--merged_name", type=str, required=True)
@click.option("--descriptive_name", type=str, required=True)
def merge(
        npz_pattern: str,
        merged_title: str,
        merged_name: str,
        descriptive_name: str,
    ) -> None:
    npz_root = Path("npz/")
    merged_root = Path("merged/")

    npz_root.mkdir(parents=True, exist_ok=True)
    merged_root.mkdir(parents=True, exist_ok=True)

    npz_dict = {}
    for npz_path in sorted(npz_root.glob(npz_pattern), key=key):
        print(f"Reading from {npz_path}")
        npz = np.load(npz_path, allow_pickle=True)
        npz_dict[npz_path.stem] = npz

    ylabels, type2config2adapt2sweeps = collect(npz_dict)
    confounder_strength = np.linspace(0, 1, 21)

    plot(
        ylabels,
        type2config2adapt2sweeps,
        confounder_strength,
        merged_title,
        merged_root,
        merged_name,
        descriptive_name,
    )


def key(path: Path) -> Tuple[Dataset, int, int, float]:
    dataset, domain, sub, tau = parse(path.stem)
    mapping = {
        "none": 0,
        "classes": 1,
        "groups": 2,
    }
    return dataset, domain, mapping[sub], tau


def collect(npz_dict: Dict[str, Dict[str, Tuple[Curves, str]]]) -> Tuple[
        Dict[str, str], Dict[str, Dict[ConfigKey, Dict[AdaptKey, List[jnp.ndarray]]]],
    ]:
    example = next(iter(npz_dict.values()))
    ylabels = {k: v.replace("Average AUC", "AUC") for k, (_, v) in example.items()}

    type2config2adapt2sweeps: Dict[str, Dict[ConfigKey, Dict[AdaptKey, List[jnp.ndarray]]]] \
            = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sweep_type in ylabels.keys():
        for config, type2adapt2sweeps in npz_dict.items():
            config_key = parse(config)
            sweeps, _ = type2adapt2sweeps[sweep_type]
            for adapt_key, sweep in sweeps.items():
                type2config2adapt2sweeps[sweep_type][config_key][adapt_key].append(sweep)

    return ylabels, type2config2adapt2sweeps


def parse(config: str) -> ConfigKey:
    if config.startswith("mnist_"):
        pattern = re.compile(r"mnist_rot(True|False)_noise(\d*\.?\d*)_domain(\d+)_sub(none|classes|groups)_tau(\d*\.?\d*)_train(\d+)_cali(\d+)_prior(\d*\.?\d*)_seed(\d*\.?\d*)")
        matching = pattern.fullmatch(config)
        assert matching is not None

        dataset = "mnist"
        rot = bool(matching.group(1))
        noise = float(matching.group(2))
        domain = int(matching.group(3))
        sub = matching.group(4)
        tau = float(matching.group(5))
        train = int(matching.group(6))
        cali = int(matching.group(7))
        prior = float(matching.group(8))
        seed = int(matching.group(9))

    elif config.startswith("chexpert-") or config.startswith("mimic-"):
        pattern = re.compile(r"(chexpert|mimic)-(embedding|pixel)_([a-zA-Z]+)_([a-zA-Z]+)_domain(\d+)_size(\d+)_sub(none|classes|groups)_tau(\d*\.?\d*)_train(\d+)_cali(\d+)_prior(\d*\.?\d*)_seed(\d*\.?\d*)")
        matching = pattern.fullmatch(config)
        assert matching is not None

        dataset = matching.group(1)
        modality = matching.group(2)
        Y_column = matching.group(3)
        Z_column = matching.group(4)
        domain = int(matching.group(5))
        size = int(matching.group(6))
        sub = matching.group(7)
        tau = float(matching.group(8))
        train = int(matching.group(9))
        cali = int(matching.group(10))
        prior = float(matching.group(11))
        seed = int(matching.group(12))

    else:
        raise ValueError(f"Unknown config {config}")

    return dataset, domain, sub, tau


def plot(
        ylabels: Dict[str, str],
        type2config2adapt2sweeps: Dict[str, Dict[ConfigKey, Dict[AdaptKey, List[jnp.ndarray]]]],
        confounder_strength: np.ndarray,
        merged_title: str,
        merged_root: Path,
        merged_name: str,
        descriptive_name: str,
    ):
    meta_styles = {
        "model": {
            ("none", 0.0): ("C0", "o", "Vanilla"),
            ("groups", 0.0): ("C1", "^", "SUBG"),
            ("none", 1.0): ("C2", "s", "Logit Adjustment"),
        },
        "batch": {
            ("none", 1.0): ("C2", "s", "Logit Adjustment"),
        }
    }
    for meta_type in meta_styles.keys():
        styles = meta_styles[meta_type]
        for sweep_type, ylabel in ylabels.items():
            fig, ax = plt.subplots(figsize=(12, 6))

            invariance_curves_labels = [], []
            adaptation_curves_labels = [], []
            oracle_curves_labels = [], []

            config2adapt2sweeps = type2config2adapt2sweeps[sweep_type]
            for (dataset, domain, sub, tau), adapt2sweeps in config2adapt2sweeps.items():
                ax.axvline(confounder_strength[domain], color="black", linestyle="dotted", linewidth=3)

                style = styles.get((sub, tau))
                if style is None:
                    continue

                color, marker, base_label = style
                markerfacecolor = color

                for ((algo, *param), argmax_joint, batch_size), sweeps in adapt2sweeps.items():
                    assert not argmax_joint

                    if meta_type == "model":
                        if algo == "Null":
                            linestyle = "dotted"
                            label = base_label
                            curves, labels = invariance_curves_labels
                        elif algo == "EM" and batch_size == 512:
                            linestyle = "dashed"
                            label = f"TTA on {base_label}"
                            curves, labels = adaptation_curves_labels
                        elif algo == "Oracle":
                            linestyle = "solid"
                            label = f"Oracle on {base_label}"
                            curves, labels = oracle_curves_labels
                        else:
                            continue
                    elif meta_type == "batch":
                        # NOTE: sorry for the ugly hack...
                        tab20c = plt.get_cmap("tab20c")
                        if algo == "Null":
                            linestyle = "dotted"
                            label = base_label
                            curves, labels = invariance_curves_labels
                            markerfacecolor = color = tab20c.colors[19]
                        elif algo == "EM":
                            linestyle = "dashed"
                            label = f"TTA with batch size {batch_size}"
                            curves, labels = adaptation_curves_labels
                            markerfacecolor = color = tab20c.colors[19 - int(np.log2(batch_size) / 3)]
                        elif algo == "Oracle":
                            linestyle = "solid"
                            label = f"Oracle"
                            curves, labels = oracle_curves_labels
                            markerfacecolor = color = tab20c.colors[16]
                        else:
                            continue
                    else:
                        raise ValueError(f"Unknown meta type {meta_type}")

                    sweep_mean, sweep_std = mean_std(sweeps)
                    confounder_strength_jitted = rand_jitter(confounder_strength)
                    curve = ax.errorbar(confounder_strength_jitted, sweep_mean, sweep_std,
                            linestyle=linestyle, marker=marker, linewidth=1, markersize=4,
                            color=color, markerfacecolor=markerfacecolor, alpha=1.0)
                    curves.append(curve)
                    labels.append(label)

                if sweep_type in {"mean", "l1", "norm"}:
                    plt.ylim((0, 1))
                else:
                    auc_limit = 0.98 if dataset == "MNIST" else 0.7
                    plt.ylim((auc_limit, 1))

            plt.xlabel("Shift parameter")
            plt.ylabel(f"{ylabel} ({descriptive_name})")
            plt.title(merged_title)
            plt.grid(True, alpha=0.5)
            legend1 = plt.legend(*invariance_curves_labels, loc="upper left", bbox_to_anchor=(0, -0.15), ncol=1, frameon=False)
            legend2 = plt.legend(*adaptation_curves_labels, loc="upper left", bbox_to_anchor=(1/3, -0.15), ncol=1, frameon=False)
            plt.legend(*oracle_curves_labels, loc="upper left", bbox_to_anchor=(2/3, -0.15), ncol=1, frameon=False)
            plt.gca().add_artist(legend1)
            plt.gca().add_artist(legend2)
            fig.tight_layout()

            format_axes(ax)
            for suffix in ("png", "pdf"):
                plt.savefig(merged_root / f"{merged_name}_{sweep_type}_{meta_type}.{suffix}", dpi=300)

            plt.close(fig)


def mean_std(sweeps: List[jnp.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    sweeps_array = np.empty((len(sweeps), len(sweeps[0]) - 1))
    for i, sweep in enumerate(sweeps):
        sweeps_array[i, :] = sweep[:-1] + 0.01 * np.random.randn(len(sweeps[0]) - 1)

    mean = np.mean(sweeps_array, axis=0)
    std = np.std(sweeps_array, axis=0)
    return mean, std


# https://stackoverflow.com/a/21276920
def rand_jitter(arr):
    stdev = .002 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


if __name__ == "__main__":
    latexify(width_scale_factor=2, fig_height=2)
    np.random.seed(42)
    merge()
