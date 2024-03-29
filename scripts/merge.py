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
ConfigKey = Tuple[bool, Dataset, int, str, float, int]
AdaptKey = Tuple[Adaptation, bool, int]


@click.command()
@click.option("--npz_pattern", type=str, required=True)
@click.option("--merged_title", type=str, required=True)
@click.option("--merged_name", type=str, required=True)
def merge(
        npz_pattern: str,
        merged_title: str,
        merged_name: str,
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
    )


def key(path: Path) -> Tuple[bool, Dataset, int, int, float, int]:
    is_tree, dataset, domain, sub, tau, cali = parse(path.stem)
    mapping = {
        "none": 0,
        "classes": 1,
        "groups": 2,
    }
    return is_tree, dataset, domain, mapping[sub], tau, cali


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
        pattern = re.compile(r"^mnist_rot(True|False)_noise(\d*\.?\d*)_domain(\d+)_sub(none|classes|groups)_tau(\d*\.?\d*)_train(\d+)_cali(\d+)_prior(\d*\.?\d*)_seed(\d*\.?\d*)$")
        matching = pattern.fullmatch(config)
        assert matching is not None

        is_tree = False
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

    elif config.startswith("tree_mnist"):
        pattern = re.compile(r"^tree_mnist_rot(True|False)_noise(\d*\.?\d*)_domain(\d+)_prior(\d*\.?\d*)_seed(\d*\.?\d*)$")
        matching = pattern.fullmatch(config)
        assert matching is not None

        is_tree = True
        dataset = "mnist"
        rot = bool(matching.group(1))
        noise = float(matching.group(2))
        domain = int(matching.group(3))
        sub = "none"
        tau = 0
        cali = 0
        prior = float(matching.group(4))
        seed = int(matching.group(5))

    elif config.startswith("chexpert-") or config.startswith("mimic-"):
        pattern = re.compile(r"^(chexpert|mimic)-(embedding|pixel)_([a-zA-Z]+)_([a-zA-Z]+)_domain(\d+)_size(\d+)_sub(none|classes|groups)_tau(\d*\.?\d*)_train(\d+)_cali(\d+)_prior(\d*\.?\d*)_seed(\d*\.?\d*)$")
        matching = pattern.fullmatch(config)
        assert matching is not None

        is_tree = False
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

    elif config.startswith("tree_chexpert-") or config.startswith("tree_mimic-"):
        pattern = re.compile(r"^tree_(chexpert|mimic)-(embedding|pixel)_([a-zA-Z]+)_([a-zA-Z]+)_domain(\d+)_size(\d+)_prior(\d*\.?\d*)_seed(\d*\.?\d*)$")
        matching = pattern.fullmatch(config)
        assert matching is not None

        is_tree = True
        dataset = matching.group(1)
        modality = matching.group(2)
        Y_column = matching.group(3)
        Z_column = matching.group(4)
        domain = int(matching.group(5))
        size = int(matching.group(6))
        sub = "none"
        tau = 0
        cali = 0
        prior = float(matching.group(7))
        seed = int(matching.group(8))

    else:
        raise ValueError(f"Unknown config {config}")

    return is_tree, dataset, domain, sub, tau, cali


def plot(
        ylabels: Dict[str, str],
        type2config2adapt2sweeps: Dict[str, Dict[ConfigKey, Dict[AdaptKey, List[jnp.ndarray]]]],
        confounder_strength: np.ndarray,
        merged_title: str,
        merged_root: Path,
        merged_name: str,
    ):
    tab20 = plt.get_cmap("tab20").colors
    meta_styles = {
        "major": {
            ("none", 0.0): ("solid", "o", "ERM"),
            ("none", 1.0): ("solid", "o", "Logit Adjustment"),
            ("groups", 0.0): ("dashed", "^", "SUBG"),
        },
    }
    for meta_type in meta_styles.keys():
        styles = meta_styles[meta_type]
        for sweep_type, ylabel in ylabels.items():
            fig, ax = plt.subplots(figsize=(12, 6))

            erm_curves_labels = [], []
            invariance_curves_labels = [], []
            adaptation_curves_labels = [], []

            config2adapt2sweeps = type2config2adapt2sweeps[sweep_type]
            for (is_tree, dataset, domain, sub, tau, cali), adapt2sweeps in config2adapt2sweeps.items():
                ax.axvline(confounder_strength[domain], color="black", linestyle="dotted", linewidth=3)

                style = styles.get((sub, tau))
                if style is None:
                    continue

                linestyle, marker, base_label = style

                for ((algo, *param), argmax_joint, batch_size), sweeps in adapt2sweeps.items():
                    assert not argmax_joint

                    adapt_on = "ERM" if is_tree else "Logit Adjustment"
                    if meta_type == "major":
                        if algo == "Null" and base_label == "ERM":
                            label = base_label
                            curves, labels = erm_curves_labels
                            markerfacecolor = color = "black"
                        elif algo == "Null" and base_label != "ERM":
                            label = base_label
                            curves, labels = invariance_curves_labels
                            markerfacecolor = color = tab20[6] if base_label == "Logit Adjustment" else tab20[2]
                        elif algo == "EM" and base_label == adapt_on and batch_size >= 64:
                            label = f"TTLSA (batch size {batch_size})"
                            curves, labels = adaptation_curves_labels
                            if batch_size == 64:
                                markerfacecolor = color = tab20[19]
                            elif batch_size >= 512:
                                markerfacecolor = color = tab20[18]
                            else:
                                raise ValueError(f"Unknown batch size {batch_size}")
                        elif algo == "Oracle" and base_label == adapt_on:
                            label = "TTLSA (oracle)"
                            curves, labels = adaptation_curves_labels
                            markerfacecolor = color = tab20[0]
                        else:
                            continue
                    else:
                        raise ValueError(f"Unknown meta type {meta_type}")

                    if algo == "TTLSA (oracle)":
                        linewidth = 1.5
                        markersize = 6
                    else:
                        linewidth = 1
                        markersize = 4

                    jitter = {
                        "Null": 0,
                        "EM": -0.005,
                        "Oracle": 0,
                    }
                    confounder_strength_jitted = confounder_strength + jitter[algo]

                    sweep_mean, sweep_std = mean_std(sweeps)
                    curve = ax.errorbar(confounder_strength_jitted, sweep_mean, sweep_std,
                            linestyle=linestyle, marker=marker,
                            linewidth=linewidth, markersize=markersize,
                            color=color, markerfacecolor=markerfacecolor, alpha=1.0)
                    curves.append(curve)
                    labels.append(label)

                if sweep_type in {"mean", "l1", "norm"}:
                    plt.ylim((0, 1))
                elif is_tree:
                    auc_limit = 0.9 if dataset == "mnist" else 0.7
                    plt.ylim((auc_limit, 1))
                else:
                    auc_limit = 0.98 if dataset == "mnist" else 0.7
                    plt.ylim((auc_limit, 1))

                plt.xlabel("Shift parameter")
                if cali == 0:
                    plt.ylabel(f"{ylabel} (without calibration)")
                else:
                    plt.ylabel(ylabel)

            plt.title(merged_title)
            plt.grid(True, alpha=0.5)

            if meta_type == "major":
                # HACK
                invariance_curves, invariance_labels = invariance_curves_labels
                if len(invariance_curves) >= 3 and len(invariance_labels) >= 3:
                    invariance_curves[0], invariance_curves[1] = invariance_curves[1], invariance_curves[0]
                    invariance_labels[0], invariance_labels[1] = invariance_labels[1], invariance_labels[0]

                adaptation_curves, adaptation_labels = adaptation_curves_labels
                if len(adaptation_curves) >= 3 and len(adaptation_labels) >= 3:
                    adaptation_curves[-1], adaptation_curves[-2] = adaptation_curves[-2], adaptation_curves[-1]
                    adaptation_labels[-1], adaptation_labels[-2] = adaptation_labels[-2], adaptation_labels[-1]

                legend1 = plt.legend(*adaptation_curves_labels, loc="upper left", bbox_to_anchor=(2/3, -0.15), ncol=1, frameon=False)
                legend2 = plt.legend(*invariance_curves_labels, loc="upper left", bbox_to_anchor=(1/3, -0.15), ncol=1, frameon=False)
                plt.legend(*erm_curves_labels, loc="upper left", bbox_to_anchor=(0, -0.15), ncol=1, frameon=False)
                plt.gca().add_artist(legend1)
                plt.gca().add_artist(legend2)
            else:
                raise ValueError(f"Unknown meta type {meta_type}")

            fig.tight_layout()

            format_axes(ax)
            for suffix in ("png", "pdf"):
                plt.savefig(merged_root / f"{merged_name}_{sweep_type}_{meta_type}.{suffix}", bbox_inches="tight", dpi=300)

            plt.close(fig)


def mean_std(sweeps: List[jnp.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    sweeps_array = np.empty((len(sweeps), len(sweeps[0]) - 1))
    for i, sweep in enumerate(sweeps):
        sweeps_array[i, :] = sweep[:-1]

    mean = np.mean(sweeps_array, axis=0)
    std = np.std(sweeps_array, axis=0) / np.sqrt(len(sweeps))
    return mean, std


if __name__ == "__main__":
    latexify(width_scale_factor=2, fig_height=2)
    np.random.seed(42)
    merge()
