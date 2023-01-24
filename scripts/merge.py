from typing import Mapping, Tuple
from pathlib import Path
import re

import click
import numpy as np
import matplotlib.pyplot as plt

from tta.common import Curves
from tta.visualize import latexify, format_axes


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
        npz = np.load(npz_path)
        print(f"Reading from {npz_path}")
        npz = np.load(npz_path, allow_pickle=True)
        npz_dict[npz_path.stem] = npz

    confounder_strength = np.linspace(0, 1, 21)

    plot(
        npz_dict,
        confounder_strength,
        merged_title,
        merged_root,
        merged_name,
    )


def key(path: Path) -> Tuple[int, int, float]:
    domain, sub, tau = parse(path.stem)
    mapping = {
        "none": 0,
        "classes": 1,
        "groups": 2,
    }
    return domain, mapping[sub], tau


def parse(config: str) -> Tuple[int, str, float]:
    if config.startswith("mnist_"):
        pattern = re.compile(r"mnist_rot(True|False)_noise(\d*\.?\d*)_domain(\d+)_sub(none|classes|groups)_tau(\d*\.?\d*)_train(\d+)_cali(\d+)_prior(\d*\.?\d*)")
        matching = pattern.fullmatch(config)
        assert matching is not None

        rot = bool(matching.group(1))
        noise = float(matching.group(2))
        domain = int(matching.group(3))
        sub = matching.group(4)
        tau = float(matching.group(5))
        train = int(matching.group(6))
        cali = int(matching.group(7))
        prior = float(matching.group(8))

    elif config.startswith("chexpert-") or config.startswith("mimic-"):
        pattern = re.compile(r"(chexpert|mimic)-(embedding|pixel)_([a-zA-Z]+)_([a-zA-Z]+)_domain(\d+)_size(\d+)_sub(none|classes|groups)_tau(\d*\.?\d*)_train(\d+)_cali(\d+)_prior(\d*\.?\d*)")
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

    else:
        raise ValueError(f"Unknown config {config}")

    return domain, sub, tau


def plot(
        npz_dict: Mapping[str, Curves],
        confounder_strength: np.ndarray,
        merged_title: str,
        merged_root: Path,
        merged_name: str,
    ):
    example = next(iter(npz_dict.values()))
    sweep_types = example.keys()

    for sweep_type in sweep_types:
        fig, ax = plt.subplots(figsize=(12, 6))
    
        _, ylabel = example[sweep_type]
        ylabel = ylabel.replace("Average AUC", "AUC")

        invariance_curves_labels = [], []
        adaptation_curves_labels = [], []
        oracle_curves_labels = [], []
        for config, all_sweeps in npz_dict.items():
            domain, sub, tau = parse(config)

            styles = {
                ("none", 0.0): ("C0", "o", "Vanilla"),
                ("groups", 0.0): ("C1", "^", "SUBG"),
                ("none", 1.0): ("C2", "s", "Logit Adjustment"),
            }
            if (sub, tau) not in styles:
                continue

            color, marker, base_label = styles[(sub, tau)]
            markerfacecolor = color

            sweeps, ylabel = all_sweeps[sweep_type]
            for ((algo, *param), argmax_joint, batch_size), sweep in sweeps.items():
                assert not argmax_joint
                if algo == "Null":
                    linestyle = "dotted"
                    label = base_label
                    curves, labels = invariance_curves_labels
                elif algo == "EM" and batch_size == 512:
                    linestyle = "dashed"
                    label = f"MLE on {base_label}"
                    curves, labels = adaptation_curves_labels
                elif algo == "Oracle":
                    linestyle = "solid"
                    label = f"Oracle on {base_label}"
                    curves, labels = oracle_curves_labels
                else:
                    continue

                curve, = ax.plot(confounder_strength, sweep[:-1],
                        linestyle=linestyle, marker=marker, linewidth=1, markersize=2,
                        color=color, markerfacecolor=markerfacecolor, alpha=1.0)
                curves.append(curve)
                labels.append(label)

            ax.axvline(confounder_strength[domain], color="black", linestyle="dotted", linewidth=3)

        # plt.ylim((0, 1))
        if sweep_type in {"mean", "l1", "norm"}:
            plt.ylim((0, 1))
        else:
            plt.ylim((0.5, 1))

        plt.xlabel("Shift parameter")
        plt.ylabel(ylabel)
        plt.title(merged_title)
        plt.grid(True)
        legend1 = plt.legend(*invariance_curves_labels, loc="upper left", bbox_to_anchor=(0, -0.15), ncol=1, frameon=False)
        legend2 = plt.legend(*adaptation_curves_labels, loc="upper left", bbox_to_anchor=(1/3, -0.15), ncol=1, frameon=False)
        plt.legend(*oracle_curves_labels, loc="upper left", bbox_to_anchor=(2/3, -0.15), ncol=1, frameon=False)
        plt.gca().add_artist(legend1)
        plt.gca().add_artist(legend2)
        fig.tight_layout()

        format_axes(ax)
        for suffix in ("png", "pdf"):
            plt.savefig(merged_root / f"{merged_name}_{sweep_type}.{suffix}", bbox_inches='tight', dpi=300)

        plt.close(fig)


if __name__ == "__main__":
    latexify(width_scale_factor=2, fig_height=2)
    merge()
