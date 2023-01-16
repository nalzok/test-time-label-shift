from typing import Mapping
from pathlib import Path
import re

import click
import numpy as np
import matplotlib.pyplot as plt

from tta.common import Curves
from tta.visualize import latexify, format_axes


@click.command()
@click.option("--npz_pattern", type=str, required=True)
@click.option("--train_domain", type=int, required=True)
@click.option("--merged_title", type=str, required=True)
@click.option("--merged_name", type=str, required=True)
def merge(
        npz_pattern: str,
        train_domain: int,
        merged_title: str,
        merged_name: str,
    ) -> None:
    npz_root = Path("npz/")
    merged_root = Path("merged/")

    npz_root.mkdir(parents=True, exist_ok=True)
    merged_root.mkdir(parents=True, exist_ok=True)

    npz_dict = {}
    for npz_path in npz_root.glob(npz_pattern):
        npz = np.load(npz_path)
        print(f"Reading from {npz_path}")
        npz = np.load(npz_path, allow_pickle=True)
        npz_dict[npz_path.stem] = npz

    confounder_strength = np.linspace(0, 1, 21)

    plot(
        npz_dict,
        confounder_strength,
        train_domain,
        merged_title,
        merged_root,
        merged_name,
    )


def plot(
        npz_dict: Mapping[str, Curves],
        confounder_strength: np.ndarray,
        train_domain: int,
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

        baseline_curves_labels = [], []
        for config, all_sweeps in npz_dict.items():
            sweeps, ylabel = all_sweeps[sweep_type]
            sweep = sweeps[("Null",), False, 64]

            tab20c = plt.get_cmap("tab20c")
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

                linestyles = {0.0: "dashed", 1.0: "solid"}
                linestyle = linestyles[tau]

                marker = "."

                colors = {"none": tab20c.colors[0], "classes": tab20c.colors[4], "groups": tab20c.colors[8]}
                markerfacecolor = color = colors[sub]
                label = config
                curves, labels = baseline_curves_labels
            else:
                raise ValueError(f"Unknown config {config}")

            curve, = ax.plot(confounder_strength, sweep[:-1], linestyle=linestyle, marker=marker,
                    color=color, markerfacecolor=markerfacecolor, linewidth=2, markersize=8)
            curves.append(curve)
            labels.append(label)

        ax.axvline(confounder_strength[train_domain], color="black", linestyle="dotted", linewidth=3)

        # plt.ylim((0, 1))
        if sweep_type in {"mean", "l1", "norm"}:
            plt.ylim((0, 1))
        else:
            plt.ylim((0.95, 1))

        plt.xlabel("Shift parameter")
        plt.ylabel(ylabel)
        plt.title(merged_title)
        plt.grid(True)
        plt.legend(*baseline_curves_labels, loc="upper left", bbox_to_anchor=(0, -0.15), ncol=1, frameon=False)
        fig.tight_layout()

        format_axes(ax)
        for suffix in ("png", "pdf"):
            plt.savefig(merged_root / f"{merged_name}_{sweep_type}.{suffix}", bbox_inches='tight', dpi=300)

        plt.close(fig)


if __name__ == "__main__":
    latexify(width_scale_factor=2, fig_height=2)
    merge()
