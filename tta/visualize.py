from typing import Set, Union

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_WIDTH = 6.0
DEFAULT_HEIGHT = 1.5

# Font sizes
SIZE_SMALL = 10
SIZE_MEDIUM = 12
SIZE_LARGE = 16

SPINE_COLOR = 'gray'


def latexify(
    width_scale_factor=1,
    height_scale_factor=1,
    fig_width=None,
    fig_height=None,
):
    f"""
    width_scale_factor: float, DEFAULT_WIDTH will be divided by this number, DEFAULT_WIDTH is page width: {DEFAULT_WIDTH} inches.
    height_scale_factor: float, DEFAULT_HEIGHT will be divided by this number, DEFAULT_HEIGHT is {DEFAULT_HEIGHT} inches.
    fig_width: float, width of the figure in inches (if this is specified, width_scale_factor is ignored)
    fig_height: float, height of the figure in inches (if this is specified, height_scale_factor is ignored)
    """
    if fig_width is None:
        fig_width = DEFAULT_WIDTH / width_scale_factor
    if fig_height is None:
        fig_height = DEFAULT_HEIGHT / height_scale_factor

    # use TrueType fonts so they are embedded
    # https://stackoverflow.com/questions/9054884/how-to-embed-fonts-in-pdfs-produced-by-matplotlib
    # https://jdhao.github.io/2018/01/18/mpl-plotting-notes-201801/
    plt.rcParams["pdf.fonttype"] = 42

    # https://stackoverflow.com/a/39566040
    plt.rc("font", size=SIZE_MEDIUM)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE_LARGE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE_LARGE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SIZE_LARGE)  # legend fontsize
    plt.rc("figure", titlesize=SIZE_LARGE)  # fontsize of the figure title

    # latexify: https://nipunbatra.github.io/blog/posts/2014-06-02-latexify.html
    plt.rcParams["backend"] = "ps"
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("figure", figsize=(fig_width, fig_height))


def format_axes(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


def plot(
    npz_path: Path,
    train_batch_size: int,
    confounder_strength: np.ndarray,
    train_domains_set: Set[int],
    dataset_label_noise: float,
    plot_title: str,
    plot_root: Path,
    config_name: str,
):
    print(f"Reading from {npz_path}")
    all_sweeps = np.load(npz_path, allow_pickle=True)

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

        argmax_joint = False
        batch_size = train_batch_size
        for baseline in ("Oracle", "Null"):
            adaptation = (baseline,)
            sweep = sweeps.pop((adaptation, argmax_joint, batch_size))
            label = f"[{baseline}]"
            ax.plot(confounder_strength, sweep[:-1], linestyle="--", linewidth=2, label=label)

        for (adaptation, argmax_joint, batch_size), sweep in sweeps.items():
            if adaptation[0] == "GMTL":
                _, alpha = adaptation
                label = f"[GMTL] {alpha = }, {argmax_joint = }, {batch_size = }"
            elif adaptation[0] == "EM":
                _, prior_strength, symmetric_dirichlet, fix_marginal = adaptation
                label = f"[EM] {prior_strength = }, {symmetric_dirichlet = }, {fix_marginal = }, {argmax_joint = }, {batch_size = }"
            else:
                raise ValueError(f"Unknown adaptation scheme {adaptation}")

            ax.plot(confounder_strength, sweep[:-1], linewidth=2, label=label)

        for i in train_domains_set:
            ax.axvline(confounder_strength[i], linestyle=":")

        plt.ylim((0, 1))
        plt.xlabel("Shift parameter")
        plt.ylabel(ylabel)
        plt.title(plot_title)
        plt.grid(True)
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=1, frameon=False)

        format_axes(ax)
        for suffix in ("png", "pdf"):
            plt.savefig(plot_root / f"{config_name}_{sweep_type}.{suffix}", bbox_inches="tight", dpi=300)

        plt.close(fig)


def bayes_accuracy(
    dataset_label_noise: float, confounder_strength: Union[float, np.ndarray]
) -> np.ndarray:
    upper_bound = np.maximum(
        np.maximum(1 - confounder_strength, confounder_strength),
        (1 - dataset_label_noise) * np.ones_like(confounder_strength),
    )
    return upper_bound
