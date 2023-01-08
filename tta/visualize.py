from typing import Set, Union

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot(
    npz_path: Path,
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
        ylabel = ylabel.replace("Average AUC", "AUC")
        fig, ax = plt.subplots(figsize=(12, 6))

        if sweep_type == "accuracy":
            if dataset_label_noise > 0:
                upper_bound = bayes_accuracy(dataset_label_noise, confounder_strength)
                ax.plot(
                    confounder_strength,
                    upper_bound,
                    color="grey",
                    linestyle="dashdot",
                    label="Upper bound",
                )

        alpha_min, alpha_max = float('inf'), -float('inf')
        prior_str_min, prior_str_max = float('inf'), -float('inf')
        batch_size_min, batch_size_max = float('inf'), -float('inf')
        for ((algo, *param), argmax_joint, batch_size), sweep in sweeps.items():
            if algo == "GMTL":
                alpha, = param
                alpha_min = min(alpha_min, alpha)
                alpha_max = max(alpha_max, alpha)
            elif algo == "EM":
                prior_str, _, _ = param
                prior_str_min = min(prior_str_min, prior_str)
                prior_str_max = max(prior_str_max, prior_str)
                batch_size_min = min(batch_size_min, batch_size)
                batch_size_max = max(batch_size_max, batch_size)

        baseline_curves_labels = [], []
        gmtl_curves_labels = [], []
        em_curves_labels = [], []
        tab20c = plt.get_cmap("tab20c")
        for ((algo, *param), argmax_joint, batch_size), sweep in sweeps.items():
            del argmax_joint
            if algo == "Null":
                linestyle = "dotted"
                marker = "."
                markerfacecolor = color = tab20c.colors[19]
                scaler = 2
                label = "[Unadapted]"
                curves, labels = baseline_curves_labels
            elif algo == "Null-unconfounded":
                linestyle = "dotted"
                marker = "."
                markerfacecolor = color = tab20c.colors[8]
                scaler = 2
                label = "[Invariant]"
                curves, labels = baseline_curves_labels
            elif algo == "Oracle":
                linestyle = "dotted"
                marker = "."
                markerfacecolor = color = tab20c.colors[16]
                scaler = 2
                label = "[Oracle]"
                curves, labels = baseline_curves_labels
            elif algo == "GMTL":
                alpha, = param

                color_min = np.array(tab20c.colors[3])
                color_max = np.array(tab20c.colors[0])
                if alpha_max == alpha_min:
                    markerfacecolor = color = color_max
                else:
                    multiplier = (alpha - alpha_min)/(alpha_max - alpha_min)
                    markerfacecolor = color = color_min + multiplier * (color_max - color_min)

                linestyle = "dashed"
                marker = "^"
                scaler = 1
                label = f"[GMTL] {alpha = }"
                curves, labels = gmtl_curves_labels
            elif algo == "EM":
                prior_str, symmetric_dirichlet, fix_marginal = param
                del symmetric_dirichlet, fix_marginal

                color_min = np.array(tab20c.colors[7])
                color_max = np.array(tab20c.colors[4])
                if batch_size_max == batch_size_min:
                    color = color_max
                else:
                    multiplier = (batch_size - batch_size_min)/(batch_size_max - batch_size_min)
                    color = color_min + multiplier * (color_max - color_min)

                markerfacecolor_min = np.array(tab20c.colors[4])
                markerfacecolor_max = np.array(tab20c.colors[7])
                if prior_str_max == prior_str_min:
                    markerfacecolor = color
                else:
                    multiplier = (prior_str - prior_str_min)/(prior_str_max - prior_str_min)
                    markerfacecolor = markerfacecolor_min + multiplier * (markerfacecolor_max - markerfacecolor_min)

                linestyle = "solid"
                marker = "s"
                scaler = 1
                label = f"[EM] N = {batch_size}"
                curves, labels = em_curves_labels
            else:
                raise ValueError(f"Unknown adaptation algorithm {algo}")

            curve, = ax.plot(confounder_strength, sweep[:-1], linestyle=linestyle, marker=marker,
                    color=color, markerfacecolor=markerfacecolor, linewidth=2*scaler, markersize=8*scaler)
            curves.append(curve)
            labels.append(label)

        for i in train_domains_set:
            ax.axvline(confounder_strength[i], color="black",
                    linestyle="dotted", linewidth=3)

        if sweep_type in {"mean", "l1", "norm"}:
            plt.ylim((0, 1))
        else:
            plt.ylim((0.5, 1))

        plt.xlabel("Shift parameter")
        plt.ylabel(ylabel)
        plt.title(plot_title)
        plt.grid(True)
        legend1 = plt.legend(*baseline_curves_labels, loc="upper left", bbox_to_anchor=(0, -0.15), ncol=1, frameon=False)
        legend2 = plt.legend(*gmtl_curves_labels, loc="upper left", bbox_to_anchor=(1/3, -0.15), ncol=1, frameon=False)
        plt.legend(*em_curves_labels, loc="upper left", bbox_to_anchor=(2/3, -0.15), ncol=1, frameon=False)
        plt.gca().add_artist(legend1)
        plt.gca().add_artist(legend2)
        fig.tight_layout()

        format_axes(ax)
        for suffix in ("png", "pdf"):
            plt.savefig(plot_root / f"{config_name}_{sweep_type}.{suffix}", bbox_inches='tight', dpi=300)

        plt.close(fig)


def bayes_accuracy(
    dataset_label_noise: float, confounder_strength: Union[float, np.ndarray]
) -> np.ndarray:
    upper_bound = np.maximum(
        np.maximum(1 - confounder_strength, confounder_strength),
        (1 - dataset_label_noise) * np.ones_like(confounder_strength),
    )
    return upper_bound


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
