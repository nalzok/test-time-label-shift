from itertools import count
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from tta.datasets.mnist import MultipleDomainMNIST
from tta.datasets import split
from tta.visualize import latexify, format_axes


def main():
    seed = 42
    generator = torch.Generator().manual_seed(seed)

    train_domains_set = {10}
    dataset_apply_rotation = False
    dataset_label_noise = 0.1

    root = Path("data/mnist")
    dataset = MultipleDomainMNIST(
        root,
        train_domains_set,
        generator,
        dataset_apply_rotation,
        dataset_label_noise,
    )

    train_fraction = 0.9
    train_calibration_fraction = 0.1
    calibration_domains_set = set()
    calibration_fraction = 0.0

    train, calibration, test_splits = split(
        dataset,
        train_domains_set,
        train_fraction,
        train_calibration_fraction,
        calibration_domains_set,
        calibration_fraction,
    )

    # Training
    X, Y, _, Z = dataset2np(train)
    M = Y * 2 + Z
    clf = HistGradientBoostingClassifier(random_state=0)
    clf = clf.fit(X, M)

    induced_prob = clf.predict_proba(X)
    source = np.mean(induced_prob, axis=0)

    # Calibration
    X, Y, _, Z = dataset2np(calibration)
    M = Y * 2 + Z
    # TODO: do calibration with gradient descent or whatever

    # Testing
    auc_unadapted = []
    auc_adapted = []
    auc_oracle = []

    prior_strength = 1
    alpha = prior_strength * 4 * source

    print("AUC")
    for i, (target_oracle, test) in enumerate(test_splits):
        target_oracle = target_oracle.numpy().flatten()
        X, Y, _, Z = dataset2np(test)
        M = Y * 2 + Z
        prob = clf.predict_proba(X)

        target = np.copy(source)

        for j in count():
            old = target

            # E step
            prob_em = target * prob / source
            normalizer = np.sum(prob_em, axis=-1, keepdims=True)
            prob_em = prob_em / normalizer

            # M step
            prob_em_count = np.sum(prob_em, axis=0) + (alpha - 1)
            target = prob_em_count / np.sum(prob_em_count)

            if np.allclose(target, old) or j > 10000:
                break

        prob_oracle = target_oracle * prob / source
        normalizer = np.sum(prob_oracle, axis=-1, keepdims=True)
        prob_oracle = prob_oracle / normalizer

        unadapted = evaluate(prob, Y)
        adapted = evaluate(prob_em, Y)
        oracle = evaluate(prob_oracle, Y)

        auc_unadapted.append(unadapted)
        auc_adapted.append(adapted)
        auc_oracle.append(oracle)
        print("*" if i in train_domains_set else " ", f"domain #{i:<2}, {unadapted = :.4f}", f"{adapted = :.4f}", f"{oracle = :.4f}")

    fig, ax = plt.subplots(figsize=(12, 6))
    tab20c = plt.get_cmap("tab20c")

    ax.plot(dataset.confounder_strength, auc_unadapted, label="Unadapted",
            linestyle="dotted", marker=".", color=np.array(tab20c.colors[19]),
            linewidth=4, markersize=16)
    ax.plot(dataset.confounder_strength, auc_oracle, label="Oracle",
            linestyle="dotted", marker=".", color=np.array(tab20c.colors[16]),
            linewidth=4, markersize=16)
    ax.plot(dataset.confounder_strength, auc_adapted, label="EM",
            linestyle="solid", marker="s", color=np.array(tab20c.colors[4]),
            linewidth=2, markersize=8)

    for i in train_domains_set:
        ax.axvline(dataset.confounder_strength[i], color="black",
                linestyle="dotted", linewidth=3)

    plt.ylim((0.7, 1))
    plt.xlabel("Shift parameter")
    plt.ylabel("AUC")
    plt.title("XGBoost on ColoredMNIST")
    plt.grid(True)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    fig.tight_layout()

    format_axes(ax)
    train_domain = next(iter(train_domains_set))
    for suffix in ("png", "pdf"):
        plt.savefig(f"plots/mnist_rot{dataset_apply_rotation}_noise{dataset_label_noise}_domain{train_domain}_tree_prior{prior_strength}_auc.{suffix}", dpi=300)

    plt.close(fig)


def dataset2np(dataset):
    X, Y, Y_tilde, Z = [], [], [], []
    for x, y_tilde, y, z_flattened in dataset:
        X.append(x)
        Y.append(y)
        Y_tilde.append(y_tilde)
        Z.append(z_flattened)

    X = np.stack(X)
    X = X.reshape(-1, np.prod(X.shape[1:]))
    Y = np.stack(Y)
    Y_tilde = np.stack(Y_tilde)
    Z = np.stack(Z)

    return X, Y, Y_tilde, Z


def evaluate(prob_M, Y):
    prob_M = prob_M.reshape((-1, 2, 2))
    prob_Y = np.sum(prob_M, axis=-1)
    score_Y = prob_Y[:, 1]  # assumes binary label
    auc = roc_auc_score(Y, score_Y)
    return auc


if __name__ == "__main__":
    latexify(width_scale_factor=2, fig_height=2)
    main()
