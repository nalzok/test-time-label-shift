from pathlib import Path
from itertools import count

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from tta.datasets.mnist import MultipleDomainMNIST
from tta.datasets.cxr.chexpert import MultipleDomainCheXpert
from tta.datasets import split
from tta.visualize import latexify, plot


def main():
    jobs = []

    for train_domain in (2, 10):
        seed = 42
        generator = torch.Generator().manual_seed(seed)

        train_domains_set = {train_domain}
        dataset_y_column = "EFFUSION"
        dataset_z_column = "GENDER"
        dataset_target_domain_count = 512
        dataset_source_domain_count = 85267
        dataset_use_embedding = True
        dataset_label_noise = 0

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

        prior_strength = 1
        plot_title = "CheXpert-embeddings (Gradient Boosting)"
        config_name = f"chexpert-embedding_{dataset_y_column}_{dataset_z_column}_domain{train_domain}_tree_prior{prior_strength}"
        jobs.append((dataset, train_domains_set, dataset_label_noise, prior_strength, plot_title, config_name))

    for train_domain in (2, 10):
        seed = 42
        generator = torch.Generator().manual_seed(seed)

        train_domains_set = {train_domain}
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

        prior_strength = 1
        plot_title = "ColoredMNIST (Gradient Boosting)"
        config_name = f"mnist_rot{dataset_apply_rotation}_noise{dataset_label_noise}_domain{train_domain}_tree_prior{prior_strength}"
        jobs.append((dataset, train_domains_set, dataset_label_noise, prior_strength, plot_title, config_name))

    for dataset, train_domains_set, dataset_label_noise, prior_strength, plot_title, config_name in jobs:
        auc_sweeps = make_auc_sweeps(dataset, train_domains_set, prior_strength)

        npz_path = Path(f"npz/{config_name}.npz")
        all_sweeps = {
            "auc": (auc_sweeps, "AUC"),
        }
        np.savez(npz_path, **all_sweeps)

        plot_root = Path("plots/")
        plot(npz_path, dataset.confounder_strength, train_domains_set, dataset_label_noise, plot_title, plot_root, config_name)


def make_auc_sweeps(dataset, train_domains_set, prior_strength):
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
    auc_oracle = []
    auc_gmtl_05 = []
    auc_gmtl_10 = []
    auc_gmtl_20 = []
    auc_em = []

    print("AUC")
    auc_sweeps = {}

    alpha = prior_strength * 4 * source
    for i, (target_oracle, test) in enumerate(test_splits):
        target_oracle = target_oracle.numpy().flatten()
        X, Y, _, Z = dataset2np(test)
        M = Y * 2 + Z
        prob = clf.predict_proba(X)

        prob_oracle = target_oracle * prob / source
        prob_oracle /= np.sum(prob_oracle, axis=-1, keepdims=True)

        prob_gmtl_05 = source**(1-0.5) * prob / source
        prob_gmtl_05 /= np.sum(prob_gmtl_05, axis=-1, keepdims=True)
        prob_gmtl_10 = source**(1-1.0) * prob / source
        prob_gmtl_10 /= np.sum(prob_gmtl_10, axis=-1, keepdims=True)
        prob_gmtl_20 = source**(1-2.0) * prob / source
        prob_gmtl_20 /= np.sum(prob_gmtl_20, axis=-1, keepdims=True)

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

        unadapted = evaluate(prob, Y)
        oracle = evaluate(prob_oracle, Y)
        gmtl_05 = evaluate(prob_gmtl_05, Y)
        gmtl_10 = evaluate(prob_gmtl_10, Y)
        gmtl_20 = evaluate(prob_gmtl_20, Y)
        em = evaluate(prob_em, Y)

        auc_unadapted.append(unadapted)
        auc_oracle.append(oracle)
        auc_gmtl_05.append(gmtl_05)
        auc_gmtl_10.append(gmtl_10)
        auc_gmtl_20.append(gmtl_20)
        auc_em.append(em)
        print("*" if i in train_domains_set else " ", f"domain #{i:<2}, {unadapted = :.4f}", f"{em = :.4f}", f"{oracle = :.4f}")

    # Dummy
    auc_unadapted.append(None)
    auc_oracle.append(None)
    auc_gmtl_05.append(None)
    auc_gmtl_10.append(None)
    auc_gmtl_20.append(None)
    auc_em.append(None)

    batch_size = len(test_splits[0][1])
    auc_sweeps[("Null",), False, batch_size] = auc_unadapted
    auc_sweeps[("Oracle",), False, batch_size] = auc_oracle
    auc_sweeps[("GMTL", 0.5), False, batch_size] = auc_gmtl_05
    auc_sweeps[("GMTL", 1.0), False, batch_size] = auc_gmtl_10
    auc_sweeps[("GMTL", 2.0), False, batch_size] = auc_gmtl_20
    auc_sweeps[("EM", prior_strength, False, False), False, batch_size] = auc_em

    return auc_sweeps


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
