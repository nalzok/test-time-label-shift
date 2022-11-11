from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from tta.datasets.mnist import MultipleDomainMNIST
from tta.datasets import split


def main():
    seed = 42
    generator = torch.Generator().manual_seed(seed)
    dataset_apply_rotation = False
    dataset_label_noise = 0.2
    train_domains_set = {9}
    train_fraction = 0.8
    train_calibration_fraction = 0.1
    calibration_domains_set = set()
    calibration_fraction = 0.0

    root = Path("data/mnist")
    dataset = MultipleDomainMNIST(
        root,
        generator,
        train_domains_set,
        dataset_apply_rotation,
        dataset_label_noise,
    )

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
    print(f"{dataset_label_noise = }")
    print("AUC")
    for i, (_, test) in enumerate(test_splits):
        X, Y, _, Z = dataset2np(test)
        M = Y * 2 + Z
        prob = clf.predict_proba(X)

        target = np.copy(source)

        while True:
            old = target

            # E step
            target_prob = target * prob / source
            normalizer = np.sum(target_prob, axis=-1, keepdims=True)
            target_prob = target_prob / normalizer

            # M step
            target_prob_count = np.sum(target_prob, axis=0)
            target = target_prob_count / np.sum(target_prob_count)

            if np.all(target == old):
                break

        unadapted = evaluate(prob, Y)
        adapted = evaluate(target_prob, Y)
        print("*" if i in train_domains_set else " ", f"domain #{i:<2}, {unadapted = :.4f}", f"{adapted = :.4f}")


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
    main()
