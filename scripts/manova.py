import sys

import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA


def fit():
    data_matrix = np.load("data/CheXpert/data_matrix.npz", allow_pickle=True)
    Y = data_matrix["features"]
    X = data_matrix["attributes"]
    X = pd.DataFrame(X, columns=data_matrix["columns"])
    Y_names = [f"Y{i}" for i in range(Y.shape[1])]
    Y = pd.DataFrame(Y, columns=Y_names)

    X = X.drop(columns=["split"])
    cutoff = np.median(X["AGE_AT_CXR"])
    X["AGE_AT_CXR"] = (X["AGE_AT_CXR"] > cutoff).astype(int)
    mask = (X["GENDER"] == 0) | (X["GENDER"] == 1)
    mask &= X["PRIMARY_RACE"] >= 0
    mask &= X["ETHNICITY"] >= 0
    X = X.loc[mask]
    Y = Y.loc[mask]

    model = MANOVA(Y, X)
    results = model.mv_test()
    print(results)


if __name__ == "__main__":
    sys.setrecursionlimit(15000)
    fit()
