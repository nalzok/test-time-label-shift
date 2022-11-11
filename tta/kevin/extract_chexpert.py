from pathlib import Path
import pandas as pd
import numpy as np

def extract_labels(root, max_rows=0):
    labels = pd.read_csv(root / "labels.csv", index_col="image_id")

    # Extract subset of rows for which all labels are available
    labels = labels.loc[labels["PNEUMONIA"].isin({1, 3})]
    labels = labels.loc[labels["EFFUSION"].isin({1, 3})]
    labels = labels.loc[labels["GENDER"] != "Unknown"]

    if max_rows == 0:
        max_rows = len(labels)

    columns = ["PNEUMONIA", "EFFUSION", "GENDER"]
    for t in columns:
        code, uniques = pd.factorize(labels[t], sort=True)
        print(t, code, uniques)
        labels[t] = code
    
    m = np.median(labels["AGE_AT_CXR"])
    print('median age ', m)
    labels["AGE_QUANTIZED"] = (labels["AGE_AT_CXR"] > m)
    columns.append("AGE_QUANTIZED")

    YZ = labels[columns].to_numpy()
    YZ = YZ[:max_rows]
    return YZ, labels, columns

def extract_features(root, labels, max_rows=0):
    datastore = np.load(root / "embeddings.npz")
    if max_rows == 0:
        max_rows = len(labels)
    x = datastore[labels.index[0]]
    ndims = len(x) # 1376
    X = np.zeros((max_rows, ndims))
    i = 0
    for fname in labels.index:
        x = datastore[fname]
        X[i,:] = x
        i += 1
        if i >= max_rows: break
    return X

def save_data(root =  "/home/kpmurphy/data/CheXpert", max_rows=20):
    YZ, labels, columns = extract_labels(root)
    X = extract_features(root, labels, max_rows)
    np.savez(root / 'data_matrix.npz', X=X, YZ=YZ, columns=columns)

def load_data(root =  "/home/kpmurphy/data/CheXpert",):
    foo = np.load(root / 'data_matrix.npz', allow_pickle=True)
    print(foo.files)
    #print(foo['X'].shape)
    return foo['X'], foo['YZ'], foo['columns']

    



