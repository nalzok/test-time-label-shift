import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from cxr_foundation import constants, train_lib


embeddings = {
    "all": np.load("data/CheXpert/embeddings.npz"),
}

for split, npz in embeddings.items():
    dirname = os.path.join("./data", "inputs", "chexpert", split)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for i, item in enumerate(npz.items()):
        image_id, embedding = item
        fname = os.path.join(dirname, f"{i:08d}.tfrecord")
        example = tf.train.Example()
        example.features.feature[constants.IMAGE_ID_KEY].bytes_list.value[:] = [
            image_id.encode("utf-8")
        ]
        example.features.feature[constants.EMBEDDING_KEY].float_list.value[
            :
        ] = embedding
        with tf.io.TFRecordWriter(fname) as writer:
            writer.write(example.SerializeToString())


labels_df = pd.read_csv("data/CheXpert/labels.csv")
labels_df.drop(columns=["Unnamed: 0"], inplace=True)


labels_df["SEX"] = labels_df["GENDER"].apply(lambda x: int(x == "Male"))

print(labels_df.SEX.value_counts())

df = labels_df

df = df[~df["SEX"].isna()]
model = train_lib.create_model(["SEX"], hidden_layer_sizes=[])
training_df, tune_df = train_test_split(df, test_size=0.2)
training_labels = dict(zip(training_df["image_id"], training_df["SEX"].astype(int)))
filenames = glob.glob(os.path.join("./data/inputs/chexpert/*/", "*.tfrecord"))
training_data = train_lib.get_dataset(filenames, labels=training_labels)
tune_labels = dict(zip(tune_df["image_id"], tune_df["SEX"].astype(int)))
tune_data = train_lib.get_dataset(filenames, labels=tune_labels).batch(1).cache()
model.fit(
    x=training_data.batch(512).prefetch(tf.data.AUTOTUNE).cache(),
    validation_data=tune_data,
    epochs=100,
)
