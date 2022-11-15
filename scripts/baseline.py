from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from flax.jax_utils import replicate, unreplicate
import flax.linen as nn
import optax
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def load_data(data_matrix, column):
    X = data_matrix["features"]

    attributes = data_matrix["attributes"]
    columns = data_matrix["columns"]
    (index_Y,) = np.flatnonzero(columns == column)
    Y = attributes[:, index_Y]

    mask = np.ones_like(Y, dtype=bool)
    if column == "split" or column == "GENDER":
        mask = (Y == 0) | (Y == 1)
    elif column == "PRIMARY_RACE":
        mask = Y >= 0
        Y = (Y == 19).astype(int)  # uniques["PRIMARY_RACE"][19] == 'WHITE'
    elif column == "ETHNICITY":
        mask = Y >= 0
        Y = (Y == 2).astype(int)  # uniques["ETHNICITY"][2] == 'Non-Hispanic/Non-Latino'
    elif column == "AGE_AT_CXR":
        cutoff = np.median(Y)
        Y = (Y > cutoff).astype(int)
    else:
        # 0 = no mention
        # 1 = positive
        # 2 = uncertain
        # 3 = negative
        mask = (Y == 1) | (Y == 3)
        Y //= 2

    X = X[mask]
    Y = Y[mask]

    return X, Y


@partial(
    jax.pmap,
    axis_name="batch",
    static_broadcasted_argnums=(2, 3),
    donate_argnums=(0, 1),
)
def train_step(params, opt_state, model, tx, X, Y):
    @partial(jax.value_and_grad, has_aux=True)
    def loss_grad_fn(params, x, y):
        logit = model.apply(params, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logit, y)
        loss = jax.lax.pmean(jnp.mean(loss), axis_name="batch")
        prob = jax.nn.softmax(logit)
        score = prob[:, 1]
        return loss, score

    (loss, score), grads = loss_grad_fn(params, X, Y)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, score


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(1,))
def test_step(params, model, X, Y):
    logit = model.apply(params, X)
    loss = optax.softmax_cross_entropy_with_integer_labels(logit, Y)
    loss = jax.lax.pmean(jnp.mean(loss), axis_name="batch")
    prob = jax.nn.softmax(logit)
    score = prob[:, 1]

    return loss, score


def baseline(data_matrix, column):
    device_count = jax.local_device_count()
    X, Y = load_data(data_matrix, column)

    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=1e-3, random_state=42)
    print("Train", X.shape, Y.shape)
    print("Test", X_test.shape, Y_test.shape)

    X = jnp.array(X[X.shape[0] % device_count :]).reshape(
        device_count, -1, *X.shape[1:]
    )
    Y = jnp.array(Y[Y.shape[0] % device_count :]).reshape(
        device_count, -1, *Y.shape[1:]
    )
    X_test = jnp.array(X_test[X_test.shape[0] % device_count :]).reshape(
        device_count, -1, *X_test.shape[1:]
    )
    Y_test = jnp.array(Y_test[Y_test.shape[0] % device_count :]).reshape(
        device_count, -1, *Y_test.shape[1:]
    )

    model = nn.Dense(features=2)
    key = jax.random.PRNGKey(42)
    dummy = jnp.empty((1, 1376))
    params = model.init(key, dummy)

    learning_rate = 1e-3
    tx = optax.sgd(learning_rate=learning_rate)
    opt_state = tx.init(params)

    train_loss = train_accuracy = train_auc = np.nan
    params = replicate(params)
    opt_state = replicate(opt_state)
    for _ in range(1001):
        params, opt_state, train_loss, train_score = train_step(
            params, opt_state, model, tx, X, Y
        )
        train_loss = unreplicate(train_loss)
        train_accuracy = jnp.mean((train_score > 0.5) == Y)
        train_auc = roc_auc_score(Y.reshape(-1), train_score.reshape(-1))

    print(
        f"[Train] Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f} ({jnp.mean(Y):.4f}), AUC: {train_auc:.4f}"
    )

    test_loss, test_score = test_step(params, model, X_test, Y_test)
    test_loss = unreplicate(test_loss)
    test_accuracy = jnp.mean((test_score > 0.5) == Y_test)
    test_auc = roc_auc_score(Y_test.reshape(-1), test_score.reshape(-1))
    print(
        f"[Test] Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f} ({jnp.mean(Y_test):.4f}), AUC: {test_auc:.4f}"
    )


if __name__ == "__main__":
    data_matrix = np.load("data/CheXpert/data_matrix.npz", allow_pickle=True)
    columns = data_matrix["columns"]
    for column in columns:
        if column == "split":
            continue
        print(f"===> {column}")
        baseline(data_matrix, column)
