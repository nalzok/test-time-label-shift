from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from flax.jax_utils import replicate, unreplicate
import flax.linen as nn
import optax


def load_data(column):
    data_matrix = np.load("data/CheXpert/data_matrix.npz", allow_pickle=True)
    X = data_matrix["features"]

    attributes = data_matrix["attributes"]
    columns = data_matrix["columns"]
    (Y_index,) = np.flatnonzero(columns == column)
    Y = attributes[:, Y_index]

    mask = (Y == 1) | (Y == 3)
    X = X[mask]
    Y = Y[mask] // 2

    return X, Y


@partial(
    jax.pmap, axis_name="batch", static_broadcasted_argnums=(2,), donate_argnums=(0, 1)
)
def train_step(params, opt_state, tx, X, Y):
    @partial(jax.value_and_grad, has_aux=True)
    def loss_grad_fn(params, x, y):
        logit = model.apply(params, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logit, y)
        loss = jax.lax.pmean(jnp.mean(loss), axis_name="batch")
        pred = jnp.argmax(logit, axis=-1)
        hits = jax.lax.psum(jnp.sum(pred == y), axis_name="batch")
        return loss, hits

    (loss_val, hits), grads = loss_grad_fn(params, X, Y)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss_val, hits


if __name__ == "__main__":
    device_count = jax.local_device_count()
    X, Y = load_data("EFFUSION")
    X = X[X.shape[0] % device_count :].reshape(8, -1, *X.shape[1:])
    Y = Y[Y.shape[0] % device_count :].reshape(8, -1, *Y.shape[1:])

    model = nn.Dense(features=2)
    key = jax.random.PRNGKey(42)
    dummy = jnp.empty((1, 1376))
    params = model.init(key, dummy)

    learning_rate = 1e-3
    tx = optax.sgd(learning_rate=learning_rate)
    opt_state = tx.init(params)

    params = replicate(params)
    opt_state = replicate(opt_state)
    for i in range(1001):
        params, opt_state, loss_val, hits = train_step(params, opt_state, tx, X, Y)
        loss = unreplicate(loss_val)
        accuracy = unreplicate(hits / np.prod(X.shape[:2]))
        print("Loss step {}: ".format(i), "Loss:", loss, "Accuracy:", accuracy)
