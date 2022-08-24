from typing import Any, Callable, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
from flax.struct import field
import optax

from .models import AdaptiveResNet18


class TrainState(train_state.TrainState):
    logits_fn: Callable = field(pytree_node=False)
    calibrated_fn: Callable = field(pytree_node=False)
    batch_stats: flax.core.FrozenDict[str, jnp.ndarray]
    prior: flax.core.FrozenDict[str, jnp.ndarray]


def create_train_state(key: Any, C: int, K: int, T: float, learning_rate: float, specimen: jnp.ndarray) -> TrainState:
    net = AdaptiveResNet18(C=C, K=K, T=T)

    variables = net.init(key, specimen, True)
    variables, params = variables.pop('params')
    variables, batch_stats = variables.pop('batch_stats')
    variables, prior = variables.pop('prior')
    assert not variables

    tx = optax.adam(learning_rate)
    state = TrainState.create(
            apply_fn=net.apply,
            params=params,
            tx=tx,
            logits_fn=partial(net.apply, method=net.logits),
            calibrated_fn=partial(net.apply, method=net.calibrated),
            batch_stats=batch_stats,
            prior=prior,
    )

    return state


@partial(jax.pmap, axis_name='batch', donate_argnums=(0,))
def train_step(state: TrainState, X: jnp.ndarray, M: jnp.ndarray) -> Tuple[TrainState, jnp.ndarray]:
    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(params):
        variables = {
            'params': params,
            'batch_stats': state.batch_stats,
            'prior': state.prior
        }
        logits, new_model_state = state.logits_fn(
            variables, X, True, mutable=['batch_stats']
        )

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, M)

        return loss.sum(), new_model_state

    (loss, new_model_state), grads = loss_fn(state.params)
    grads = jax.lax.psum(grads, axis_name='batch')

    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state['batch_stats'],
    )

    return state, jax.lax.psum(loss, axis_name='batch')


@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3,), donate_argnums=(0,))
def calibration_step(state: TrainState, X: jnp.ndarray, M: jnp.ndarray, multiplier: float) -> Tuple[TrainState, jnp.ndarray]:
    @partial(jax.value_and_grad)
    def loss_fn(params):
        variables = {
            'params': params,
            'batch_stats': state.batch_stats,
            'prior': state.prior
        }
        source_likelihood = state.calibrated_fn(variables, X, False)
        logits = jnp.log(source_likelihood)

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, M)

        return loss.sum()

    loss, grads = loss_fn(state.params)
    grads = jax.lax.psum(grads, axis_name='batch')
    grads = jax.tree_util.tree_map(lambda x: multiplier * x, grads)

    state = state.apply_gradients(grads=grads)

    return state, jax.lax.psum(loss, axis_name='batch')


cross_replica_mean: Callable = jax.pmap(lambda x: jax.lax.pmean(x, 'batch'), 'batch')


@partial(jax.pmap, axis_name='batch')
def induce_step(state: TrainState, X: jnp.ndarray) -> jnp.ndarray:
    variables = {
        'params': state.params,
        'batch_stats': state.batch_stats,
        'prior': state.prior
    }
    source_likelihood = state.calibrated_fn(variables, X, False)
    source_likelihood = jax.lax.psum(jnp.sum(source_likelihood, axis=0), axis_name='batch')

    return source_likelihood


@partial(jax.pmap, axis_name='batch', donate_argnums=(0,))
def adapt_step(state: TrainState, X: jnp.ndarray) -> TrainState:
    variables = {
        'params': state.params,
        'batch_stats': state.batch_stats,
        'prior': state.prior
    }
    source_likelihood = state.calibrated_fn(variables, X, False)
    source_prior = state.prior['source']

    init_val = (-1.0, 0.0, source_prior)

    def cond_fun(val):
        prev_objective, objective,  _ = val
        return objective - prev_objective > 0

    def body_fun(val):
        _, prev_objective, target_prior = val

        # E step
        target_likelihood = target_prior * source_likelihood / source_prior
        normalizer = jnp.sum(target_likelihood, axis=-1, keepdims=True)
        target_likelihood = target_likelihood / normalizer

        # M step
        target_prior = jax.lax.pmean(jnp.mean(target_likelihood, axis=0), axis_name='batch')

        log_w = jnp.log(target_prior) - jnp.log(source_prior)
        objective_i = jax.nn.logsumexp(log_w, axis=-1, b=source_likelihood)
        objective = jax.lax.psum(jnp.sum(objective_i), axis_name='batch')

        return prev_objective, objective, target_prior

    _, _, target_prior = jax.lax.while_loop(cond_fun, body_fun, init_val)

    prior = state.prior.unfreeze()
    prior['target'] = target_prior
    state = state.replace(prior=flax.core.frozen_dict.freeze(prior))

    return state

 
@partial(jax.pmap, axis_name='batch')
def test_step(state: TrainState, image: jnp.ndarray, label: jnp.ndarray) -> jnp.ndarray:
    variables = {
        'params': state.params,
        'batch_stats': state.batch_stats,
        'prior': state.prior
    }
    target_likelihood = state.apply_fn(variables, image, False)
    prediction = jnp.argmax(target_likelihood, axis=-1)
    hit = jnp.sum(prediction == label)

    return jax.lax.psum(hit, axis_name='batch')
