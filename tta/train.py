from typing import Any, Callable, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
from flax.struct import field
import optax

from .models import AdaptiveNN


class TrainState(train_state.TrainState):
    raw_fn: Callable = field(pytree_node=False)
    calibrated_fn: Callable = field(pytree_node=False)
    batch_stats: flax.core.FrozenDict[str, jnp.ndarray]
    prior: flax.core.FrozenDict[str, jnp.ndarray]


def create_train_state(key: Any, C: int, K: int, model: str,
        learning_rate: float, specimen: jnp.ndarray, device_count: int) -> TrainState:
    net = AdaptiveNN(C=C, K=K, model=model)

    variables = net.init(key, specimen, True, method=net.adapted_prob)
    variables, params = variables.pop('params')
    if 'batch_stats' in variables:
        variables, batch_stats = variables.pop('batch_stats')
    else:
        batch_stats = {'dummy': jnp.empty(device_count)}
    variables, prior = variables.pop('prior')
    assert not variables

    tx = optax.adamw(learning_rate)
    state = TrainState.create(
            apply_fn=partial(net.apply, method=net.adapted_prob),
            params=params,
            tx=tx,
            raw_fn=partial(net.apply, method=net.raw_logit),
            calibrated_fn=partial(net.apply, method=net.calibrated_logit),
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
        logit, new_model_state = state.raw_fn(
            variables, X, True, mutable=['batch_stats']
        )

        loss = optax.softmax_cross_entropy_with_integer_labels(logit, M)

        return loss.sum(), new_model_state

    (loss, new_model_state), grads = loss_fn(state.params)
    loss = jax.lax.psum(loss, axis_name='batch')
    grads = jax.lax.psum(grads, axis_name='batch')

    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state['batch_stats'],
    )

    return state, loss


@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3,), donate_argnums=(0,))
def calibration_step(state: TrainState, X: jnp.ndarray, M: jnp.ndarray, learning_rate: float) -> Tuple[TrainState, jnp.ndarray]:
    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(params):
        variables = {
            'params': params,
            'batch_stats': state.batch_stats,
            'prior': state.prior
        }
        logit, new_model_state = state.calibrated_fn(
            variables, X, True, mutable=['batch_stats']
        )

        loss = optax.softmax_cross_entropy_with_integer_labels(logit, M)

        return loss.sum(), new_model_state

    (loss, new_model_state), grads = loss_fn(state.params)
    loss = jax.lax.psum(loss, axis_name='batch')
    grads = jax.lax.psum(grads, axis_name='batch')

    new_params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, state.params, grads)
    state = state.replace(
        params=new_params,
        batch_stats=new_model_state['batch_stats'],
    )

    return state, loss


cross_replica_mean: Callable = jax.pmap(lambda x: jax.lax.pmean(x, 'batch'), 'batch')


@partial(jax.pmap, axis_name='batch')
def induce_step(state: TrainState, X: jnp.ndarray) -> jnp.ndarray:
    variables = {
        'params': state.params,
        'batch_stats': state.batch_stats,
        'prior': state.prior
    }
    logit = state.calibrated_fn(variables, X, False)
    prob = jax.nn.softmax(logit)
    prob_sum = jax.lax.psum(jnp.sum(prob, axis=0), axis_name='batch')

    return prob_sum


@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(2, 3, 5, 6), donate_argnums=(0,))
def adapt_step(state: TrainState, X: jnp.ndarray, C: int, K: int,
        prior_strength: float, symmetric_dirichlet: bool, fix_marginal: bool) -> TrainState:
    M = C * K
    source_prior = state.prior['source']
    if symmetric_dirichlet:
        alpha = jnp.ones(M)
    else:
        alpha = source_prior * M
    alpha = prior_strength * alpha

    variables = {
        'params': state.params,
        'batch_stats': state.batch_stats,
        'prior': state.prior
    }
    logit = state.calibrated_fn(variables, X, False)
    prob = jax.nn.softmax(logit)

    init_target_prior = source_prior
    init_objective = jnp.sum((alpha - 1) * jnp.log(source_prior))
    init_val = init_target_prior, init_objective, init_objective - 1

    def cond_fun(val):
        _, objective, prev_objective = val
        return objective > prev_objective

    def body_fun(val):
        target_prior, prev_objective, _ = val

        # E step
        target_prob = target_prior * prob / source_prior
        normalizer = jnp.sum(target_prob, axis=-1, keepdims=True)
        target_prob = target_prob / normalizer

        # M step
        target_prob_count = jax.lax.psum(jnp.sum(target_prob, axis=0), axis_name='batch')
        target_prior_count = target_prob_count + (alpha - 1)    # add pseudocount
        target_prior = target_prior_count / jnp.sum(target_prior_count)

        # Objective
        log_w = jnp.log(target_prior) - jnp.log(source_prior)
        mle_objective_i = jax.nn.logsumexp(log_w, axis=-1, b=prob)
        mle_objective = jax.lax.psum(jnp.sum(mle_objective_i), axis_name='batch')
        regularizer = jnp.sum((alpha - 1) * jnp.log(target_prior))
        objective = mle_objective + regularizer

        return target_prior, objective, prev_objective

    target_prior, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_val)

    if fix_marginal:
        # Make sure the marginal distribution of Y does not change
        source_prior = source_prior.reshape((C, K))
        target_prior = target_prior.reshape((C, K))
        source_marginal = jnp.sum(source_prior, axis=-1, keepdims=True)
        target_marginal = jnp.sum(target_prior, axis=-1, keepdims=True)
        target_prior = target_prior / target_marginal * source_marginal
        target_prior = target_prior.flatten()

    prior = state.prior.unfreeze()
    prior['target'] = target_prior
    state = state.replace(prior=flax.core.frozen_dict.freeze(prior))

    return state

 
@partial(jax.pmap, axis_name='batch')
def test_step(state: TrainState, image: jnp.ndarray, label: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    variables = {
        'params': state.params,
        'batch_stats': state.batch_stats,
        'prior': state.prior
    }
    target_prob = state.apply_fn(variables, image, False)
    score = target_prob[:, 1]   # assumes binary label
    prediction = jnp.argmax(target_prob, axis=-1)
    hit = jax.lax.psum(jnp.sum(prediction == label), axis_name='batch')

    return score, hit
