from typing import Any, Callable, Tuple
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
from flax.struct import field
import optax

from .models import AdaptiveResNet


class TrainState(train_state.TrainState):
    logits_fn: Callable = field(pytree_node=False)
    calibrated_fn: Callable = field(pytree_node=False)
    batch_stats: flax.core.FrozenDict[str, jnp.ndarray]
    prior: flax.core.FrozenDict[str, jnp.ndarray]


def create_train_state(key: Any, C: int, K: int, T: float, num_layers: int, learning_rate: float, specimen: jnp.ndarray) -> TrainState:
    net = AdaptiveResNet(C=C, K=K, T=T, num_layers=num_layers)

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


def restore_train_state(state: TrainState, checkpoint_path: Path) -> TrainState:
    # pretrained = restore_checkpoint(checkpoint_path, None)
    # save_checkpoint('/tmp', state, 0)
    # print(jax.tree_util.tree_structure(state))
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


@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(2, 3, 4, 6), donate_argnums=(0,))
def adapt_step(state: TrainState, X: jnp.ndarray, C: int, K: int,
        symmetric_dirichlet: bool, prior_strength: float, fix_marginal: bool) -> TrainState:
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
    source_likelihood = state.calibrated_fn(variables, X, False)

    init_target_prior = source_prior
    init_objective = jnp.sum((alpha - 1) * jnp.log(source_prior))
    init_val = init_target_prior, init_objective, init_objective - 1

    def cond_fun(val):
        _, objective, prev_objective = val
        return objective > prev_objective

    def body_fun(val):
        target_prior, prev_objective, _ = val

        # E step
        target_likelihood = target_prior * source_likelihood / source_prior
        normalizer = jnp.sum(target_likelihood, axis=-1, keepdims=True)
        target_likelihood = target_likelihood / normalizer

        # M step
        target_likelihood_count = jax.lax.psum(jnp.sum(target_likelihood, axis=0), axis_name='batch')
        target_prior_raw = target_likelihood_count + (alpha - 1)    # add pseudocount
        target_prior = target_prior_raw / jnp.sum(target_prior_raw)

        # Objective
        log_w = jnp.log(target_prior) - jnp.log(source_prior)
        mle_objective_i = jax.nn.logsumexp(log_w, axis=-1, b=source_likelihood)
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
    target_likelihood = state.apply_fn(variables, image, False)
    log_likelihood_ratio = jnp.log(target_likelihood[:, 1]) - jnp.log(target_likelihood[:, 0])
    log_likelihood_ratio = jnp.where(
            jnp.isnan(log_likelihood_ratio),
            jnp.zeros_like(log_likelihood_ratio),
            log_likelihood_ratio)
    prediction = jnp.argmax(target_likelihood, axis=-1)
    hit = jax.lax.psum(jnp.sum(prediction == label), axis_name='batch')

    return log_likelihood_ratio, hit
