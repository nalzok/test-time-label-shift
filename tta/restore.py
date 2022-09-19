from typing import Union, Mapping, Any
from pathlib import Path
from collections.abc import MutableMapping
import logging

import flax
from flax.training.checkpoints import restore_checkpoint, convert_pre_linen

from .train import TrainState


# JAX team is working on type annotation for pytree:
# https://github.com/google/jax/issues/1555
PyTree = Union[Mapping[str, Mapping], Any]


def restore_train_state(state: TrainState, checkpoint_path: Path) -> TrainState:
    restored_train_state = restore_checkpoint(checkpoint_path, None)
    if restored_train_state is None:
        raise ValueError(
            f"No checkpoint for the pretrained model is found in: {checkpoint_path}"
        )

    if "params" in restored_train_state:
        # restored_train_state was trained using optax
        restored_params = restored_train_state["params"]
    else:
        # restored_train_state was trained using flax.optim. Note that this does
        # not convert the naming of pre-Linen checkpoints.
        restored_params = restored_train_state["optimizer"]["target"]
        if "params" in restored_params:  # Backward compatibility.
            restored_params = restored_params["params"]
            restored_params = dict(convert_pre_linen(restored_params))

    # HACK: adapt the scenic checkpoint to our format
    del restored_params["pre_logits"]
    restored_params["output_projection"] = state.params["net"]["output_projection"]
    restored_params = {"net": restored_params, "b": state.params["b"]}
    restored_params = flax.core.freeze(restored_params)

    restored_batch_stats = flax.core.freeze(restored_train_state["model_state"])

    # Inspect and compare the parameters of the model with the init-model.
    params = inspect_params(
        expected_params=state.params,
        restored_params=restored_params,
        fail_if_extra=True,
        fail_if_missing=True,
        fail_if_shapes_mismatch=True,
    )

    state = state.replace(
        # Inspect and compare the parameters of the model with the init-model.
        params=params,
        batch_stats=restored_batch_stats,
    )

    return state


def inspect_params(
    *,
    expected_params: PyTree,
    restored_params: PyTree,
    fail_if_extra: bool = True,
    fail_if_missing: bool = True,
    fail_if_shapes_mismatch: bool = False,
) -> PyTree:
    """Inspects whether the params are consistent with the expected keys."""

    def _flatten_params(d, parent_key="", sep="/"):
        """Flattens a dictionary, keeping empty leaves."""
        items = []
        for k, v in d.items():
            path = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(_flatten_params(v, path, sep=sep).items())
            else:
                items.append((path, v))
        # Keeps the empty dict if it was set explicitly.
        if parent_key and not d:
            items.append((parent_key, {}))
        return dict(items)

    expected_flat = _flatten_params(flax.core.unfreeze(expected_params))
    restored_flat = _flatten_params(flax.core.unfreeze(restored_params))
    missing_keys = expected_flat.keys() - restored_flat.keys()
    extra_keys = restored_flat.keys() - expected_flat.keys()

    is_shape_mismatch = False
    for key in restored_flat:
        if key in expected_flat:
            restored_shape = None
            expected_shape = None
            # Handle empty nodes (without trainable params)
            if not isinstance(restored_flat[key], dict):
                restored_shape = restored_flat[key].shape
            if not isinstance(expected_flat[key], dict):
                expected_shape = expected_flat[key].shape

            if restored_shape != expected_shape:
                is_shape_mismatch = True
                logging.warning(
                    "Key: %s. Expected shape: %s. Restored shape: %s",
                    key,
                    expected_flat[key].shape,
                    restored_flat[key].shape,
                )

    # Adds back empty dict explicitly, to support layers without weights.
    # Context: FLAX ignores empty dict during serialization.
    empty_keys = set()
    for k in missing_keys:
        if isinstance(expected_flat[k], dict) and not expected_flat[k]:
            restored_params[k] = {}  # pytype: disable=unsupported-operands
            empty_keys.add(k)
    missing_keys -= empty_keys

    if empty_keys:
        logging.warning("Inspect recovered empty keys:\n%s", empty_keys)

    logging.info("Inspect missing keys:\n%s", missing_keys)
    logging.info("Inspect extra keys:\n%s", extra_keys)

    if fail_if_shapes_mismatch and is_shape_mismatch:
        raise ValueError("Shape mismatch between restored and target model")

    if (missing_keys and fail_if_missing) or (extra_keys and fail_if_extra):
        raise ValueError(
            f"Missing params from checkpoint: {missing_keys}.\n"
            f"Extra params in checkpoint: {extra_keys}.\n"
            f"Restored params from checkpoint: {restored_flat.keys()}.\n"
            f"Expected params from code: {expected_flat.keys()}."
        )
    return restored_params
