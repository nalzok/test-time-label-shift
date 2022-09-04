"""Implementation of ResNet."""

import functools
from typing import Tuple, Callable, Any, Optional, Union, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp


class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


class ResidualBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  strides: Tuple[int, int] = (1, 1)
  dtype: jnp.dtype = jnp.float32
  bottleneck: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    needs_projection = x.shape[-1] != self.filters * 4 or self.strides != (1, 1)
    nout = self.filters * 4 if self.bottleneck else self.filters

    batch_norm = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype)
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)

    residual = x
    if needs_projection:
      residual = conv(nout, (1, 1), self.strides, name='proj_conv')(residual)
      residual = batch_norm(name='proj_bn')(residual)

    if self.bottleneck:
      x = conv(self.filters, (1, 1), name='conv1')(x)
      x = batch_norm(name='bn1')(x)
      x = IdentityLayer(name='relu1')(nn.relu(x))

    y = conv(
        self.filters, (3, 3),
        self.strides,
        padding=[(1, 1), (1, 1)],
        name='conv2')(x)
    y = batch_norm(name='bn2')(y)
    y = IdentityLayer(name='relu2')(nn.relu(y))

    if self.bottleneck:
      y = conv(nout, (1, 1), name='conv3')(y)
    else:
      y = conv(nout, (3, 3), padding=[(1, 1), (1, 1)], name='conv3')(y)
    y = batch_norm(name='bn3', scale_init=jax.nn.initializers.zeros)(y)
    y = IdentityLayer(name='relu3')(nn.relu(residual + y))
    return y


class ResNet(nn.Module):
  """ResNet architecture.

  Attributes:
    num_outputs: Num output classes. If None, a dict of intermediate feature
      maps is returned.
    num_filters: Num filters.
    num_layers: Num layers.
    kernel_init: Kernel initialization.
    bias_init: Bias initialization.
    dtype: Data type, e.g. jnp.float32.
  """
  num_outputs: Optional[int]
  num_filters: int = 64
  num_layers: int = 50
  kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_normal()
  bias_init: Callable[..., Any] = jax.nn.initializers.zeros
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      train: bool = False,
      debug: bool = False) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Applies ResNet model to the inputs.

    Args:
      x: Inputs to the model.
      train: Whether it is training or not.
      debug: Whether the debug mode is enabled. debug=True enables model
        specific logging/storing some values using jax.host_callback.

    Returns:
       Un-normalized logits.
    """
    if self.num_layers not in BLOCK_SIZE_OPTIONS:
      raise ValueError('Please provide a valid number of layers')
    block_sizes, bottleneck = BLOCK_SIZE_OPTIONS[self.num_layers]
    x = nn.Conv(
        self.num_filters,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding=[(3, 3), (3, 3)],
        use_bias=False,
        dtype=self.dtype,
        name='stem_conv')(x)
    x = nn.BatchNorm(
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
        name='init_bn')(x)
    x = IdentityLayer(name='init_relu')(nn.relu(x))
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=[(1, 1), (1, 1)])

    residual_block = functools.partial(
        ResidualBlock, dtype=self.dtype, bottleneck=bottleneck)
    representations = {'stem': x}
    for i, block_size in enumerate(block_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        filters = self.num_filters * 2**i
        x = residual_block(filters=filters, strides=strides)(x, train)
      representations[f'stage_{i + 1}'] = x

    # Head.
    if self.num_outputs:
      x = jnp.mean(x, axis=(1, 2))
      x = IdentityLayer(name='pre_logits')(x)
      x = nn.Dense(
          self.num_outputs,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          dtype=self.dtype,
          name='output_projection')(x)
      return x
    else:
      return representations


# A dictionary mapping the number of layers in a resnet to the number of
# blocks in each stage of the model. The second argument indicates whether we
# use bottleneck layers or not.
BLOCK_SIZE_OPTIONS = {
    5: ([1], True),  # Only strided blocks. Total stride 4.
    8: ([1, 1], True),  # Only strided blocks. Total stride 8.
    11: ([1, 1, 1], True),  # Only strided blocks. Total stride 16.
    14: ([1, 1, 1, 1], True),  # Only strided blocks. Total stride 32.
    9: ([1, 1, 1, 1], False),  # Only strided blocks. Total stride 32.
    18: ([2, 2, 2, 2], False),
    26: ([2, 2, 2, 2], True),
    34: ([3, 4, 6, 3], False),
    50: ([3, 4, 6, 3], True),
    101: ([3, 4, 23, 3], True),
    152: ([3, 8, 36, 3], True),
    200: ([3, 24, 36, 3], True)
}
