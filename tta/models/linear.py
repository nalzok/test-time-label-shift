"""Implementation of linear regression."""

import flax.linen as nn


class Linear(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x, train: bool):
        del train
        x = nn.Dense(features=self.num_outputs)(x)
        return x
