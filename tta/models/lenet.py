"""Implementation of LeNet."""

import flax.linen as nn


class LeNet(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x, train: bool):
        del train
        x = nn.Conv(features=6, kernel_size=(5, 5), padding=((2, 2), (2, 2)))(x)
        x = nn.sigmoid(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding=((0, 0), (0, 0)))
        x = nn.Conv(features=16, kernel_size=(5, 5), padding=[(0, 0), (0, 0)])(x)
        x = nn.sigmoid(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding=((0, 0), (0, 0)))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=120)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=84)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=self.num_outputs)(x)
        return x
