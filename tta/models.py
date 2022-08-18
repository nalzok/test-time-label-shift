import jax
import jax.numpy as jnp
from flax import linen as nn

from .resnet import ResNet18


class AdaptiveResNet18(nn.Module):
    C: int
    K: int
    T: float

    def setup(self):
        self.M = self.C * self.K
        self.resnet = ResNet18(num_classes=self.M)
        self.b = self.param('b', jax.nn.initializers.zeros, (self.M,))
        self.source_prior = self.param('source_prior',
                                       jax.nn.initializers.constant(1/self.M,),
                                       (self.M,))
        self.target_prior = self.param('target_prior',
                                       jax.nn.initializers.constant(1/self.M,),
                                       (self.M,))

    def logits(self, x, train: bool):
        l = self.resnet(x, train)

        return l

    def calibrated(self, x, train: bool):
        logits = self.logits(x, train)
        logits = jax.lax.stop_gradient(logits)

        # bias corrected temperature scaling
        source_likelihood = jnp.exp(logits/self.T + self.b)
        source_likelihood = source_likelihood / jnp.sum(source_likelihood, axis=-1, keepdims=True)

        return source_likelihood

    def __call__(self, x, train: bool):
        source_likelihood = self.calibrated(x, train)

        # adaptation
        w = self.target_prior / self.source_prior
        source_likelihood = w * source_likelihood
        source_likelihood = source_likelihood.reshape((-1, self.C, self.K))
        source_likelihood = jnp.sum(source_likelihood, axis=-1)

        return source_likelihood
