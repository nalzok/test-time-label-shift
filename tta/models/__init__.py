import jax
import jax.numpy as jnp
from flax import linen as nn

from .linear import Linear
from .lenet import LeNet
from .resnet import ResNet


class AdaptiveNN(nn.Module):
    C: int
    K: int
    model: str

    def setup(self):
        self.M = self.C * self.K

        if self.model == 'Linear':
            self.net = Linear(num_outputs=self.M)
        elif self.model == 'LeNet':
            self.net = LeNet(num_outputs=self.M)
        elif self.model.startswith('ResNet'):
            self.num_layers = int(self.model[6:])
            self.net = ResNet(num_outputs=self.M, num_layers=self.num_layers)
        else:
            raise ValueError(f'Unknown network architecture {self.model}')

        self.b = self.param('b', jax.nn.initializers.zeros, (self.M,))
        self.T = self.param('T', jax.nn.initializers.ones, ())
        self.source_prior = self.variable('prior', 'source',
                                          jax.nn.initializers.constant(1/self.M,),
                                          None,
                                          (self.M,))
        self.target_prior = self.variable('prior', 'target',
                                          jax.nn.initializers.constant(1/self.M,),
                                          None,
                                          (self.M,))

    def raw_logit(self, x, train: bool):
        logit = self.net(x, train)

        return logit

    def calibrated_logit(self, x, train: bool):
        logit = self.raw_logit(x, train)
        logit = jax.lax.stop_gradient(logit)

        # bias corrected temperature scaling
        logit = logit/self.T + self.b

        return logit

    def adapted_prob(self, x, train: bool):
        logit = self.calibrated_logit(x, train)

        # adaptation
        w = self.target_prior.value / self.source_prior.value
        logit_max = jnp.max(logit, axis=-1, keepdims=True)
        unnormalized = w * jnp.exp(logit - jax.lax.stop_gradient(logit_max))
        prob = unnormalized / jnp.sum(unnormalized, axis=-1, keepdims=True)

        prob = prob.reshape((-1, self.C, self.K))

        return prob
