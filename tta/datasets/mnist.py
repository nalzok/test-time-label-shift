# Forked from https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py
from collections import Counter
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms as T
from PIL import Image

from . import MultipleDomainDataset


class MultipleDomainMNIST(MultipleDomainDataset):

    def __init__(self, root, generator, train_domains, apply_rotation, label_noise):
        self.colors = torch.ByteTensor([
            (1, 0, 0),
            (0, 1, 0),
        ])
        if apply_rotation:
            self.angles = [0, 15]
        else:
            self.angles = [0]
        self.Z = torch.LongTensor([(c_idx, r_idx) for c_idx in range(len(self.colors)) for r_idx in range(len(self.angles))])

        input_shape = (1, 28, 28, 3)
        C = 2
        K = len(self.Z)
        confounder_strength = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        super().__init__(input_shape, C, K, confounder_strength)

        self.train_domains = train_domains
        self.label_noise = label_noise

        if root is None:
            raise ValueError('Data directory not specified!')

        self.generator = generator

        original_dataset_tr = MNIST(root, train=True, download=False)
        original_dataset_te = MNIST(root, train=False, download=False)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))
        original_labels = (original_labels < 5).long()

        shuffle = torch.randperm(len(original_images), generator=self.generator)

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        # P(Z|Y)
        if apply_rotation:
            confounder1 = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
            confounder2 = np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.0, 0.0]])
        else:
            confounder1 = np.array([[1.0, 0.0], [0.0, 1.0]])
            confounder2 = np.array([[0.0, 1.0], [1.0, 0.0]])

        for i, strength in enumerate(self.confounder_strength):
            offset = 0 if i in train_domains else 1
            images = original_images[offset::2]
            labels = original_labels[offset::2]
            conditional = torch.from_numpy(strength * confounder1 + (1-strength) * confounder2)
            domain = self.shift(images, labels, conditional)

            counter = Counter(labels.numpy())
            y_count = torch.zeros(C)
            for label in counter:
                y_count[label] += counter[label]
            y_freq = y_count / len(labels)
            joint_M = y_freq[:, np.newaxis] * conditional

            self.domains.append((joint_M, domain))


    def shift(self, images, y_tilde, conditional):
        lookup_table = torch.cumsum(conditional, dim=1)
        to_tensor = T.ToTensor()
        N = y_tilde.size(0)

        # inject noise to Y
        if self.label_noise > 0:
            weights = torch.ones((N, self.C))
            weights[torch.arange(N), y_tilde] += 1/self.label_noise - 2
        else:
            weights = torch.zeros((N, self.C))
            weights[torch.arange(N), y_tilde] = 1
        y = torch.multinomial(weights, 1, generator=self.generator).squeeze(dim=-1)

        # generate Z condition on Y
        values = torch.rand((N, 1), generator=self.generator)
        z_idx = torch.searchsorted(lookup_table[y], values).squeeze(dim=-1)
        z = self.Z[z_idx]
        z_flattened = len(self.angles) * z[:, 0] + z[:, 1]

        # transform X based on Z
        x = torch.empty((N, *self.input_shape[1:]))
        for i, (image, (color_idx, angle_idx)) in enumerate(zip(images, z)):
            color = self.colors[color_idx]
            angle = self.angles[angle_idx]

            image = color * image.unsqueeze(-1)
            image = Image.fromarray(image.numpy())
            image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=tuple(color.tolist()))
            image = to_tensor(image)
            image = image.permute(1, 2, 0)

            x[i] = image

        return TensorDataset(x, y_tilde, y, z_flattened)
