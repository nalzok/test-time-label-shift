# Forked from https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py
from collections import Counter
from hashlib import sha256

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms as T
from PIL import Image

from tta.datasets import MultipleDomainDataset


class MultipleDomainMNIST(MultipleDomainDataset):

    def __init__(self, root, train_domains, generator, apply_rotation: bool, feature_noise: float, label_noise: float):
        if len(train_domains) != 1:
            raise NotImplementedError(
                "Training on multiple source distributions is not supported yet."
            )
        train_domain = next(iter(train_domains))

        self.colors = torch.ByteTensor([
            (1, 0, 0),
            (0, 1, 0),
        ])
        if apply_rotation:
            self.angles = torch.ShortTensor([0, 15])
        else:
            self.angles = torch.ShortTensor([0])
        self.Z = torch.LongTensor([(c_idx, r_idx) for c_idx in range(len(self.colors)) for r_idx in range(len(self.angles))])

        input_shape = (1, 28, 28, 3)
        C = 2
        K = len(self.Z)
        confounder_strength = np.linspace(0, 1, 21)

        m = sha256()
        m.update(self.__class__.__name__.encode())
        m.update(str(sorted(train_domains)).encode())
        m.update(generator.get_state().numpy().data.hex().encode())
        m.update(str(apply_rotation).encode())
        m.update(str(feature_noise).encode())
        m.update(str(label_noise).encode())
        m.update(self.colors.numpy().data.hex().encode())
        m.update(self.angles.numpy().data.hex().encode())

        m.update(str(input_shape).encode())
        m.update(str(C).encode())
        m.update(str(K).encode())
        m.update(confounder_strength.data.hex().encode())
        m.update(str(train_domain).encode())
        hexdigest = m.hexdigest()

        super().__init__(input_shape, C, K, confounder_strength, train_domain, hexdigest)

        cache_key = f'{train_domain}_{apply_rotation}_{feature_noise}_{label_noise}_{hexdigest}'
        cache_file = root / 'cached' / f'{cache_key}.pt'
        if cache_file.is_file():
            # NOTE: The torch.Generator state won't be the same if we load from cache
            print(f'Loading cached datasets from {cache_file}')
            self.domains = torch.load(cache_file)
            return

        print('Building datasets... (this may take a while)')
        if root is None:
            raise ValueError('Data directory not specified!')

        self.generator = generator
        self.train_domains = train_domains
        self.feature_noise = feature_noise
        self.label_noise = label_noise

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

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
            anchor1 = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
            anchor2 = np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.0, 0.0]])
        else:
            anchor1 = np.array([[1.0, 0.0], [0.0, 1.0]])
            anchor2 = np.array([[0.0, 1.0], [1.0, 0.0]])

        for i, strength in enumerate(self.confounder_strength):
            offset = 0 if i in train_domains else 1
            images = original_images[offset::2]
            labels = original_labels[offset::2]
            conditional = torch.from_numpy(strength * anchor1 + (1-strength) * anchor2)
            domain = self.shift(images, labels, conditional)

            counter = Counter(labels.numpy())
            y_count = torch.zeros(C)
            for label in counter:
                y_count[label] += counter[label]
            y_freq = y_count / len(labels)
            joint_M = y_freq[:, np.newaxis] * conditional

            self.domains.append((domain, joint_M))

        # cache_file.parent.mkdir(parents=True, exist_ok=True)
        # print(f'Saving cached datasets to {cache_file}')
        # torch.save(self.domains, cache_file)


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
            image = image.rotate(angle.item(), resample=Image.BILINEAR)
            image = to_tensor(image)
            image = image.permute(1, 2, 0)

            noise = self.feature_noise * torch.randn(image.size(), generator=self.generator)
            image = torch.clamp(image + noise, 0, 1)

            x[i] = image

        return TensorDataset(x, y_tilde, y, z_flattened)
