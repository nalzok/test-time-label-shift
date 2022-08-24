# Forked from https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms as T
from torchvision.transforms import functional as F

from . import MultipleDomainDataset


class MultipleDomainMNIST(MultipleDomainDataset):
    def __init__(self, root, generator, environments, dataset_transform, input_shape):
        super().__init__(input_shape)

        if root is None:
            raise ValueError('Data directory not specified!')

        self.generator = generator

        original_dataset_tr = MNIST(root, train=True, download=False)
        original_dataset_te = MNIST(root, train=False, download=False)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images), generator=generator)

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.domains.append(dataset_transform(images, labels, environments[i]))


class ColoredMNIST(MultipleDomainMNIST):
    def __init__(self, root, generator):
        super().__init__(root, generator, [0.1, 0.2, 0.9], self.color_dataset, (1, 28, 28, 2))

    def color_dataset(self, images, labels, environment):
        # Assign a binary label based on the digit
        labels = (labels < 5).float()

        # Flip label with probability 0.25
        labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))

        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
        # (N, C, H, W) -> (N, H, W, C)
        images = images.permute(0, 2, 3, 1)

        x = images.float().div_(255.0)
        y = labels.long()
        z = colors.long()

        return TensorDataset(x, y, z)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size, generator=self.generator) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleDomainMNIST):
    def __init__(self, root, generator):
        super().__init__(root, generator, [0, 15, 30, 45, 60, 75], self.rotate_dataset, (1, 28, 28, 1))

    def rotate_dataset(self, images, labels, angle):
        rotation = T.Compose([
            T.ToPILImage(),
            T.Lambda(lambda x: F.rotate(x, angle, fill=[0,], interpolation=T.InterpolationMode.BILINEAR)),
            T.ToTensor()])

        x = torch.zeros(len(images), 28, 28, 1)
        for i in range(len(images)):
            # (N, H, W) -> (N, H, W, C)
            x[i] = rotation(images[i]).reshape(-1, 28, 28, 1)

        y = labels
        z = angle // 15 * torch.ones_like(y)

        return TensorDataset(x, y, z)
