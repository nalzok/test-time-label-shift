from collections import Counter
import numpy as np
import torch
from torchvision import transforms as T
from wilds.datasets.waterbirds_dataset import WaterbirdsDataset

from . import MultipleDomainDataset


class MultipleDomainWaterbirds(MultipleDomainDataset):

    domain_names = ['train', 'val', 'test']

    def __init__(self, root, generator):
        input_shape = (1, 224, 224, 3)
        C = 2
        K = 2
        confounder_strength = np.array([0, 1, 2])
        super().__init__(input_shape, C, K, confounder_strength)

        if root is None:
            raise ValueError('Data directory not specified!')

        self.generator = generator

        self.waterbirds = WaterbirdsDataset(root_dir=root)

        # make Z compliant in shape
        self.waterbirds._metadata_array = self.waterbirds._metadata_array[:, 0]

        # P(Z|Y)
        conditionals = [
            np.array([[0.95, 0.05], [0.05, 0.95]]),
            np.array([[0.95, 0.05], [0.05, 0.95]]),
            np.array([[0.5, 0.5], [0.5, 0.5]]),
        ]

        # ImageNet augmentation
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        permute = T.Lambda(lambda x: x.permute(1, 2, 0))
        random_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
            permute,
        ])
        deterministic_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
            permute,
        ])

        for env in self.confounder_strength:
            conditional = torch.from_numpy(conditionals[env])
            domain_name = self.domain_names[env]
            transform = random_transform if domain_name == 'train' else deterministic_transform
            domain = self.waterbirds.get_subset(domain_name, transform=transform)

            counter = Counter(int(label) for _, label, _ in domain)
            y_count = torch.zeros(C)
            for label in counter:
                y_count[label] += counter[label]
            y_freq = y_count / len(domain)
            joint_YZ = y_freq[:, np.newaxis] * conditional

            self.domains.append((joint_YZ, domain))
