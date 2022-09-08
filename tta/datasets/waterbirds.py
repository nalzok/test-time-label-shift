import numpy as np
import torch
from torchvision import transforms as T
from wilds.datasets.waterbirds_dataset import WaterbirdsDataset

from . import MultipleDomainDataset


class MultipleDomainWaterbirds(MultipleDomainDataset):
    environments = ['train', 'val', 'test']

    def __init__(self, train_domains, root, generator):
        input_shape = (1, 224, 224, 3)
        C = 2
        K = 2
        super().__init__(input_shape, C, K)

        self.train_domains = train_domains

        if root is None:
            raise ValueError('Data directory not specified!')

        self.generator = generator

        self.waterbirds = WaterbirdsDataset(root_dir=root)

        # make Z compliant in shape
        self.waterbirds._metadata_array = self.waterbirds._metadata_array[:, 0]

        # joint distribution of Y and Z
        conditionals = {
            'train': np.array([[0.95, 0.05], [0.05, 0.95]]),
            'val': np.array([[0.95, 0.05], [0.05, 0.95]]),
            'test': np.array([[0.5, 0.5], [0.5, 0.5]]),
        }

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Lambda(lambda x: x.permute(1, 2, 0))
        ])

        for domain_name in self.environments:
            conditional = torch.from_numpy(conditionals[domain_name])
            domain = self.waterbirds.get_subset(domain_name, transform=transform)

            joint = torch.zeros_like(conditional)
            for _, label, _ in domain:
                joint[label] += conditional[label]
            joint /= len(domain)
            self.domains.append((joint, domain))
