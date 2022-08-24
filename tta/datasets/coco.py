# Forked from https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision.transforms import ToTensor
from PIL import Image
from pycocotools.coco import COCO

from . import MultipleDomainDataset


class ColoredCOCO(MultipleDomainDataset):
    categories = [
        'boat',
        'airplane',
        'truck',
        'dog',
        'zebra',
        'horse',
        'bird',
        'train',
        'bus',
    ]

    backgrounds = [
        (  0, 100,   0),
        (188, 143, 143),
        (255,   0,   0),
        (255, 215,   0),
        (  0, 255,   0),
        ( 65, 105, 225),
        (  0, 225, 225),
        (  0,   0, 255),
        (255,  20, 147),
    ]

    environments = [0.9, 0.8, 0.1]

    def __init__(self, root, annFile, generator):
        super().__init__((1, 64, 64, 3))
        if root is None:
            raise ValueError('Data directory not specified!')

        self.root = root
        self.coco = COCO(annFile)

        self.cat_ids = self.coco.getCatIds(catNms=self.categories)
        self.image_ids_set = set()
        for cat_id in self.cat_ids:
            self.image_ids_set.update(self.coco.getImgIds(catIds=cat_id))
        self.image_ids = list(self.image_ids_set)

        self.generator = generator

        shuffle = torch.randperm(len(self.image_ids), generator=generator)

        num_classes = len(self.backgrounds)
        confounding = np.eye(num_classes)
        independent = np.ones_like(confounding) * 1/num_classes

        for i, strength in enumerate(self.environments):
            indices = shuffle[i::len(self.environments)]
            prob = torch.from_numpy(strength * confounding + (1 - strength) * independent)
            domain = self.dataset_transform(indices, prob)
            self.domains.append(domain)

    def dataset_transform(self, indices, prob) -> TensorDataset:
        X, Y, Z = [], [], []
        p = torch.cumsum(prob, dim=1)
        to_tensor = ToTensor()

        for sample_idx in indices:
            image_id = self.image_ids[sample_idx]
            image_json, = self.coco.loadImgs(image_id)
            image = Image.open(os.path.join(self.root, image_json['file_name'])).convert('RGB')
            anns = self.coco.loadAnns(self.coco.getAnnIds(
                imgIds=image_id,
                catIds=self.cat_ids,
                areaRng=(10000, float('inf'))
            ))

            max_area = 0
            ann = None
            for candidate in anns:
                if max_area < candidate['area']:
                    max_area = candidate['area']
                    ann = candidate

            if ann is None:
                continue

            cat_idx = self.cat_ids.index(ann['category_id'])
            background_idx = torch.searchsorted(p[cat_idx], torch.rand(1, generator=self.generator))
            background_color = self.backgrounds[background_idx]

            mask = 255 * self.coco.annToMask(ann)
            mask = Image.fromarray(mask)

            background = Image.new('RGB', image.size, background_color)
            image = Image.composite(image, background, mask)
            image = image.resize((64, 64))
            # (C, H, W) -> (H, W, C)
            image = to_tensor(image).permute(1, 2, 0)

            X.append(image)
            Y.append(cat_idx)
            Z.append(background_idx)

        X = torch.stack(X)
        Y = torch.Tensor(Y).long()
        Z = torch.cat(Z)

        return TensorDataset(X, Y, Z)
