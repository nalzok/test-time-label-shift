# Forked from https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py
from pathlib import Path
from hashlib import sha256

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision.transforms import ToTensor
from PIL import Image

from tta.datasets import MultipleDomainDataset


class ColoredCOCO(MultipleDomainDataset):
    def __init__(self, root: Path, annFile: Path, generator: torch.Generator):
        self.categories = [
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

        self.backgrounds = [
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

        input_shape = (1, 64, 64, 3)
        C = len(self.categories)
        K = len(self.backgrounds)
        confounder_strength = np.array([0.9, 0.8, 0.1])
        super().__init__(input_shape, C, K, confounder_strength)

        m = sha256()
        m.update(str(annFile).encode())
        cache_key = m.hexdigest()
        cache_file = root / 'cached' / f'{cache_key}.pt'
        if cache_file.is_file():
            # NOTE: The torch.Generator state won't be the same if we load from cache
            print(f'Loading cached datasets from {cache_file}')
            self.domains = torch.load(cache_file)
            return

        if root is None:
            raise ValueError('Data directory not specified!')

        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)

        self.cat_ids = self.coco.getCatIds(catNms=self.categories)
        self.image_ids_set = set()
        for cat_id in self.cat_ids:
            self.image_ids_set.update(self.coco.getImgIds(catIds=cat_id))
        self.image_ids = list(self.image_ids_set)

        self.generator = generator

        shuffle = torch.randperm(len(self.image_ids), generator=self.generator)

        independent = np.ones((C, K)) * 1/K
        confounding1 = np.eye(C, K)
        confounding1 = 0.75 * confounding1 + 0.25 * independent
        confounding2 = np.roll(confounding1, shift=1, axis=1)

        for i, strength in enumerate(self.confounder_strength):
            indices = shuffle[i::len(self.confounder_strength)]
            prob = torch.from_numpy(strength * confounding1 + (1-strength) * confounding2)
            domain = self.dataset_transform(indices, prob)
            self.domains.append((prob, domain))     # FIXME: prob should be joint

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print(f'Saving cached datasets to {cache_file}')
        torch.save(self.domains, cache_file)

    def dataset_transform(self, indices: torch.Tensor, prob: torch.Tensor) -> TensorDataset:
        X, Y, Z = [], [], []
        p = torch.cumsum(prob, dim=1)
        to_tensor = ToTensor()

        for sample_idx in indices:
            image_id = self.image_ids[sample_idx]
            image_json, = self.coco.loadImgs(image_id)
            image = Image.open(self.root / image_json['file_name']).convert('RGB')
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

        return TensorDataset(X, Y, Y, Z)
