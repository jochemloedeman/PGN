import os.path
import pathlib
from typing import Optional

import pytorch_lightning as pl
import torch.utils.data
import torchvision
from torchvision.transforms import InterpolationMode

from pgn.datamodules.utils import Solarize
from pgn.datasets.json_dataset import JSONDataset


class SUN397DataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root,
            train_batch_size,
            test_batch_size,
            num_workers,
            scale_lower_bound,
            jitter_prob,
            greyscale_prob,
            solarize_prob,
            **kwargs
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.scale_lower_bound = scale_lower_bound
        self.jitter_prob = jitter_prob
        self.greyscale_prob = greyscale_prob
        self.solarize_prob = solarize_prob

        self.nr_of_classes = 397
        self.prompt_prefix = "This is a photo of a"

    def setup(self, stage: Optional[str] = None) -> None:
        root_dir = pathlib.Path(self.data_root) / 'sun397'

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(
                224,
                scale=(self.scale_lower_bound, 1.),
                interpolation=InterpolationMode.BICUBIC
            ),

            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)
            ], p=self.jitter_prob),
            torchvision.transforms.RandomGrayscale(p=self.greyscale_prob),
            torchvision.transforms.RandomApply([Solarize()],
                                               p=self.solarize_prob),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])

        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=224),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])

        self.id_to_class = self._load_id_to_class(root_dir)
        self._create_index_to_classes()
        self._create_prompts()

        if stage == 'fit':
            self.train_set = JSONDataset(
                json_path=os.path.join(root_dir,
                                       'split_zhou_SUN397.json'),
                data_root=os.path.join(root_dir, 'images'),
                split='train',
                transforms=train_transform
            )
            self.val_set = JSONDataset(
                json_path=os.path.join(root_dir,
                                       'split_zhou_SUN397.json'),
                data_root=os.path.join(root_dir, 'images'),
                split='test',
                transforms=test_transform
            )

        if stage == 'test':
            self.test_set = JSONDataset(
                json_path=os.path.join(root_dir,
                                       'split_zhou_SUN397.json'),
                data_root=os.path.join(root_dir, 'images'),
                split='test',
                transforms=test_transform
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False)

    def _create_index_to_classes(self):
        index_to_classes = {
            key: value.replace("_", " ") for key, value in
            self.id_to_class.items()
        }
        self.index_to_classes = dict(sorted(index_to_classes.items()))

    def _load_id_to_class(self, root_dir):
        dummy_set = JSONDataset(
                json_path=os.path.join(root_dir,
                                       'split_zhou_SUN397.json'),
                data_root=os.path.join(root_dir, 'images'),
                split='train',
                transforms=None
            )
        class_to_idx = dummy_set.class_to_idx
        return {
            idx: class_label for class_label, idx in class_to_idx.items()
        }

    def _create_prompts(self):
        prompts = [
            self.prompt_prefix + " " + text_label.lower()
            for text_label in self.index_to_classes.values()
        ]
        self.prompts = prompts
