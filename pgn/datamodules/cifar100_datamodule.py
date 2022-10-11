import os.path
import pathlib
from copy import copy
from typing import Optional

import pytorch_lightning as pl
import torch.utils.data
import torchvision
from torchvision.datasets import CIFAR100
from torchvision.transforms import InterpolationMode

from pgn.datamodules.utils import Solarize


class CIFAR100DataModule(pl.LightningDataModule):
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

        self.nr_of_classes = 100
        self.prompt_prefix = "A photo of a"

    def setup(self, stage: Optional[str] = None) -> None:
        root_dir = pathlib.Path(self.data_root) / 'cifar100'

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
        self._calculate_index_to_classes()
        self._create_prompts()

        if stage == 'fit':
            self.train_set = CIFAR100(root=root_dir.as_posix(),
                                      download=True, train=True,
                                      transform=train_transform)

            self.val_set = CIFAR100(root=root_dir.as_posix(),
                                    download=True, train=False,
                                    transform=test_transform)

        if stage == 'test':
            self.test_set = CIFAR100(root=root_dir.as_posix(),
                                     download=True, train=False,
                                     transform=test_transform)

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

    def _calculate_index_to_classes(self):
        classes = [
            class_str.replace("_", " ")
            for class_str in self.id_to_class.values()
        ]
        self.index_to_classes = {
            idx: classes[idx] for idx in range(len(classes))
        }

    def _load_id_to_class(self, root_dir):
        dummy_set = CIFAR100(root=root_dir.as_posix(),
                             download=True, train=True)
        class_to_idx = dummy_set.class_to_idx
        return {
            idx: class_label for class_label, idx in class_to_idx.items()
        }

    def _create_prompts(self):
        self.prompts = [
            self.prompt_prefix + " " + text_label.lower()
            for text_label in self.index_to_classes.values()
        ]


if __name__ == '__main__':
    datamodule = CIFAR100DataModule(
        data_root="/home/jochem/Documents/ai/scriptie/data",
        train_batch_size=8,
        test_batch_size=8,
        num_workers=0,
        scenario='fewshot'
    )
    datamodule.setup(stage='fit')
    print()
