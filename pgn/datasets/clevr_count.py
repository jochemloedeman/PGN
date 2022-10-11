import os.path
from typing import Optional

import PIL.Image
import numpy as np
import torchvision
from torch.utils.data import Dataset


class CLEVRCount(Dataset):
    train_images = "train_images.npy"
    train_labels = "train_labels.npy"
    val_images = "val_images.npy"
    val_labels = "val_labels.npy"

    count_map = {
        "count_3": 0,
        "count_4": 1,
        "count_5": 2,
        "count_6": 3,
        "count_7": 4,
        "count_8": 5,
        "count_9": 6,
        "count_10": 7,
    }

    def __init__(
            self,
            root: str,
            train: bool,
            transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        super().__init__()
        self.root = root
        self.train = train
        self.transforms = transforms

        if self.train:
            self.image_paths = np.load(
                os.path.join(self.root, self.train_images))
            self.labels = np.load(os.path.join(self.root, self.train_labels))
        else:
            self.image_paths = np.load(os.path.join(self.root, self.val_images))
            self.labels = np.load(os.path.join(self.root, self.val_labels))
        print()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        label = self.labels[item]
        image = PIL.Image.open(image_path)
        if self.transforms is not None:
            image = self.transforms(image.convert('RGB'))
        label = self.count_map[label]
        return image, label