import json
import os.path
from typing import Optional

import PIL.Image
import torchvision.transforms
from torch.utils.data import Dataset


class JSONDataset(Dataset):
    def __init__(
            self,
            json_path: str,
            data_root: str,
            split: str,
            transforms: Optional[torchvision.transforms.Compose] = None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms

        with open(json_path) as json_file:
            full_samples = json.load(json_file)
            samples = full_samples[split]
            if split == 'train':
                samples.extend(
                    full_samples['val']
                )
            self.samples = samples


        self.class_to_idx = self._create_class_to_idx(full_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rel_path, num_label, text_label = self.samples[index]
        image = PIL.Image.open(os.path.join(self.data_root, rel_path))
        if self.transforms is not None:
            image = self.transforms(image.convert('RGB'))
        return image, num_label

    @staticmethod
    def _create_class_to_idx(full_samples):
        class_to_idx = {}
        for split in full_samples:
            samples = full_samples[split]
            for sample in samples:
                path, num_label, text_label = sample
                if text_label in class_to_idx:
                    assert class_to_idx[text_label] == num_label
                class_to_idx[text_label] = num_label

        return class_to_idx