import json
import os

import numpy as np
from tqdm import tqdm


def create_clevr_count_disk_filelist(input_path: str, output_path: str):
    """
    Transform the CLEVR_v1.0 dataset in folder 'input_path' to a classifcation dataset following the
    disk_folder format at 'output_path' where the goal is to count the number of objects in the scene
    """
    train_unique_targets = set()
    for split in ("train", "val"):
        print(f"Processing the {split} split...")

        # Read the scene description, holding all object information
        input_image_path = os.path.join(input_path, "images", split)
        scenes_path = os.path.join(input_path, "scenes",
                                   f"CLEVR_{split}_scenes.json")
        with open(scenes_path) as f:
            scenes = json.load(f)["scenes"]
        image_names = [scene["image_filename"] for scene in scenes]
        targets = [len(scene["objects"]) for scene in scenes]

        # Make sure that the categories in the train and validation sets are the same
        # and assigning an identifier to each of the unique target
        if split == "train":
            train_unique_targets = set(targets)
            print("Number of classes:", len(train_unique_targets))
        else:
            valid_indices = {
                i for i in range(len(image_names)) if
                targets[i] in train_unique_targets
            }
            image_names = [
                image_name
                for i, image_name in enumerate(image_names)
                if i in valid_indices
            ]
            targets = [target for i, target in enumerate(targets) if
                       i in valid_indices]

        # List the images and labels of the partition
        image_paths = []
        image_labels = []
        for image_name, target in tqdm(zip(image_names, targets),
                                       total=len(targets)):
            image_paths.append(os.path.join(input_image_path, image_name))
            image_labels.append(f"count_{target}")

        # Save the these lists in the disk_filelist format
        os.makedirs(output_path, exist_ok=True)
        img_info_out_path = os.path.join(output_path, f"{split}_images.npy")
        label_info_out_path = os.path.join(output_path, f"{split}_labels.npy")
        np.save(img_info_out_path, np.array(image_paths))
        np.save(label_info_out_path, np.array(image_labels))


if __name__ == '__main__':
    data_path = "clevr"
    output_path = os.path.join(data_path, "clevr_count")
    create_clevr_count_disk_filelist(
        input_path=os.path.join(data_path, "CLEVR_v1.0"),
        output_path=output_path
    )
