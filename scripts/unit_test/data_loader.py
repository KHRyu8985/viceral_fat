import autorootcwd
import abc
from typing import Optional
from monai.data import DataLoader, Dataset
from monai.transforms import (
    EnsureChannelFirstd,
    LoadImaged,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandRotated,
    RandShiftIntensityd,
    CropForegroundd,
    Compose,
)
from monai.config import get_config_values, get_optional_config_values
import os
import logging
import json
import yaml
import nibabel as nib
import numpy as np
import time


def load_data_splits(yaml_path):
    # Read the YAML file
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    # Extract train, val, test splits from fold_1
    fold_1 = data["cross_validation_splits"][0]["fold_1"]
    train_split = fold_1["train"]
    val_split = fold_1["val"]
    test_split = fold_1["test"]

    # Add folder path to each entry in the splits and create dictionaries
    base_dir = os.path.dirname(yaml_path)
    train_split = [
        {
            "image": os.path.join(base_dir, entry, "CT.nii.gz"),
            "label": os.path.join(base_dir, entry, "vf.nii.gz"),
        }
        for entry in train_split
    ]
    val_split = [
        {
            "image": os.path.join(base_dir, entry, "CT.nii.gz"),
            "label": os.path.join(base_dir, entry, "vf.nii.gz"),
        }
        for entry in val_split
    ]
    test_split = [
        {
            "image": os.path.join(base_dir, entry, "CT.nii.gz"),
            "label": os.path.join(base_dir, entry, "vf.nii.gz"),
        }
        for entry in test_split
    ]

    return train_split, val_split, test_split


def create_datasets(data_splits):
    # Define transformations
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-200,
                a_max=100,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(128, 128, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandRotated(keys=["image", "label"],
                        mode=["bilinear", "nearest"],
                        prob=0.2,
                        range_x=0.1,
                        range_y=0.1,
                        range_z=0.1),
            RandShiftIntensityd(keys="image", offsets=0.05, prob=0.5),
        ]
    )
    
    val_test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-200,
                a_max=100,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )


    train, val, test = data_splits
    # Create datasets
    train_dataset = Dataset(data=train, transform=train_transforms)
    val_dataset = Dataset(data=val, transform=val_test_transforms)
    test_dataset = Dataset(data=test, transform=val_test_transforms)

    return train_dataset, val_dataset, test_dataset


def main():
    results_dir = "results/test_data_loader"
    yaml_path = "data/KU-PET-CT/data_splits.yaml"  # Updated path to YAML file
    os.makedirs(results_dir, exist_ok=True)

    # Set up basic logging configuration
    logging.basicConfig(
        filename=os.path.join(results_dir, "logs.log"),
        level=logging.INFO,  # 원하는 로깅 레벨로 변경
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Add a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(console_handler)

    # Printing MONAI version values
    # Logging the configuration values
    logging.info(json.dumps(get_config_values(), indent=4))
    logging.info(json.dumps(get_optional_config_values(), indent=4))

    # Load data splits
    train_split, val_split, test_split = load_data_splits(yaml_path)

    logging.info(f"Train split: {len(train_split)} subjects")
    logging.info(f"Validation split: {len(val_split)} subjects")
    logging.info(f"Test split: {len(test_split)} subjects")

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        (train_split, val_split, test_split)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for i, test_data in enumerate(train_loader):
        start_time = time.time()  # Start timing

        if i >= 10:  # Stop after 10 examples
            break
        logging.info(f"Test data {i}: {test_data['image'].shape}, {test_data['label'].shape}")

        # Save test data as NIfTI files
        image_nifti = nib.Nifti1Image(test_data['image'][0,0].numpy(), np.eye(4))
        label_nifti = nib.Nifti1Image(test_data['label'][0,0].numpy(), np.eye(4))
        nib.save(image_nifti, os.path.join(results_dir, f'test_image_{i}.nii.gz'))
        nib.save(label_nifti, os.path.join(results_dir, f'test_label_{i}.nii.gz'))

        end_time = time.time()  # End timing
        logging.info(f"Time taken for batch {i}: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
