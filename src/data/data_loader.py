import autorootcwd
from monai.data import Dataset
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
import os
import logging
import yaml
import nibabel as nib
import numpy as np

def load_data_splits(yaml_path, fold_number=1):
    # Read the YAML file
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    # Extract train, val, test splits from the specified fold
    fold_key = f"fold_{fold_number}"
    fold = data["cross_validation_splits"][0][fold_key]
    train_split = fold["train"]
    val_split = fold["val"]
    test_split = fold["test"]

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
    logging.info(f"Loaded data splits from {yaml_path} for fold {fold_number}")
    logging.info(f"Train: {len(train_split)} subjects")
    logging.info(f"Validation: {len(val_split)} subjects")
    logging.info(f"Test: {len(test_split)} subjects")

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
