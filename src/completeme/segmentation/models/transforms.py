# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from monai.config import KeysCollection
import numpy as np
from monai.transforms import (
    CastToTyped,
    Compose,
    RandSimulateLowResolutiond,
    Spacingd,
    ScaleIntensityd,
    LoadImaged,
    RandScaleIntensityd,
    RandRotate90d,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandZoomd,
    RandAffined,
    RandAdjustContrastd,
    ApplyPendingd,
    SpatialPadd,
    EnsureTyped,
    MapTransform,
)

from completeme.segmentation.models.task_params import patch_size, spacing

from monai.transforms.compose import MapTransform
import nibabel as nib
import matplotlib.pyplot as plt
from monai.data import DataLoader

__all__ = ["get_task_transforms", "get_post_transforms"]

def visualize_augmented_data(
    data_loader: DataLoader,
    output_dir: str,
    num_samples: int = 5,
    max_intensity: float = 255.0, # Target max intensity for PNG (e.g., 255 for uint8)
):
    os.makedirs(output_dir, exist_ok=True)

    label_colors = [(0, 0, 0),(0, 128, 0),(0, 0, 255),(255, 255, 0),(0, 255, 255),(255, 0, 255),(255, 165, 0),(128, 0, 128),(0, 255, 0), (255, 192, 203)] 

    #label_colors[0] = 0*label_colors[0]
    samples_saved = 0
    for i, batch_data in enumerate(data_loader):
        if samples_saved >= num_samples:
            break

        image_tensor = batch_data["image"] # This will be (N, C, H, W) or potentially (N, C, H, W, D')
        label_tensor = batch_data["label"] # This will be (N, C, H, W) or (N, C, H, W, D')

        # Iterate through items in the batch
        for j in range(image_tensor.shape[0]):
            if samples_saved >= num_samples:
                break

            current_image_tensor = image_tensor[j] # (C, H, W) or (C, H, W, D')
            current_label_tensor = label_tensor[j] # (C, H, W) or (C, H, W, D')

            # Convert to NumPy and remove ALL singleton dimensions to get to (H, W)
            # np.squeeze() removes all dimensions of size 1.
            # E.g., (1, 256, 256) -> (256, 256)
            # E.g., (1, 256, 256, 1) -> (256, 256)
            image_np = np.squeeze(current_image_tensor.cpu().numpy())
            label_np = np.squeeze(current_label_tensor.cpu().numpy())

            # Validate final image_np shape for 2D plotting
            if image_np.ndim != 2:
                print(f"ERROR: Sample {samples_saved+1} image shape after full squeeze is {image_np.shape}. Expected (H, W). Skipping this sample.")
                continue # Skip to next sample if shape is unexpected

            # --- Robust Image Normalization and Scaling for Saving ---
            min_val = image_np.min()
            max_val = image_np.max()
            
            if max_val - min_val > 1e-6: # Avoid division by zero for flat images
                image_normalized = (image_np - min_val) / (max_val - min_val)
            else:
                image_normalized = np.zeros_like(image_np)
                print(f"WARNING: Sample {samples_saved+1} image has no intensity variation (min=max={min_val}). Saving as black.")

            image_scaled = (image_normalized * max_intensity).astype(np.uint8)

            # --- Process Label for Saving ---
            label_rgb = np.zeros((*label_np.shape, 3), dtype=np.uint8)
            for class_val in range(label_np.max()+1):
                label_rgb[label_np == class_val] = label_colors[class_val]

            # Perform the blending: final_color = alpha * foreground + (1 - alpha) * background
            image_scaled_float = image_scaled.astype(np.float32)
            image_scaled_rgb_float = np.stack([image_scaled_float, image_scaled_float, image_scaled_float], axis=-1)
            blending_ratio = 0.4
            blended_image_float = (blending_ratio * label_rgb) + ((1.0 - blending_ratio) * image_scaled_rgb_float)

            # Clip values to 0-255 range and convert back to uint8
            blended_image_uint8 = np.clip(blended_image_float, 0, 255).astype(np.uint8)

            # --- Save the Blended Image ---
            blended_filename = os.path.join(output_dir, f"augmented_blended_{samples_saved:03d}.png")
            plt.imsave(blended_filename, blended_image_uint8) # plt.imsave handles RGB directly

            print(f"Saved blended image: {blended_filename}")

            samples_saved += 1

def get_task_transforms(mode, task_id, pos_sample_num, neg_sample_num, num_samples):

    if mode != "test":
        keys = ["image", "label"]
        spacing_mode = ["bilinear", "nearest"]
    else:
        keys = ["image", "label"]
        spacing_mode = ["bilinear", "nearest"]

    load_transforms = [
        LoadImaged(keys=keys, ensure_channel_first=True, image_only=True),
    ]

    # 2. sampling
    sample_transforms = [
        Spacingd(keys, pixdim=(*spacing[task_id], -1), mode=spacing_mode, lazy=True),
    ]

    # 3. spatial transforms
    if mode == "train":
        other_transforms = [

            # 1. Spatial Augmentations (Strictly In-Plane for (C, H, W) data)
            RandRotate90d(["image", "label"], prob=0.5, spatial_axes=[0, 1], lazy=True),
            
            RandAffined(
                 keys=keys,
                 prob=0.5,
                 rotate_range=[-np.pi/8, np.pi/8], # Example: +/- 22.5 degrees
                 mode=("bilinear", "nearest"),
                 lazy=True
            ),
            RandFlipd(["image", "label"], spatial_axis=[0], prob=0.75, lazy=True),  # Randomly flips the image along axes
            RandFlipd(["image", "label"], spatial_axis=[1], prob=0.75, lazy=True),  # Randomly flips the image along axes
            SpatialPadd(keys=["image", "label"], spatial_size=[*patch_size[task_id], -1], mode='minimum', lazy=True),
            #SpatialPadd(keys=["image", "label"], spatial_size=patch_size[task_id]),
            RandZoomd(
                keys=["image", "label"],
                min_zoom=[0.7, 1.0], #min_zoom=[0.9, 1.0],
                max_zoom=[1.3, 1.0], #max_zoom=[1.2, 1.0],
                mode=("trilinear", "nearest"),
                align_corners=(True, None),
                prob=0.3,
                lazy=True,
            ), # Randomly zooms input arrays with given probability within given zoom range
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=[*patch_size[task_id], -1],
                #spatial_size=patch_size[task_id],
                pos=pos_sample_num,
                neg=neg_sample_num,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
                lazy=True,
            ),

            # This ensures only one resampling operation.
            ApplyPendingd(keys=["image", "label"]), 

            # 2. Intensity Augmentations
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),  #Add Gaussian noise to image
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 2.0), #sigma_x=(0.5, 1.15),
                sigma_y=(0.5, 2.0), #sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 2.0), #sigma_z=(0.5, 1.15)
                prob=0.15, #prob=0.15,
            ),  # Apply Gaussian smooth/blur to the input data based on specified sigma parameter.
            RandAdjustContrastd(keys=["image"], gamma=(0.5, 2.0), prob=0.3), # CM added
            RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15), # Added/Increased this
            RandSimulateLowResolutiond(keys=["image"], prob = 0.35, zoom_range=(0.5,1), downsample_mode="nearest", upsample_mode="trilinear"), # CM added
            ScaleIntensityd(keys=["image"], channel_wise=True),
            CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
            EnsureTyped(keys=["image", "label"]),
        ]
    elif mode == "validation":
        other_transforms = [
        ScaleIntensityd(keys=["image"], channel_wise=True),
        CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
        EnsureTyped(keys=["image", "label"]),
        ]
    else:
        other_transforms = [
            ScaleIntensityd(keys=["image"], channel_wise=True),
            CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
            EnsureTyped(keys=["image", "label"]),
        ]

    all_transforms = load_transforms + sample_transforms + other_transforms
    return Compose(all_transforms)
 
class SaveImaged(MapTransform):
    """
    This save the image to the output directory.
    """

    def __init__(self, keys: KeysCollection, output_dir: str, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.output_dir = output_dir

    def __call__(self, data):

        image = data[self.keys]
        pixel_array = image.get_array()[0]
        try:
            filename = os.path.basename(image.meta["filename_or_obj"]).replace("_0000", "")
            nifti_data = nib.Nifti1Image(pixel_array, affine=image.meta["affine"])
        except:
            filename = "test.nii.gz"
            nifti_data = nib.Nifti1Image(pixel_array, affine=np.eye(4))
        nifti_data.header.set_xyzt_units('mm', 'sec')
        nifti_data.header['qform_code'] = 2
        nifti_data.header['sform_code'] = 2
        nib.save(nifti_data, os.path.join(self.output_dir, filename))
        print(f"save {filename} with shape: {pixel_array.shape}, mean values: {pixel_array.mean()}")

        return data

class SaveImagedValidation(MapTransform):
    """
    This custom transform saves a batch of image tensors to the output directory.
    It expects the input 'data[key]' to be a batched tensor, typically [N, C, D, H, W]
    where N is batch size, C is channel (e.g., num_classes for one-hot, or 1 for single-channel).
    It saves N separate files.
    """

    def __init__(
        self,
        keys: KeysCollection,
        output_dir: str,
        output_postfix: str = "seg",  # Added for customizable suffix
        allow_missing_keys: bool = False,
        image_list: list = [],

    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.keys = keys  # Ensure keys is always a list for iteration
        self.output_dir = output_dir
        self.output_postfix = output_postfix
        self.image_list = image_list
        os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists when initialized

    def __call__(self, data):
        # Create a copy of the dictionary to ensure non-destructive operations
        d = dict(data)

        for key in self.keys:
            if key not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"Missing key '{key}' in data for SaveImaged.")

            images_to_save = d[key] 

            pixel_array = np.array(images_to_save.get_array()).argmax(axis=0)
            output_filename = f"{os.urandom(4).hex()}_{self.output_postfix}.nii.gz"

            # Create NIfTI image and save
            # Cast to an appropriate integer type (e.g., uint8) for segmentation labels
            nifti_data = nib.Nifti1Image(pixel_array.astype(np.uint8), affine=np.eye(4))
            nifti_data.header.set_xyzt_units('mm', 'sec')
            nifti_data.header['qform_code'] = 2 # Best practice for NIfTI
            nifti_data.header['sform_code'] = 2 # Best practice for NIfTI
            full_output_path = os.path.join(self.output_dir, output_filename)
            nib.save(nifti_data, full_output_path)

        return d 