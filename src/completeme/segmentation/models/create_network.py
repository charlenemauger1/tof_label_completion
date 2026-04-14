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
import torch
from monai.networks.nets import DynUNet
from completeme.segmentation.models.task_params import deep_supr_num, patch_size, spacing
from typing import Dict

def get_kernels_strides(task_id : str) -> tuple[list[list[int]], list[list[int]]]:
    """
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.

    """
    sizes, spacings = patch_size[task_id], spacing[task_id]
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides

def get_network(properties: Dict, task_id: str, pretrain_path: str, checkpoint: str = None) -> DynUNet:
    """
    Initializes and configures a DynUNet model for a specific task.

    This function creates a MONAI DynUNet model, setting its architecture
    (kernel sizes, strides, etc.) based on the provided task properties.
    It supports optional loading of pre-trained weights from a specified
    checkpoint file.

    Args:
        properties (Dict): A dictionary containing dataset properties, such as
                          "labels" (to determine output classes) and "modality"
                          (to determine input channels).
        task_id (str): The identifier for the task, used to determine network
                       architecture parameters like kernel and stride sizes.
        pretrain_path (str): The base directory where pre-trained checkpoints
                             are stored.
        checkpoint (str, optional): The filename of the checkpoint to load.
                                   If None, no pre-trained weights are loaded.
                                   Defaults to None.

    Returns:
        monai.networks.nets.DynUNet: The initialized DynUNet model,
                                     potentially with pre-trained weights.
    """
    n_class = len(properties["labels"])
    in_channels = len(properties["modality"])
    kernels, strides = get_kernels_strides(task_id)

    net = DynUNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=n_class,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=deep_supr_num[task_id],
    )

    if checkpoint is not None:
        pretrain_path = os.path.join(pretrain_path, checkpoint)

        print("pretrain_path", pretrain_path)
        if os.path.exists(pretrain_path):
            net.load_state_dict(torch.load(pretrain_path, weights_only=True))
            print("pretrained checkpoint: {} loaded".format(pretrain_path))
        else:
            print("no pretrained checkpoint")

    return net
