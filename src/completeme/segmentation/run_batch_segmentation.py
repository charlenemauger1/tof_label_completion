import os
import sys
import argparse
import datetime
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from loguru import logger
from monai.bundle import ConfigParser
from monai.networks.nets import DynUNet
from collections import defaultdict
from tqdm import tqdm

# Assuming task_params.py exists and defines deep_supr_num
from dynunet.models.task_params import deep_supr_num

# Assuming create_network.py exists and defines get_kernels_strides
from dynunet.models.create_network import get_kernels_strides
from dynunet.models.task_params import patch_size, spacing
# --- Mappings ---
TERM_MAPPING = {
    "SA": "SA",
    "LT": "2Ch_LT",
    "RT": "2Ch_RT",
    "3CH": "3Ch",
    "4CH": "4Ch",
    "RVOT": "RVOT"
}

TASK_MAPPING = {
    "SA": "11",
    "2Ch_LT": "12",
    "2Ch_RT": "16",
    "3Ch": "13",
    "4Ch": "14",
    "RVOT": "15",
}

EXPRESSION_MAPPING = {
    "SA": "nnUNet2D_CAP_SAX",
    "2Ch_LT": "nnUNet2D_CAP_2CH",
    "3Ch": "nnUNet2D_CAP_3CH",
    "4Ch": "nnUNet2D_CAP_4CH",
    "RVOT": "nnUNet2D_CAP_RVOT",
    "2Ch_RT": "nnUNet2D_CAP_RVT",
}

CLASS_MAPPING = {
    "SA": 5,
    "2Ch_LT": 4,
    "3Ch": 7,
    "4Ch": 7,
    "RVOT": 4,
    "2Ch_RT": 3,
}

from dynunet.data_preprocessing.preprocessing_params import (
    N_FOLDS, 
)

from dynunet.models.create_network import get_network
# --- Helper Functions ---
def identify_view_from_path(file_path: Path):
    """Identify view classification from file name."""
    for key in TERM_MAPPING:
        #print(key, TERM_MAPPING[key])
        if key in str(file_path):
            return TERM_MAPPING[key]
    return None

def get_network(n_class: int, task_id: str, pretrain_path: Path, checkpoint: str):
    """Initializes and loads a DynUNet model."""
    kernels, strides = get_kernels_strides(task_id)
    
    net = DynUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=n_class,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=deep_supr_num[task_id],
    )
    
    checkpoint_path = pretrain_path / checkpoint
    
    if checkpoint_path.exists():
        net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    else:
        logger.error(f"No pretrained checkpoint found at {checkpoint_path}")
        return None
    
    return net

def load_models_for_view(image_view: str, path_to_checkpoints: Path, folds: int = N_FOLDS):
    """
    Loads a collection of pre-trained models for a specific anatomical view.

    This function iterates through a specified number of cross-validation folds,
    loading a trained model for each fold. It assumes a specific directory structure
    for the checkpoints based on the task ID and expression name derived from
    the `image_view`. All loaded models are moved to the available GPU or CPU and
    are set to evaluation mode.

    Args:
        image_view (str): The string identifier for the anatomical view (e.g., '2CH').
        path_to_checkpoints (Path): The root directory containing the checkpoint
                                    subdirectories for each fold.
        folds (int, optional): The number of cross-validation folds to load. Defaults to 5.

    Returns:
        Union[Tuple[List[torch.nn.Module], int, str], None]:
            A tuple containing:
            - models (List[torch.nn.Module]): A list of the loaded DynUNet models.
            - n_class (int): The number of output classes for this view.
            - task_id (str): The nnUNet task ID for this view.
            Returns `None` if the view is invalid, no models are found, or an
            error occurs during loading.
    """
    models = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_class = CLASS_MAPPING.get(image_view)
    task_id = TASK_MAPPING.get(image_view)
    expr_name = EXPRESSION_MAPPING.get(image_view)

    if not all([n_class, task_id, expr_name]):
        logger.error(f"Invalid view '{image_view}'. Cannot load models.")
        return None

    for i in range(folds):
        val_output_dir = path_to_checkpoints / f"runs_{task_id}_fold{i}_{expr_name}"
        
        if not val_output_dir.exists():
            logger.warning(f"Validation output directory not found for {image_view} fold {i}: {val_output_dir}")
            continue
            
        checkpoints = os.listdir(val_output_dir)
        
        if len(checkpoints) != 1:
            logger.error(f'Expected one checkpoint in {val_output_dir}, but found {len(checkpoints)}. Skipping fold {i}.')
            continue
            
        model = get_network(n_class, task_id, val_output_dir, checkpoints[0])
        if model:
            models.append(model.to(device).eval())
    
    if not models:
        logger.error(f"No models were loaded for view: {image_view}. Check checkpoint paths.")
        return None
        
    logger.info(f"Successfully loaded {len(models)} models for view '{image_view}'.")
    return models, n_class, task_id

def run_inference_on_file(nifti_file: Path, segmentation_folder: Path, preloaded_networks: list, num_classes: int, task_id: int):
    """Performs inference on a single NIfTI file using pre-loaded models."""
    patient_folder = nifti_file.resolve().parent
    image_view = identify_view_from_path(patient_folder / nifti_file)
    
    if not image_view:
        logger.error(f"Could not identify view for file: {nifti_file}")
        return
        
    bundle_root = os.path.abspath(os.path.join(__file__, "../monaibundle/model", image_view))
    sys.path.append(os.path.join(bundle_root))
    sys.path.append(os.path.join(bundle_root, "scripts"))

    try: 
        cp = ConfigParser()
        cp.read_meta(f"{bundle_root}/configs/metadata.json")
        cp.read_config(f"{bundle_root}/configs/inference.json")
        cp['bundle_root'] = bundle_root
        cp['dataset_dir'] = patient_folder
        cp['datadicts'] = [{'image': nifti_file}]
        cp['output_dir'] = Path(segmentation_folder, os.path.basename(patient_folder))
        cp['num_classes'] = num_classes
        cp['evaluator']["networks"] = preloaded_networks
        cp['args']["patch_size"] = patch_size[task_id]
        cp['args']["spacing"] = spacing[task_id]
    
        #try:
        evaluator = cp.get_parsed_content("evaluator")
        evaluator.run()
        
        # Post-processing: ensure output is in correct format (e.g., uint8)
        print(segmentation_folder)
        output_path_seg = segmentation_folder / nifti_file.name
        
        if output_path_seg.exists():
            image = nib.load(output_path_seg)
            new_nifti = image.get_fdata()
            img_nii = nib.Nifti1Image(new_nifti.astype(np.uint8), image.affine)
            nib.save(img_nii, output_path_seg)
            logger.info(f"Successfully processed and saved segmentation for {nifti_file.name}")
        else:
            logger.error(f"Inference did not produce an output file at {output_path_seg}")

    except Exception as e:
        logger.exception(f"Error processing {nifti_file}: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto-segmentation pipeline for SCMR data')
    parser.add_argument('-b', '--base-folder', type=str, required=True,
                        help='Base directory containing Nifti files')
    parser.add_argument('-o', '--output_folder', type=str, required=True,
                        help='Output folder')
    parser.add_argument('-ckpt', '--path_to_checkpoints', type=str, default="./monaibundle/checkpoints",
                        help='Path to checkpoints')
    parser.add_argument('-log', action="store_true", help="Print log in the console")
    
    args = parser.parse_args()

    assert Path(args.base_folder).exists(), f'base-folder does not exist. Cannot find {args.base_folder}!'
    
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    nifti_segmentation_folder = output_folder / 'labels'
    nifti_segmentation_folder.mkdir(parents=True, exist_ok=True)
    
    # Configure the logging
    if not args.log:
        logger.remove()
        
    log_level = "INFO"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
    
    logger_id = logger.add(
        f'{output_folder}/log_file_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log',
        level=log_level, format=log_format,
        colorize=False, backtrace=True,
        diagnose=True)

    # --- Step 1: Discover all files and group them by view ---
    logger.info("Discovering and grouping NIfTI files by view...")
    p = Path(args.base_folder).glob('**/*')
    nifti_files = [x for x in p if x.is_file() and 'nii.gz' in Path(x).name]
    
    files_by_view = defaultdict(list)
    for file_path in nifti_files:
        view = identify_view_from_path(file_path.resolve().parent / file_path)
        if view:
            files_by_view[view].append(file_path)
        else:
            logger.warning(f"Skipping file with unrecognized view: {file_path}")

    if not files_by_view:
        logger.error("No NIfTI files with a recognized view found. Exiting.")
        sys.exit(1)

    # --- Step 2: Process each view sequentially ---
    for image_view, file_list in files_by_view.items():
        logger.info(f"--- Starting inference for view: '{image_view}' with {len(file_list)} files ---")
        
        # Load models for the current view once
        models_data = load_models_for_view(image_view, Path(args.path_to_checkpoints))

        if not models_data:
            logger.error(f"Failed to load models for view '{image_view}'. Skipping inference for this view.")
            continue
            
        preloaded_networks, num_classes, task_id = models_data  
        # Run inference on all files for the current view using the pre-loaded models

        try:
            for nifti_file in tqdm(file_list, desc=f"Inference for {image_view}"):

                run_inference_on_file(
                    nifti_file,
                    nifti_segmentation_folder,
                    preloaded_networks,
                    num_classes,
                    task_id
                )
        except:
            logger.warning("Failed to process {nifti_file}")

    logger.info("All views processed. Inference complete.")