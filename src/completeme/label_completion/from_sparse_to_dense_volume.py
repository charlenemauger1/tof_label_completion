"""
rom_sparse_to_dense_volume.py

Converts sparse 3D NIfTI segmentation volumes to dense 3D one-hot encoded
volumes, then applies the Label Completion Network (LTN) to reconstruct
complete cardiac anatomy.

Pipeline per case:
    1. Convert each NIfTI label map to a 9-channel one-hot encoding,
       optionally with Gaussian smoothing (CPU or GPU via CuPy).
    2. Run the LTN model on the one-hot volume to produce a dense prediction.
    3. Clean up the intermediate one-hot directory.

Usage:
    python from_sparse_to_dense_volume.py -i <input_folder> -o <output_folder>
    python rom_sparse_to_dense_volume.py -i <input_folder> -o <output_folder> --use_gpu --max_workers 16
"""

from __future__ import annotations

import os
import time
import shutil
import argparse
import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy import ndimage
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

from completeme import LABEL_COMPLETION_CHECKPOINT_DIR
from LTN import eval_model


# ── Constants ────────────────────────────────────────────────────────────────

EXPECTED_LABELS: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
LTN_IN_CHANNELS:  int = 9
LTN_OUT_CHANNELS: int = 12


# ── GPU filter ───────────────────────────────────────────────────────────────

def gaussian_filter_gpu(mask: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply a Gaussian filter on the GPU using CuPy.

    Args:
        mask: Input float32 array.
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        Filtered array as a NumPy float32 array.
    """
    return cp.asnumpy(cp.ndimage.gaussian_filter(cp.asarray(mask), sigma=sigma))


# ── One-hot encoding ─────────────────────────────────────────────────────────

def one_hot_labelmap(
    path: str | Path,
    smoothing_sigma: float = 0.0,
    save_path: str | Path | None = None,
    use_gpu: bool = False,
) -> str:
    """
    Convert a multi-label NIfTI segmentation to a 9-channel one-hot encoding.

    All labels in EXPECTED_LABELS are guaranteed to be present. If a label is
    absent from the image a single random foreground voxel is inserted so the
    channel is not entirely empty — this preserves downstream model expectations.

    Optionally applies per-channel Gaussian smoothing before saving.

    Args:
        path: Path to the input NIfTI label map.
        smoothing_sigma: Gaussian smoothing sigma. 0 disables smoothing.
        save_path: Directory in which to save the output one-hot NIfTI.
        use_gpu: If True and CuPy is available, run smoothing on GPU.

    Returns:
        Absolute path of the saved one-hot NIfTI file.

    Raises:
        ValueError: If save_path is not provided.
    """
    if save_path is None:
        raise ValueError("save_path must be provided.")

    path = Path(path)
    save_path = Path(save_path)

    labelmap = sitk.ReadImage(str(path), sitk.sitkInt64)
    lab_array = sitk.GetArrayFromImage(labelmap)   # (H, W, D)
    h, w, d = lab_array.shape
    present_labels = set(np.unique(lab_array))

    one_hot = np.zeros((h, w, d, len(EXPECTED_LABELS)), dtype=np.float32)

    for idx, label in enumerate(EXPECTED_LABELS):
        if label in present_labels:
            mask = (lab_array == label).astype(np.float32)
        else:
            logger.warning(f"{path.name}: label {label} missing — inserting random point.")
            mask = np.zeros((h, w, d), dtype=np.float32)
            mask[
                np.random.randint(h),
                np.random.randint(w),
                np.random.randint(d),
            ] = 1.0

        if smoothing_sigma > 0:
            mask = (
                gaussian_filter_gpu(mask, sigma=smoothing_sigma)
                if use_gpu
                else ndimage.gaussian_filter(mask, sigma=smoothing_sigma, mode="nearest")
            )

        one_hot[..., idx] = mask

    out_img = sitk.GetImageFromArray(one_hot, isVector=True)
    out_img.CopyInformation(labelmap)

    output_file = save_path / path.name
    sitk.WriteImage(out_img, str(output_file))
    return str(output_file)


# ── Worker ───────────────────────────────────────────────────────────────────

def _one_hot_worker(
    path: str,
    output_path: str | Path,
    smoothing_sigma: float = 0.0,
    use_gpu: bool = False,
) -> str | None:
    """
    Worker wrapper for one_hot_labelmap, used by the process pool.

    Args:
        path: Path to the input NIfTI label map.
        output_path: Directory to save the one-hot output.
        smoothing_sigma: Gaussian smoothing sigma.
        use_gpu: Whether to run smoothing on GPU.

    Returns:
        Output file path on success, None on failure.
    """
    try:
        return one_hot_labelmap(path, smoothing_sigma, save_path=output_path, use_gpu=use_gpu)
    except Exception as e:
        logger.warning(f"Failed to process {path}: {e}")
        return None


# ── One-hot batch conversion ─────────────────────────────────────────────────

def to_one_hot(
    label_folder: str | Path,
    output_path: str | Path,
    is_sparse: bool = True,
    max_workers: int = 8,
    use_gpu: bool = False,
) -> None:
    """
    Convert all NIfTI label maps in a folder to one-hot encoding in parallel.

    Saves a CSV index of the processed files alongside the outputs.

    Args:
        label_folder: Directory containing input NIfTI label maps.
        output_path: Directory to save one-hot outputs and the CSV index.
        is_sparse: Used to label the output CSV (sparse vs dense).
        max_workers: Number of parallel worker processes.
        use_gpu: Whether to run Gaussian smoothing on GPU.
    """
    label_folder = Path(label_folder)
    output_path = Path(output_path)
    volume_type = "sparse" if is_sparse else "dense"

    file_paths = sorted(str(p) for p in label_folder.glob("*.nii.gz"))
    if not file_paths:
        logger.warning(f"No NIfTI files found in {label_folder}")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_one_hot_worker, f, output_path, 0, use_gpu): f
            for f in file_paths
        }
        output_files = [
            future.result()
            for future in as_completed(futures)
            if future.result() is not None
        ]

    pd.DataFrame({"img": output_files}).to_csv(
        output_path / f"3d_{volume_type}_oh.csv", index=False
    )


# ── Per-case processing ───────────────────────────────────────────────────────

def process_case(
    case: Path,
    output_path: Path,
    max_workers: int,
    use_gpu: bool,
) -> None:
    """
    Run the full label completion pipeline for a single case.

    Converts the case's NIfTI label maps to one-hot encoding, applies the LTN
    model, then removes the intermediate one-hot directory.

    Args:
        case: Path to the case directory containing NIfTI label maps.
        output_path: Root output directory; subdirectories are created within it.
        max_workers: Number of parallel workers for the one-hot conversion step.
        use_gpu: Whether to run Gaussian smoothing on GPU.
    """
    logger.info(f"Processing: {case.name}")

    one_hot_dir = output_path / case.name
    one_hot_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — one-hot encoding
    to_one_hot(case, one_hot_dir, max_workers=max_workers, use_gpu=use_gpu)

    # Step 2 — label completion network
    ltn_output_dir = output_path / f"{case.name}_LTN"
    ltn_output_dir.mkdir(parents=True, exist_ok=True)

    eval_model(
        Path(LABEL_COMPLETION_CHECKPOINT_DIR),
        one_hot_dir,
        ltn_output_dir,
        out_channel=LTN_OUT_CHANNELS,
        in_channel=LTN_IN_CHANNELS,
    )

    # Step 3 — clean up intermediate one-hot files
    shutil.rmtree(one_hot_dir)
    logger.info(f"Finished: {case.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert sparse 3D NIfTI segmentations to dense one-hot volumes via LTN.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("./volume_niftis"),
        help="Folder containing per-case sparse NIfTI volumes.",
    )
    parser.add_argument(
        "-o", "--output_path",
        type=Path,
        default=Path("./volume_niftis_oh"),
        help="Folder to save LTN output volumes.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Max parallel workers for one-hot conversion (default: 8).",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU (CuPy) for Gaussian filtering.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.use_gpu and not GPU_AVAILABLE:
        logger.warning("GPU requested but CuPy is not installed. Falling back to CPU.")
        args.use_gpu = False

    assert args.input.exists(), f"Input path not found: {args.input}"
    assert Path(LABEL_COMPLETION_CHECKPOINT_DIR).exists(), (
        f"LTN checkpoint not found: {LABEL_COMPLETION_CHECKPOINT_DIR}"
    )

    args.output_path.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(p for p in args.input.iterdir() if p.is_dir())
    logger.info(f"Found {len(case_dirs)} cases.")

    start = time.time()
    for case in case_dirs:
        try:
            process_case(case, args.output_path, args.max_workers, args.use_gpu)
        except Exception as e:
            logger.warning(f"Failed: {case.name} — {e}")

    logger.success(
        f"Done. Processed {len(case_dirs)} cases in {time.time() - start:.2f}s."
    )

#import os
#import time
#import numpy as np
#import SimpleITK as sitk
#import pandas as pd
#from scipy import ndimage
#from pathlib import Path
#from concurrent.futures import ProcessPoolExecutor, as_completed
#from loguru import logger
#import argparse
#import shutil
#import concurrent.futures
#try:
#    import cupy as cp
#    GPU_AVAILABLE = True
#except ImportError:
#    cp = None
#    GPU_AVAILABLE = False
#from completeme import LABEL_COMPLETION_CHECKPOINT_DIR
#from LTN import eval_model
#
#def gaussian_filter_gpu(mask, sigma):
#    arr_gpu = cp.asarray(mask)
#    filtered_gpu = cp.ndimage.gaussian_filter(arr_gpu, sigma=sigma)
#    return cp.asnumpy(filtered_gpu)
#
#def one_hot_labelmap_with_mask_fast(path, smoothing_sigma=0, save_path=None, use_gpu=False):
#    """
#    Converts a multi-label segmentation to a one-hot representation, ensuring all 
#    expected labels are present. If a label is missing, it adds a single random
#    point for that label.
#    
#    Args:
#        path (str): The file path to the input segmentation label map.
#        expected_labels (list or tuple): The complete set of labels to include
#            in the one-hot output (e.g., [0, 1, 2, 3, 4, 5, 6, 7, 8]).
#        smoothing_sigma (float): Sigma for Gaussian smoothing. Set to 0 for no smoothing.
#        save_path (str): The directory to save the output one-hot image.
#        use_gpu (bool): Flag to use GPU-based Gaussian filter (if available).
#    
#    Returns:
#        str: The file path of the saved one-hot image.
#    """
#    # Read the input label map
#    labelmap = sitk.ReadImage(path, sitk.sitkInt64)
#    lab_array = sitk.GetArrayFromImage(labelmap)
#    expected_labels = [0,1,2,3,4,5,6,7,8]
#    h, w, d = lab_array.shape
#    
#    # Create the final one-hot array with a channel for every expected label
#    lab_array_one_hot = np.zeros((h, w, d, len(expected_labels)), dtype=np.float32)
#
#    # Convert the array to a set for fast lookup
#    present_labels = set(np.unique(lab_array))
#    
#    for idx, lab in enumerate(expected_labels):
#        if lab in present_labels:
#            # If the label is present in the image, create the mask as before
#            mask = (lab_array == lab).astype(np.float32)
#        else:
#            # If the label is missing, create a blank mask and add a random point.
#            # This is done to ensure the channel exists.
#            print(f"Warning: Label {lab} is missing. Adding a random point to the output.")
#            mask = np.zeros((h, w, d), dtype=np.float32)
#            
#            # Generate a random 3D coordinate and set the voxel to 1
#            rand_z, rand_y, rand_x = np.random.randint(0, h), np.random.randint(0, w), np.random.randint(0, d)
#            mask[rand_z, rand_y, rand_x] = 1.0
#
#        if smoothing_sigma > 0:
#            if use_gpu:
#                # Assuming this function is defined elsewhere
#                mask = gaussian_filter_gpu(mask, sigma=smoothing_sigma)
#            else:
#                mask = ndimage.gaussian_filter(mask, sigma=smoothing_sigma, mode='nearest')
#        
#        lab_array_one_hot[..., idx] = mask
#
#    # Convert the one-hot array back to a SimpleITK image
#    labelmap_one_hot = sitk.GetImageFromArray(lab_array_one_hot, isVector=True)
#    labelmap_one_hot.CopyInformation(labelmap)
#
#    # Save the output image
#    output_file = os.path.join(save_path, os.path.basename(path))
#    sitk.WriteImage(labelmap_one_hot, output_file)
#    
#    return output_file
#
##def one_hot_labelmap_with_mask_fast(path, smoothing_sigma=0, save_path=None, use_gpu=False):
##    labelmap = sitk.ReadImage(path, sitk.sitkInt64)
##    lab_array = sitk.GetArrayFromImage(labelmap)
##    labels = np.unique(lab_array)
##    labels.sort()
##
##    h, w, d = lab_array.shape
##    lab_array_one_hot = np.zeros((h, w, d, labels.size), dtype=np.float32)
##
##    for idx, lab in enumerate(labels):
##        mask = (lab_array == lab).astype(np.float32)
##        if smoothing_sigma > 0:
##            mask = gaussian_filter_gpu(mask, sigma=smoothing_sigma) if use_gpu else ndimage.gaussian_filter(mask, sigma=smoothing_sigma, mode='nearest')
##        lab_array_one_hot[..., idx] = mask
##
##    labelmap_one_hot = sitk.GetImageFromArray(lab_array_one_hot, isVector=True)
##    labelmap_one_hot.CopyInformation(labelmap)
##
##    output_file = save_path / os.path.basename(path)
##    sitk.WriteImage(labelmap_one_hot, str(output_file))
##    return str(output_file)
#
#def process_label_file_worker(path, output_path, smoothing_sigma=0, use_gpu=False):
#    try:
#        return one_hot_labelmap_with_mask_fast(path, smoothing_sigma, save_path=output_path, use_gpu=use_gpu)
#    except Exception as e:
#        logger.warning(f"Failed to process {path}: {e}")
#        return None
#
#def to_one_hot(label_folder, output_path, is_sparse=True, max_workers=8, use_gpu=False):
#    volume_type = 'sparse' if is_sparse else 'dense'
#    file_paths = sorted([str(Path(label_folder) / f) for f in os.listdir(label_folder) if f.endswith('.nii.gz')])
#
#    with ProcessPoolExecutor(max_workers=max_workers) as executor:
#        futures = {
#            executor.submit(process_label_file_worker, f, output_path, 0, use_gpu): f
#            for f in file_paths
#        }
#        output_files = []
#        for future in as_completed(futures):
#            result = future.result()
#            if result:
#                output_files.append(result)
#
#    # Save processed list
#    pd.DataFrame({'img': output_files}).to_csv(output_path / f'3d_{volume_type}_oh.csv', index=False)
#
#def process_case(case: Path, output_path: Path, path_ltn: Path, max_workers: int, use_gpu: bool):
#    logger.info(f"Processing case: {case.name}")
#    output_case_dir = output_path / case.name
#    output_case_dir.mkdir(parents=True, exist_ok=True)
#
#    # Step 1: convert to one-hot
#    to_one_hot(case, output_case_dir, max_workers=max_workers, use_gpu=use_gpu)
#
#    # Step 2: apply LTN
#    output_LTN = output_path / f"{case.name}_LTN"
#    output_LTN.mkdir(parents=True, exist_ok=True)
#
#    eval_model(path_ltn, output_case_dir, output_LTN, out_channel=12, in_channel=9)
#    shutil.rmtree(output_case_dir)
#
#def main():
#    parser = argparse.ArgumentParser(description='Convert 2D sparse volumes to 3D one-hot encoded volumes')
#    parser.add_argument('-i', '--input', type=Path, default='./volume_niftis', help='Folder containing sparse volumes')
#    parser.add_argument('-o', '--output_path', type=Path, default='./volume_niftis_oh', help='Folder to save output')
#    parser.add_argument('--max_workers', type=int, default=8, help='Max number of worker threads')
#    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for filtering (requires CuPy)')
#    args = parser.parse_args()
#
#    if args.use_gpu and not GPU_AVAILABLE:
#        logger.warning("GPU requested but CuPy is not installed. Falling back to CPU.")
#        args.use_gpu = False
#
#    assert args.input.exists(), f'Input path not found: {args.input}'
#    assert Path(LABEL_COMPLETION_CHECKPOINT_DIR).exists(), f'LTN checkpoint not found: {Path(LABEL_COMPLETION_CHECKPOINT_DIR)}'
#    args.output_path.mkdir(parents=True, exist_ok=True)
#
#    case_dirs = [Path(args.input, c) for c in sorted(os.listdir(args.input)) if Path(args.input, c).is_dir()]
#
#    logger.info(f"Found {len(case_dirs)} cases to process.")
#    start_time = time.time()
#
#    for case in case_dirs:
#        try:
#            logger.info(f"Processing case: {case.name}")
#            output_case_dir = args.output_path / case.name
#            output_case_dir.mkdir(parents=True, exist_ok=True)
#
#            to_one_hot(case, output_case_dir, max_workers=args.max_workers, use_gpu=args.use_gpu)
#
#            logger.info(f"Running LTN for {case.name}")
#            output_LTN = args.output_path / f"{case.name}_LTN"
#            output_LTN.mkdir(parents=True, exist_ok=True)
#            eval_model(Path(LABEL_COMPLETION_CHECKPOINT_DIR), output_case_dir, output_LTN, out_channel=12, in_channel=9)
#        except Exception as e:
#            logger.warning(f"Failed to process {case}: {e}")
#            continue
#            #return None
#    elapsed = time.time() - start_time
#    logger.success(f"Done. Processed {len(case_dirs)} cases in {elapsed:.2f} seconds.")
#
#if __name__ == '__main__':
#    main()