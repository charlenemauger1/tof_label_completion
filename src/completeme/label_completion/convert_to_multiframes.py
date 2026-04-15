import nibabel as nib
import numpy as np
from pathlib import Path
import argparse
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import ndimage  

mappings = {
    "sa":     "0:0,1:2,2:1,3:3,4:8",    
    "lt":     "0:0,1:2,2:1,3:5,4:0,5:0,6:0",
    "rt":     "0:0,1:3,2:8,3:0,4:0,5:0,6:0",
    "3ch":    "0:0,1:2,2:1,3:3,4:8,5:5,6:6",
    "4ch":    "0:0,1:2,2:1,3:3,4:8,5:5,6:4",
    "rvot":   "0:0,1:3,2:8,3:7,4:0,5:0,6:0",
}

# Configuration for the network

# LV_MYO_LABEL = 1
# LV_LABEL = 2
# RV_LABEL = 3
# RA_LABEL = 4
# LA_LABEL = 5
# AORTA = 6
# PULMONARY = 7
# RV_MYO = 8

def parse_mapping(mapping_str: str) -> dict[int, int]:
    """
    Parse a label remapping string into a dictionary of integer label mappings.

    Args:
        mapping_str: Comma-separated pairs of original:new label indices
                     (e.g. "1:3,2:4,3:5").

    Returns:
        Dict mapping original label indices to new label indices.
    """
    return {int(orig): int(new) for orig, new in (pair.split(":") for pair in mapping_str.split(","))}

parsed_mappings: dict[str, dict[int, int]] = {
    key: parse_mapping(value) for key, value in mappings.items()
}

def resample_nifti(nifti_image_data: np.ndarray, reference_timeframe: int) -> np.ndarray:
    """
    Resample a 4D NIfTI array along the temporal axis to a target number of frames.

    Spatial dimensions are preserved; only the third axis (time) is resampled
    using nearest-neighbour interpolation.

    Args:
        nifti_image_data: Input array of shape (H, W, T, N).
        reference_timeframe: Target number of time frames.

    Returns:
        Resampled float32 array of shape (H, W, reference_timeframe, N).
    """
    zoom_factors = (1.0, 1.0, reference_timeframe / nifti_image_data.shape[2], 1.0)
    return ndimage.zoom(nifti_image_data, zoom_factors, order=0).astype(np.float32)

def remap_labels(data: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """
    Remap integer label values in an array using a lookup table.

    Builds a lookup table (LUT) of size max(label) + 1 and applies it in a
    single vectorised operation, which is significantly faster than iterating
    over individual label values.

    Args:
        data: Integer label array of any shape.
        mapping: Dict mapping original label indices to new label indices.

    Returns:
        Remapped array of the same shape and dtype as data.
    """
    max_index = max(int(data.max()), max(mapping.keys()))
    lut = np.arange(max_index + 1)
    for orig_val, new_val in mapping.items():
        lut[orig_val] = new_val

    return lut[data.astype(int)]

def process_patient_folder(
    patient_folder: Path,
    output_folder: Path,
    logging_enabled: bool,
) -> None:
    """
    Process all NIfTI files in a patient folder, resampling files whose
    temporal dimension differs from the most common frame count.

    Loads each file once to determine the modal frame count, then processes
    each file — passing a resample target only when needed.

    Args:
        patient_folder: Directory containing the patient's NIfTI files.
        output_folder: Directory to write processed outputs.
        logging_enabled: Whether to enable logging during processing.
    """

    nifti_files = sorted(patient_folder.glob("*.nii.gz"))
    if not nifti_files:
        return

    # Compute frame counts without caching file handles — avoids memory leak
    frame_counts = []
    for f in nifti_files:
        data = nib.load(f).get_fdata()
        if data.ndim < 4:
            data = data[..., np.newaxis]
        frame_counts.append(data.shape[2])

    modal_frames = max(set(frame_counts), key=frame_counts.count)

    for nifti_file, n_frames in zip(nifti_files, frame_counts):
        resample = None if n_frames == modal_frames else modal_frames
        convert_single_frame(patient_folder, nifti_file, output_folder, logging_enabled, resample=resample)

def convert_single_frame(
    patient_folder: Path,
    nifti_file: Path,
    output_folder: Path,
    logging_enabled: bool,
    resample: int | None = None,
) -> None:
    """
    Convert a single NIfTI cine file into per-frame NIfTI files.

    Optionally remaps label indices based on filename keywords and resamples
    the temporal dimension to a target frame count. Skips frames with empty
    segmentations. Each frame is saved as a separate NIfTI file under:
        output_folder / patient_id / patient_id_<frame_idx> / <slice>_f<frame>.nii.gz

    Args:
        patient_folder: Patient directory (used to derive output subfolder names).
        nifti_file: Path to the input NIfTI file.
        output_folder: Root output directory.
        logging_enabled: Whether to log each saved file.
        resample: If set, resample the temporal axis to this number of frames.
    """
    try:
        img = nib.load(nifti_file)
        data = img.get_fdata()
        affine = img.get_qform()

        # Ensure 4D shape (H, W, T, N)
        if data.ndim < 4:
            data = data[..., np.newaxis]

        if resample is not None:
            data = resample_nifti(data, resample)

        num_frames = data.shape[2]

        # Remap labels if filename matches a known mapping key
        lower_name = nifti_file.name.lower()
        for key, mapping in parsed_mappings.items():
            if key in lower_name:
                data = remap_labels(data, mapping).astype(np.float32)
                break

        # Normalise the through-plane component of the affine once
        affine[:3, 2] /= np.linalg.norm(affine[:3, 2])

        patient_id = patient_folder.name

        for frame_idx in range(num_frames):
            # Skip empty segmentation frames
            if np.count_nonzero(data[:, :, frame_idx]) == 0:
                logger.info(f"{nifti_file} frame {frame_idx + 1} has no segmentation, skipping.")
                continue

            frame_folder = output_folder / patient_id / f"{patient_id}_{(frame_idx + 1):03}"
            frame_folder.mkdir(parents=True, exist_ok=True)

            frame_file = frame_folder / f"{nifti_file.stem}_f{(frame_idx + 1):03}.nii.gz"
            nib.save(nib.Nifti1Image(data[:, :, frame_idx].astype(np.uint8), affine=affine), frame_file)

            if logging_enabled:
                logger.info(f"Saved: {frame_file}")

    except nib.filebasedimages.ImageFileError as e:
        logger.error(f"Error loading {nifti_file}: {e}. Skipping.")
    except Exception as e:
        logger.error(f"Unexpected error processing {nifti_file}: {e}")

def from_multiframe_to_single_frame(
    input_path: str | Path,
    output_folder: str | Path,
    logging_enabled: bool,
    is_subfolder_mode: bool,
) -> None:
    """
    Convert multi-frame NIfTI cine files to per-frame NIfTI files for one or
    more patient folders, processing them in parallel using a thread pool.

    Args:
        input_path: Path to a single patient folder (if is_subfolder_mode)
                    or a root folder containing one subfolder per patient.
        output_folder: Root directory to write per-frame outputs.
        logging_enabled: Whether to log progress to file and stdout.
        is_subfolder_mode: If True, treat input_path as a single patient folder.
                           If False, iterate over all subfolders within input_path.
    """
    output_folder = Path(output_folder)
    input_path = Path(input_path)

    # Configure logging
    if logging_enabled:
        logger.add("conversion_log.log", level="INFO", rotation="1 MB", enqueue=True)
        logger.info("Logging enabled.")
    else:
        logger.remove()
        logger.configure(handlers=[{"sink": lambda msg: None}])

    # Collect patient folders to process
    if is_subfolder_mode:
        if not input_path.is_dir():
            logger.error(f"'{input_path}' is not a valid directory.")
            return
        patient_folders = [input_path]
        logger.info(f"Running on single subfolder: {input_path.name}")
    else:
        if not input_path.is_dir():
            logger.error(f"'{input_path}' is not a valid directory.")
            return
        patient_folders = [f for f in input_path.iterdir() if f.is_dir()]
        logger.info(f"Found {len(patient_folders)} subfolders in: {input_path}")

    if not patient_folders:
        logger.warning("No patient folders found to process. Exiting.")
        return

    # Process patient folders in parallel
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_patient_folder, folder, output_folder, logging_enabled): folder
            for folder in patient_folders
        }
        for future in as_completed(futures):
            folder = futures[future]
            try:
                future.result()
                logger.info(f"Finished: {folder.name}")
            except Exception as e:
                logger.error(f"Failed: {folder.name} — {e}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert multi-frame NIfTI files to single-frame NIfTI slices.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )

    # Create a mutually exclusive group for input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-i", "--input_main_folder", 
        help="Path to the main input folder containing N subfolders (e.g., patient IDs)."
    )
    group.add_argument(
        "-s", "--input_subfolder", 
        help="Path to a single subfolder (e.g., a specific patient ID) to process."
    )

    parser.add_argument(
        "-o", "--output_folder", 
        required=True, 
        help="Output folder for single-frame NIfTI files. "
             "Subfolders will be created within this based on patient IDs."
    )
    parser.add_argument(
        "--log", 
        action="store_true", 
        help="Enable logging to a 'conversion_log.log' file and console."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Determine which input mode was used
    is_subfolder_mode = False
    input_path = None

    if args.input_main_folder:
        input_path = args.input_main_folder
    elif args.input_subfolder:
        input_path = args.input_subfolder
        is_subfolder_mode = True

    if input_path:
        from_multiframe_to_single_frame(input_path, args.output_folder, args.log, is_subfolder_mode)
    else:
        # This case should ideally not be reached because of required=True in the mutually exclusive group
        logger.error("No input folder provided. Use either --input_main_folder or --input_subfolder.")
