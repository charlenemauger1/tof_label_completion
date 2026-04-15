"""
transform_to_volume.py

Resamples multi-view 2D+t NIfTI segmentation files into a single dense 3D
sparse volume in cardiac coordinate space, using GPU-accelerated grid sampling.

For each case, landmarks are extracted from the 4-chamber view to define a
cardiac coordinate system. Each NIfTI slice is then resampled into this common
space and merged via max-projection into a final 3D volume.

Usage:
    python transform_to_volume.py -s <patient_folder> -o <output_folder>
    python transform_to_volume.py -i <main_folder>    -o <output_folder>
"""

from __future__ import annotations

import os
import time
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import scipy.ndimage
from scipy.spatial import distance
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

from run_SSA import extract_info

# ── Label map ───────────────────────────────────────────────────────────────

LABEL_MAP: dict[str, int] = {
    "lv-myo": 1,
    "lv":     2,
    "rv":     3,
    "ra":     4,
    "la":     5,
    "ao":     6,
    "pv":     7,
    "rv-myo": 8,
}

# Labels to zero-out per view — keeps only anatomically relevant structures
VIEW_ZERO_LABELS: dict[str, list[int]] = {
    "sax":    [LABEL_MAP["ra"], LABEL_MAP["la"], LABEL_MAP["ao"], LABEL_MAP["pv"]],
    "2ch_lt": [LABEL_MAP["ra"], LABEL_MAP["rv"], LABEL_MAP["rv-myo"], LABEL_MAP["ao"], LABEL_MAP["pv"]],
    "rvot":   [LABEL_MAP["lv"], LABEL_MAP["lv-myo"], LABEL_MAP["ra"], LABEL_MAP["ao"], LABEL_MAP["la"]],
    "3ch":    [LABEL_MAP["ra"], LABEL_MAP["pv"]],
    "2ch_rt": [LABEL_MAP["lv"], LABEL_MAP["lv-myo"], LABEL_MAP["ra"], LABEL_MAP["ao"], LABEL_MAP["la"], LABEL_MAP["pv"]],
    "4ch":    [LABEL_MAP["ao"], LABEL_MAP["pv"]],
}

# Views that require RV myocardium largest-connected-component filtering
VIEWS_WITH_RV_MYO_FILTER: set[str] = {"rvot", "4ch"}

OUTPUT_SIZE = 160

# ── Precomputed global grid ──────────────────────────────────────────────────
# Built once at module load; shared across all worker threads.

_arange = np.arange(OUTPUT_SIZE)
_grid = np.stack(
    np.meshgrid(_arange, _arange, _arange, indexing="ij"), axis=-1
)  # (160, 160, 160, 3)

_ijk_mtx = np.ones((OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 4), dtype=np.float32)
_ijk_mtx[..., :3] = _grid
IJK_MTX_FLAT: np.ndarray = _ijk_mtx.reshape(-1, 4).T  # (4, 160³)


# ── Utilities ────────────────────────────────────────────────────────────────

def get_largest_cc(segmentation: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask containing only the largest connected component.

    Args:
        segmentation: Binary or integer array of any shape.

    Returns:
        Boolean array of the same shape, True only within the largest component.
        Returns an all-False array if no components are found.
    """
    labels, n = scipy.ndimage.label(segmentation)
    if n == 0:
        return np.zeros_like(segmentation, dtype=bool)
    sizes = np.bincount(labels.flat)[1:]  # skip background (label 0)
    return labels == (np.argmax(sizes) + 1)


def _normalise(v: np.ndarray) -> np.ndarray:
    """Normalise a vector, returning a zero vector if the norm is zero."""
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def _safe_cross_normalise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of a and b, normalised. Falls back to Y then X axis."""
    v = np.cross(a, b)
    norm = np.linalg.norm(v)
    if norm != 0:
        return v / norm
    for fallback in (np.array([0, 1, 0]), np.array([1, 0, 0])):
        v = np.cross(a, fallback)
        norm = np.linalg.norm(v)
        if norm != 0:
            return v / norm
    return np.array([1.0, 0.0, 0.0])


# ── Landmark generation ───────────────────────────────────────────────────────

def landmark_gen(
    path: str | Path,
    output_size: int = OUTPUT_SIZE,
) -> tuple[list[np.ndarray], np.ndarray, float, float, np.ndarray]:
    """
    Derive a cardiac coordinate system from a 4-chamber segmentation.

    Computes anatomical landmarks (apex, mitral valve centre, tricuspid valve
    centre, LV/RV centroids, heart centre) and uses them to build an affine
    matrix mapping voxel indices to patient-space cardiac coordinates.

    Args:
        path: Path to the 4-chamber NIfTI segmentation file.
        output_size: Size of the cubic output volume (default 160).

    Returns:
        Tuple of:
            landmarks_list_voxel_space: List of landmark coordinates in voxel space.
            landmarks_patient_space: (3, 6) array of landmarks in patient space.
            new_space: Isotropic voxel spacing for the output volume (mm).
            max_distance: Apex-to-base distance used to set the field of view (mm).
            new_affine_mtx: (4, 4) affine mapping output voxel indices to patient space.
    """
    seg_img = nib.load(path)
    seg = seg_img.get_fdata()
    affine = seg_img.get_qform()

    # Binary masks for each chamber, keeping only the largest connected component
    segs = {
        name: get_largest_cc(seg == val)
        for val, name in {2: "LV", 3: "RV", 4: "RA", 5: "LA"}.items()
    }
    seg_binary = get_largest_cc(seg != 0)

    # Mitral valve centre: intersection of LV and dilated LA
    mvc = np.mean(
        np.where(segs["LV"] & scipy.ndimage.binary_dilation(segs["LA"])), axis=1
    )

    # Apex: LV point furthest from MVC
    lv_coords = np.argwhere(segs["LV"])
    if lv_coords.size == 0:
        raise ValueError("No LV segmentation found.")
    apex = lv_coords[np.argmax(np.sum((lv_coords - mvc) ** 2, axis=1))]

    # Chamber centroids
    lvc = np.mean(np.argwhere(segs["LV"]), axis=0)
    rvc = np.mean(np.argwhere(segs["RV"]), axis=0)

    # Tricuspid valve centre: intersection of RV and dilated RA
    tvc = np.mean(
        np.where(segs["RV"] & scipy.ndimage.binary_dilation(segs["RA"])), axis=1
    )

    # Heart centre of mass
    coh = np.mean(np.argwhere(seg_binary), axis=0)

    # Transform landmarks to patient space in one matrix multiplication
    voxel_landmarks = np.array([apex, mvc, tvc, lvc, rvc, coh]).T  # (3, 6)
    homogeneous = np.vstack([voxel_landmarks, np.ones((1, 6))])     # (4, 6)
    patient_landmarks = (affine @ homogeneous)[:3]                  # (3, 6)

    apex_ps, mvc_ps, tvc_ps, coh_ps = (
        patient_landmarks[:, 0],
        patient_landmarks[:, 1],
        patient_landmarks[:, 2],
        patient_landmarks[:, 5],
    )

    # Field of view: distance from apex to the farthest heart surface point
    # projected along the apex→mvc axis
    surface = scipy.ndimage.binary_dilation(seg_binary) & ~seg_binary
    surface_pts = np.argwhere(surface)

    if surface_pts.size == 0:
        logger.warning("No surface points found; defaulting max_distance to 1.")
        max_distance = 1.0
    else:
        vec_apex_mvc = apex - mvc
        dots = np.dot(surface_pts - mvc, vec_apex_mvc)
        farthest_ps = (affine @ np.append(surface_pts[np.argmin(dots)], 1))[:3]
        max_distance = distance.euclidean(farthest_ps, apex_ps)

    # Build cardiac coordinate basis (orthonormal)
    vec3 = _normalise(mvc_ps - apex_ps)
    vec2 = _safe_cross_normalise(mvc_ps - apex_ps, tvc_ps - mvc_ps)
    vec1 = _safe_cross_normalise(vec2, mvc_ps - apex_ps)
    vec3 = _normalise(np.cross(vec1, vec2))  # recompute for orthonormality

    new_space = max(max_distance, 1e-6) * 1.3 / output_size

    rotation = np.column_stack([vec1, vec2, vec3]) * new_space  # (3, 3)
    centre = np.full(3, output_size / 2)
    translation = coh_ps - rotation @ centre

    new_affine = np.eye(4)
    new_affine[:3, :3] = rotation
    new_affine[:3, 3] = translation

    landmarks_voxel = [apex, mvc, tvc, lvc, rvc, coh]
    return landmarks_voxel, patient_landmarks, new_space, max_distance, new_affine


# ── Per-slice processing ──────────────────────────────────────────────────────

def _prepare_slice(
    data: np.ndarray,
    thickness: int = 3,
) -> np.ndarray:
    """
    Extract a 2D slice from an array of arbitrary dimensionality and repeat
    it along a new depth axis to give it physical thickness.

    Args:
        data: Input array of shape (W, H), (W, H, D), or (W, H, D, 1).
        thickness: Number of times to repeat the slice along the depth axis.

    Returns:
        Float32 array of shape (thickness, H, W).

    Raises:
        ValueError: If the array has an unsupported shape.
    """
    if data.ndim == 2:
        s = data.T
    elif data.ndim == 3:
        s = data[:, :, data.shape[2] // 2].T
    elif data.ndim == 4 and data.shape[3] == 1:
        s = data[:, :, data.shape[2] // 2, 0].T
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
    return np.stack([s] * thickness, axis=0).astype(np.float32)  # (thickness, H, W)


def _apply_view_filter(data: np.ndarray, view_key: str) -> np.ndarray:
    """
    Zero out labels that are not expected in a given CMR view.

    For RVOT and 4CH views, additionally keeps only the largest connected
    component of RV myocardium.

    Args:
        data: Integer label array (any shape).
        view_key: Lowercase view identifier (e.g. 'sax', '4ch', 'rvot').

    Returns:
        Filtered label array of the same shape.
    """
    result = data.copy()

    for key, labels in VIEW_ZERO_LABELS.items():
        if key in view_key:
            result[np.isin(result, labels)] = 0
            break

    if any(v in view_key for v in VIEWS_WITH_RV_MYO_FILTER):
        rv_myo = LABEL_MAP["rv-myo"]
        rv_mask = get_largest_cc(result == rv_myo)
        result[result == rv_myo] = 0
        result[rv_mask] = rv_myo

    return result


def _resample_slice_to_volume(
    input_tensor: torch.Tensor,
    data_affine: np.ndarray,
    new_affine: np.ndarray,
    ijk_flat: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Resample a 2D+thickness slice into the target 3D cardiac coordinate volume
    using GPU-accelerated grid sampling.

    Args:
        input_tensor: Float32 tensor of shape (1, 1, D, H, W) on CPU.
        data_affine: (4, 4) affine of the source NIfTI slice.
        new_affine: (4, 4) target cardiac coordinate affine.
        ijk_flat: Precomputed (4, N) grid of homogeneous voxel coordinates.
        device: Torch device to run sampling on.

    Returns:
        Float32 NumPy array of shape (160, 160, 160).
    """
    affine_inter = torch.from_numpy(
        (np.linalg.inv(data_affine) @ new_affine).astype(np.float32)
    ).to(device)

    ijk_dev = torch.from_numpy(ijk_flat).to(device)
    xyz = (affine_inter @ ijk_dev)[:3]  # (3, N)

    grid = xyz.T.reshape(OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 3).contiguous()
    grid = grid.unsqueeze(0)  # (1, 160, 160, 160, 3)

    # Normalise to [-1, 1] as required by grid_sample
    _, _, D, H, W = input_tensor.shape
    grid[..., 0] = (grid[..., 0] - (W - 1) / 2) / ((W - 1) / 2 + 0.5)
    grid[..., 1] = (grid[..., 1] - (H - 1) / 2) / ((H - 1) / 2 + 0.5)
    grid[..., 2] = (grid[..., 2] - (D - 1) / 2) / ((D - 1) / 2 + 0.5)

    grid = grid.to(device)
    out = F.grid_sample(
        input_tensor.to(device),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )
    return out[0, 0].cpu().numpy()


# ── Case processing ───────────────────────────────────────────────────────────

def process_case(
    case_name: str,
    data_path: Path,
    output_folder: Path,
    ijk_mtx_flat: np.ndarray,
) -> None:
    """
    Resample all NIfTI slices for a single case into a 3D cardiac volume.

    Locates the 4-chamber view to define the cardiac coordinate system, then
    resamples each non-5CH NIfTI slice into that space, merging via
    max-projection into a final 3D sparse volume saved as a NIfTI file.

    Args:
        case_name: Name of the case subfolder (e.g. 'patient_001_2').
        data_path: Parent directory containing case subfolders.
        output_folder: Directory to save the output NIfTI volume.
        ijk_mtx_flat: Precomputed (4, N) grid of homogeneous voxel coordinates.
    """
    case_path = data_path / case_name
    ref_name = "_".join(case_name.split("_")[:-1]) + "_001"
    ref_path = data_path / ref_name

    # Locate 4-chamber reference file
    fch_files = list(ref_path.glob("*4ch*")) + list(ref_path.glob("*4CH*"))
    if not fch_files:
        logger.warning(f"No 4CH file found for {ref_name} — skipping {case_name}.")
        return

    _, _, _, _, new_affine = landmark_gen(str(fch_files[0]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    volume = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)

    nii_files = [
        f for f in case_path.iterdir()
        if f.suffix in {".nii", ".gz"} or f.name.endswith(".nii.gz")
        if "5ch" not in f.name.lower()
    ]

    for nii_path in nii_files:
        info = extract_info(str(nii_path), False)
        data_array = info[0]
        data_affine = info[1].copy().astype(np.float64)

        view_key = nii_path.name.lower()
        data_array = _apply_view_filter(data_array, view_key)

        # Normalise through-plane affine column
        col_norm = np.linalg.norm(data_affine[:, 2])
        if col_norm != 0:
            data_affine[:, 2] /= col_norm

        slice_3d = _prepare_slice(data_array, thickness=3)
        input_tensor = torch.from_numpy(slice_3d[None, None])  # (1, 1, D, H, W)

        resampled = _resample_slice_to_volume(
            input_tensor, data_affine, new_affine, ijk_mtx_flat, device
        )
        np.maximum(volume, resampled, out=volume)

    nib.save(
        nib.Nifti1Image(volume.astype(np.uint8), affine=new_affine),
        output_folder / f"{case_name}.nii.gz",
    )
    logger.info(f"Saved: {case_name}.nii.gz")


# ── Patient folder processing ─────────────────────────────────────────────────
def process_patient_folder(
    patient_folder: Path,
    output_path: Path,
) -> None:
    """
    Process all case subfolders within a single patient folder.

    Creates the output directory and iterates over all non-JSON entries,
    calling process_case for each.

    Args:
        patient_folder: Directory containing per-timeframe case subfolders.
        output_path: Root output directory; a subfolder named after the patient
                     will be created within it.
    """
    logger.info(f"Processing patient: {patient_folder.name}")
    output_nifti = output_path / patient_folder.name
    output_nifti.mkdir(parents=True, exist_ok=True)

    cases = sorted(
        d for d in os.listdir(patient_folder) if not d.endswith(".json")
    )
    for case_name in cases:
        process_case(case_name, patient_folder, output_nifti, IJK_MTX_FLAT)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Resample multi-view 2D+t NIfTI segmentations into a dense 3D "
            "sparse volume in cardiac coordinate space."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-i", "--input_main_folder",
        type=Path,
        help="Root folder containing one patient subfolder per subject.",
    )
    group.add_argument(
        "-s", "--input_subfolder",
        type=Path,
        help="Single patient subfolder to process.",
    )
    parser.add_argument(
        "-o", "--output_path",
        type=Path,
        default=Path("./corrected_niftis"),
        help="Output directory for 3D NIfTI volumes.",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable logging to file and console.",
    )
    parser.add_argument(
        "--no_multiprocessing",
        action="store_true",
        help="Disable parallel processing across patient folders.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of worker threads (default: 4).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    assert args.output_path.parent.exists(), (
        f"Cannot create output directory: parent of {args.output_path} does not exist."
    )
    args.output_path.mkdir(parents=True, exist_ok=True)

    if args.log:
        logger.add(
            str(args.output_path / "transform_log.log"),
            rotation="5 MB",
            level="INFO",
            enqueue=True,
        )
        logger.info("Logging enabled.")
    else:
        logger.remove()
        logger.configure(handlers=[{"sink": lambda msg: None}])

    # Collect patient folders
    if args.input_main_folder:
        if not args.input_main_folder.is_dir():
            logger.error(f"'{args.input_main_folder}' is not a valid directory.")
            raise SystemExit(1)
        patient_folders = sorted(
            f for f in args.input_main_folder.iterdir() if f.is_dir()
        )
        logger.info(f"Found {len(patient_folders)} patient folders.")
    else:
        if not args.input_subfolder.is_dir():
            logger.error(f"'{args.input_subfolder}' is not a valid directory.")
            raise SystemExit(1)
        patient_folders = [args.input_subfolder]

    if not patient_folders:
        logger.warning("No patient folders found. Exiting.")
        raise SystemExit(0)

    start = time.time()
    use_mp = args.input_main_folder and not args.no_multiprocessing

    if use_mp:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(process_patient_folder, f, args.output_path): f
                for f in patient_folders
            }
            for future in as_completed(futures):
                folder = futures[future]
                try:
                    future.result()
                    logger.info(f"Finished: {folder.name}")
                except Exception as e:
                    logger.error(f"Failed: {folder.name} — {e}")
    else:
        for folder in patient_folders:
            try:
                process_patient_folder(folder, args.output_path)
            except Exception as e:
                logger.error(f"Failed: {folder.name} — {e}")

    logger.info(f"Processed {len(patient_folders)} patient(s) in {time.time() - start:.1f}s")
    logger.success(f"Done. Results saved to {args.output_path}")