"""
run_full_pipeline.py

End-to-end orchestration for the ToF label completion pipeline.

Runs in sequence:
  1. UNet segmentation   — sparse 2D masks from CMR NIfTI files
  2. Label completion    — dense 3D whole-heart volumes from sparse masks

Usage:
    python run_full_pipeline.py --input_dir <nifti_dir> --output_dir <output_dir> [options]

Example:
    python run_full_pipeline.py \
        --input_dir ./example/nifti \
        --output_dir ./output \
        --log
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path


SEGMENTATION_SCRIPT = Path("./src/completeme/segmentation/run_batch_segmentation.py")
LABEL_COMPLETION_SCRIPT = Path("./src/completeme/label_completion/run-pipeline.py")
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log: bool, output_dir: Path) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log:
        output_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(output_dir / "pipeline.log"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run_step(step_name: str, cmd: list[str]) -> None:
    """Run a subprocess command, streaming output. Raises on non-zero exit."""
    logging.info("=" * 60)
    logging.info(f"STARTING: {step_name}")
    logging.info(f"Command : {' '.join(str(c) for c in cmd)}")
    logging.info("=" * 60)

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        logging.error(f"FAILED: {step_name} (exit code {result.returncode})")
        sys.exit(result.returncode)

    logging.info(f"COMPLETED: {step_name}")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step1_segmentation(input_dir: Path, seg_output_dir: Path) -> None:
    """Run UNet segmentation on raw CMR NIfTI files."""
    cmd = [
        sys.executable,
        str(SEGMENTATION_SCRIPT),
        "-b", str(input_dir),
        "-o", str(seg_output_dir),
    ]
    run_step("Step 1 — UNet Segmentation", cmd)


def step2_label_completion(seg_output_dir: Path, dense_output_dir: Path, log: bool) -> None:
    """Run label completion to produce dense 3D volumes from sparse masks."""
    cmd = [
        sys.executable,
        str(LABEL_COMPLETION_SCRIPT),
        "--all_components",
        "--input_dir", str(seg_output_dir),
        "--output_dir", str(dense_output_dir),
    ]
    if log:
        cmd.append("--log")
    run_step("Step 2 — Label Completion (sparse → dense 3D)", cmd)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end ToF label completion pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Root folder containing per-patient subdirectories of raw CMR NIfTI files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./pipeline_output"),
        help="Root folder for all pipeline outputs.",
    )
    parser.add_argument(
        "--seg_dir",
        type=Path,
        default=None,
        help=(
            "Where to write Step 1 segmentation masks. "
            "Defaults to <output_dir>/segmentation_masks."
        ),
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Write a pipeline.log file inside --output_dir.",
    )
    parser.add_argument(
        "--skip_segmentation",
        action="store_true",
        help=(
            "Skip Step 1 and use existing masks in --seg_dir "
            "(or <output_dir>/segmentation_masks). "
            "Useful when re-running Step 2 after a failure."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve intermediate segmentation directory
    seg_dir = args.seg_dir or (args.output_dir / "segmentation_masks")

    setup_logging(args.log, args.output_dir)

    logging.info("ToF Label Completion — full pipeline")
    logging.info(f"  Input NIfTI dir  : {args.input_dir}")
    logging.info(f"  Segmentation dir : {seg_dir}")
    logging.info(f"  Dense volume dir : {args.output_dir}")

    # Validate input
    if not args.input_dir.exists():
        logging.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Step 1
    if args.skip_segmentation:
        logging.info("Skipping Step 1 (--skip_segmentation set).")
        if not seg_dir.exists():
            logging.error(
                f"--skip_segmentation requires existing masks at: {seg_dir}"
            )
            sys.exit(1)
    else:
        step1_segmentation(args.input_dir, seg_dir)

    # Step 2
    step2_label_completion(seg_dir, args.output_dir, args.log)

    logging.info("=" * 60)
    logging.info("Pipeline complete.")
    logging.info(f"Dense 3D volumes saved to: {args.output_dir / '3d_dense_volumes'}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()