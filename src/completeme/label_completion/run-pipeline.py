import argparse
import warnings
from pathlib import Path
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import time
from loguru import logger
import sys
import subprocess

def process_single_case(case_path: Path, args, output_dir: Path):
    """
    Processes a single case directory through the defined pipeline steps.
    This function will be run by each worker in the ThreadPoolExecutor.
    """
    case_stem = case_path.stem
    logger.info(f"--- Starting processing for case: {case_stem} ---")

    script_dir = Path(__file__).parent
    if args.all_components:
        args.preprocessing, args.slice_shifting, args.volume_conversion, args.sparse_to_dense = True, True, True, True
        output_dir = args.output_dir

    try:
        if args.preprocessing:
            output_folder_preproc = output_dir / Path('segmentations_cleaned')
            output_folder_preproc.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
            logger.info(f"  Preprocessing: convert_to_multiframes.py for {case_stem}")
            subprocess.run([
                 "python", str(script_dir / "convert_to_multiframes.py"),
                 "-s", str(case_path),
                 "-o", str(output_folder_preproc)
             ])

        if args.slice_shifting:
            # Note: For slice_shifting, it uses the STEM of the case for base_folder
            base_folder_ssa = output_dir / Path('segmentations_cleaned') / case_stem
            output_folder_aligned = output_dir / Path('segmentations_aligned')
            output_folder_aligned.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
            logger.info(f"  Slice Shifting: run_SSA.py (calculate) for {case_stem}")
            subprocess.run([
                 "python", str(script_dir / "run_SSA.py"),
                 "-s", str(base_folder_ssa),
                 "-o", str(output_folder_aligned),
                 "-ed", '1',
                 "-step", "calculate"
             ])
            logger.info(f"  Slice Shifting: run_SSA.py (infer) for {case_stem}")
            subprocess.run([
                 "python", str(script_dir / "run_SSA.py"),
                 "-s", str(base_folder_ssa),
                 "-o", str(output_folder_aligned),
                 "-step", "infer"
             ])
        
        if args.volume_conversion:
            # Step 1: Sparse Volume Conversion
            if args.all_components:
                base_folder_sparse = output_dir / Path('segmentations_aligned') / case_stem
                output_folder_sparse_volumes = output_dir / Path('3d_sparse_volumes')
                output_folder_sparse_volumes.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
                logger.info(f"  Volume Conversion: transform_to_volume.py for {case_stem}")
                subprocess.run([
                     "python", str(script_dir / "transform_to_volume.py"),
                     "-s", str(base_folder_sparse),
                     "-o", str(output_folder_sparse_volumes)
                 ])
            else:
                base_folder_sparse = args.input_dir
                output_folder_sparse_volumes = args.output_dir
                subprocess.run([
                     "python", str(script_dir / "transform_to_volume.py"),
                     "-i", str(base_folder_sparse),
                     "-o", str(output_folder_sparse_volumes)
                 ])

        if args.sparse_to_dense:
            # Step 2: Dense Volume Conversion (uses output from previous step as input)
            if args.all_components:
                base_folder_dense = output_dir / Path('3d_sparse_volumes') / case_stem
                output_folder_dense_volumes = output_dir / Path('3d_dense_volumes')
                output_folder_dense_volumes.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
                logger.info(f"  Dense Volume Conversion: from_sparse_to_dense_volume.py for {case_stem}")              
                subprocess.run([
                    "python", str(script_dir / "from_sparse_to_dense_volume.py"),
                    "-s", str(base_folder_dense),
                    "-o", str(output_folder_dense_volumes)
                ])
            else:
                base_folder_dense = args.input_dir
                output_folder_dense_volumes = args.output_dir
                logger.info(f"  Dense Volume Conversion: from_sparse_to_dense_volume.py ")  
                subprocess.run([
                    "python", str(script_dir / "from_sparse_to_dense_volume.py"),
                    "-i", str(base_folder_dense),
                    "-o", str(output_folder_dense_volumes)
                ])
        return
    except Exception as e:
        logger.error(f"!!! Error processing case {case_stem}: {e}")
        

    logger.success(f"--- Finished processing for case: {case_stem} ---")
    return case_stem # Return the case ID for tracking completion

def main():
    parser = argparse.ArgumentParser(description="Pipeline for case processing.")
    parser.add_argument('--input_dir', type=str, required=True, help="Base directory containing case folders.")
    parser.add_argument('--output_dir', type=str, required=True, help="Base directory for all outputs.")
    parser.add_argument('--preprocessing', action='store_true', help="Enable preprocessing step.")
    parser.add_argument('--slice_shifting', action='store_true', help="Enable slice shifting step.")
    parser.add_argument('--volume_conversion', action='store_true', help="Enable volume conversion steps.")
    parser.add_argument('--sparse_to_dense', action='store_true', help="Enable sparse to dense steps.")
    parser.add_argument('--all_components', action='store_true', help="Assume all component flags are true and manage output paths automatically. This might be redundant with the explicit flags, adjust based on your original script's logic.")
    parser.add_argument("--log", action="store_true", help="Enable detailed logging to a file and console.")
    
    # Parse args early to get input_dir and output_dir
    args = parser.parse_args()

    if args.log:
        logger.add(str(Path(args.output_dir) / "lc-pipeline_log.log"), rotation="5 MB", level="INFO", enqueue=True)
        logger.info("Logging enabled.")
    else:
        logger.remove() # Remove default handler
        # To completely silence loguru:
        logger.configure(handlers=[{"sink": lambda msg: None}]) 
        logger.info("Logging disabled. No messages will be shown or saved.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure the top-level output directory exists

    # Discover case directories
    # Assuming 'case_dirs' are subdirectories within input_dir that represent individual cases
    case_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    if not case_dirs:
        logger.error(f"No case directories found in {input_dir}. Exiting.")
        return

    logger.info(f"Found {len(case_dirs)} cases to process.")
    time1 = time.time()

    try:
        for case in case_dirs:
            process_single_case(case, args, output_dir=output_dir)

    except KeyboardInterrupt:
        logger.info(f"Program interrupted by the user")
        sys.exit(0)

    logger.info("\n--- All case processing submitted. Monitoring completion... ---")
    logger.info(f"--- Pipeline execution finished in {time.time() - time1}. ---")


if __name__ == "__main__":
    main()
   
