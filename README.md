# Filling the Gaps: Generating 4D Dense Cardiac Anatomy from Sparse CMR for Enhanced Tetralogy of Fallot Assessment

A two-stage deep learning pipeline for reconstructing dense 3D whole-heart segmentations from sparse 2D cine CMR images in repaired Tetralogy of Fallot (rToF) patients.

The label completion logic in this repository is adapted from the original implementation found at [XESchong/label_completion_pipeline](https://github.com/XESchong/label_completion_pipeline)

Key updates in this version include:
- ToF Compatibility: Retrained weights specifically for Tetralogy of Fallot pathologies.\
- Expanded Label Set: Support for additional cardiac structures beyond the original implementation.\
- Handle highly sparse cardiac segmentations


## Table of Contents
- [**Pipeline**](#pipeline-overview)
- [**Installation**](#installation-guide)
- [**Generating dense segmentation from sparse nifti**](#how-to-run-the-pipline)
    - [Example usage - full pipeline](#example-usage)
    - [Generate segmentation masks from sparse nifti](#generate-segmentations-from-sparse-nifti)
    - [Generate dense volumes from sparse segmentations](#generate-dense-volumes-from-sparse-segmentations)
- [**Contact**](#contact) 

If you use this code for your research, please cite the following publications:

**1. Main Pipeline & Results**\
For a detailed description of this pipeline and the results presented in the paper:


    Mauger, Charlene et al. "Filling the Gaps: Generating 4D Dense Cardiac Anatomy from Sparse CMR for Enhanced Tetralogy of Fallot Assessment" *In review*.

**2. Label Completion Architecture**\
For the methodology regarding the underlying 3D label completion architecture:

    Xu, Yiyang, et al. "Improved 3D whole heart geometry from sparse CMR slices." International Workshop on Statistical Atlases and Computational Models of the Heart. Cham: Springer Nature Switzerland, 2024.

## Pipeline Overview

Standard clinical CMR acquires sparse 2D slices with anisotropic resolution, inter-slice gaps, and motion artifacts — challenges amplified in paediatric populations where incomplete acquisitions are frequent. This pipeline leverages CT-derived 3D segmentations to train a reconstruction network that bridges CMR's clinical accessibility with CT's spatial resolution, without requiring rToF CT data at inference.

```
2D Cine CMR (short + long axis)
        │
        ▼
┌─────────────────────┐
│  UNet Segmentation  │   ← trained on multi-centre rToF CMR
└─────────────────────┘
        │
   Sparse 2D labels
        │
        ▼
┌──────────────────────────┐
│  Label Completion Network │   ← trained on 1,715 CT segmentations
└──────────────────────────┘        + synthetic RV myocardium
        │                           + simulated ToF-specific views
        ▼
Dense 3D Whole-Heart Segmentation
(LV, RV, LVmyo, RVmyo, LA, RA)
```


### Key Features
 
- **Robust to missing data** — validated with 50–70% of short-axis slices randomly removed, simulating incomplete paediatric acquisitions
- **RV myocardium reconstruction** — critical for risk stratification in rToF and absent from prior CMR-based reconstruction methods
- **Cross-modality training** — CT data augmented with synthetic RV myocardium and ToF-specific views, removing the need for large-scale rToF CT datasets
- **Cardiac cycle generalisation** — reconstructs geometries throughout the cardiac cycle despite training only on diastasis frames
- **Multi-centre validation** — tested on clinical CMR data from multiple centres with real motion artifacts


## Installation Guide

The easiest way to set up this repository is using the provided conda environment (Python 3.11). Follow steps 1–5 below to create and activate the `completeme-311` environment.

### Step 1: Clone this repository
In a Git-enabled terminal, run:

```bash
git clone https://github.com/charlenemauger1/tof_label_completion.git
```
Alternatively, you can use a GUI client such as [GitHub Desktop](https://desktop.github.com/download/) or [GitKraken](https://www.gitkraken.com/) to clone the repository. If prompted to initialise submodules after cloning, **select no** — these will be initialised in Step 4.

Step 2: Set Up the Virtual Environment
If you have [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) or [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) installed, create and activate the conda environment by running the following in your terminal (or Anaconda Command Prompt on Windows):

```bash
cd tof_label_completion
conda create -n completeme-311 python=3.11
conda activate completeme-311
```

### Step 3: Install the complete-me Packages
With the conda environment active, navigate to the cloned repository and install the required packages:

```bash
pip install -e .
```

### Step 4: Download Pretrained Models
All model checkpoints are available on Hugging Face: [charlenemauger1/complete-me](https://huggingface.co/charlenemauger1/complete-me)

To download the pretrained weights:

```python
python download_pretrained_weights.py
```

This script downloads the pretrained model checkpoints from Hugging Face into the exact directory structure expected by the pipeline — segmentation models into `src/completeme/segmentation/checkpoints/` and the label completion network into `src/completeme/label_completion/checkpoints/`.  

### Step 5: Install PyTorch
This project requires PyTorch, which is not included in the default `completeme-311` environment. Visit the [**PyTorch installation page**](https://pytorch.org/get-started/locally/) and follow the instructions for your GPU and operating system.



## How to run the pipeline

### End-to-end pipeline

### Generate segmentations from sparse nifti

`run_batch_segmentation.py` runs the segmentation network on CMR NIfTI images and outputs segmentation masks. It automatically selects the correct model based on the view keyword in the filename (`SA`, `4CH`, `2CH_LT`, `2CH_RT`, `3CH`, `RVOT`).
Each `.nii.gz` is a 2D+t cine volume (x, y, time frames). The short-axis stack has one file per slice position. Input files should be organised as follows:

```bash
input_folder/
    ├── patient_001/
    │   ├── *SA*.nii.gz        # required
    │   ├── *SA*.nii.gz        # optional
    │   ├── *SA*.nii.gz        # optional
    │   ├── [...]              # optional    
    │   ├── *4CH*.nii.gz       # required
    │   ├── *2CH_LT*.nii.gz    # optional
    │   ├── *2CH_RT*.nii.gz    # optional
    │   ├── *3CH*.nii.gz       # optional
    │   └── *RVOT*.nii.gz      # optional
    ├── patient_002/
    │   └── ...
```

To run the segmentation network on the provided example, run the following command
```bash
python ./src/completeme/segmentation/run_batch_segmentation.py -b ./example/nifti/ -o ./tof_mask
```

The resulting segmentations should be identical to the ones in the `./example/segmented-nifti/` and look like this: 

![Segmentation](images/case_1_collage.gif) 

### Generate dense volumes from sparse segmentations

`run-pipeline.py` automates the generation of dense 3D volumes from the sparse segmentation masks created in the previous step. It handles data cleaning, slice alignment (SSA), and 3D interpolation. Input files should be organized as follows:
```
input_folder/
    ├── patient_001/           # contains .nii.gz masks from segmentation step
    │   ├── *SA*.nii.gz        
    │   ├── *4CH*.nii.gz       
    │   └── ...
    ├── patient_002/
    │   └── ...
```

To generate dense volumes from the provided example segmentations, run:

```bash
python ./src/completeme/label_completion/run-pipeline.py --all_components --input_dir ./example/segmented-nifti --output_dir ./output_volume --log
```

The script produces a final 3D reconstruction by interpolating the sparse slices into a continuous volume. The output will be saved in the --output_dir under `3d_dense_volumes/`.


Each part of the label completion can be run separately so if you only need to run specific parts of the pipeline, use these individual flags instead of `--all_components:`

`--preprocessing`: Cleans and converts data to multiframe format. \
`--slice_shifting`: Performs Slice Shift Alignment (SSA).\
`--volume_conversion`: Converts aligned slices into 3D sparse volumes.\
`--sparse_to_dense`: Interpolates sparse data into a final dense 3D volume.

The resulting reconstructions should look like this: 

![Segmentation](images/dense_volume.gif) 

## Contact
For questions or issues, please open an issue on GitHub or contact [charlene.1.mauger@kcl.ac.uk](charlene.1.mauger@kcl.ac.uk) 
