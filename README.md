# Filling the Gaps: Generating 4D Dense Cardiac Anatomy from Sparse CMR for Enhanced Tetralogy of Fallot Assessment

A two-stage deep learning pipeline for reconstructing dense 3D whole-heart segmentations from sparse 2D cine CMR images in repaired Tetralogy of Fallot (rToF) patients.


## Table of Contents
- [**Pipeline**](#pipeline-overview)
- [**Installation**](#installation-guide)
- [**Generating dense segmentation from sparse nifti**](#how-to-run-complete-me)
    - [Example usage](#example-usage)
    - [Generate segmentation mask from sparse nifti](#generate-segmentation-from-sparse-nifti)
    - [Generate dense segmentation from sparse segmentation mask](#generate-dense-segmentation-from-sparse-segmentation-mask)
- [**Contact**](#contact) 


For a detailed description of this pipeline, please refer to:

    Mauger et al.

For a detailed description regarding the label completion, please refer to:

    Yiyang et al.

Depending on how you use this repo, please cite the relevant publication(s) above.


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

The repository contains two folders:
- `dynUnet_segmentation/` — 5-fold ensemble checkpoints for each CMR view
- `label_completion/` — label completion network checkpoint

To download the pretrained weights:

```python
python download_pretrained_weights.py
```

### Step 5: Install PyTorch
This project requires PyTorch, which is not included in the default `completeme-311` environment. Visit the [**iPyTorch installation page**](https://pytorch.org/get-started/locally/) and follow the instructions for your GPU and operating system.




## Contribution - Notation

If you wish to contribute to this project, please follow the naming conventions outlined below:

| **Category**         | **Naming Convention**                                               | **Example**                                               |
|----------------------|---------------------------------------------------------------------|-----------------------------------------------------------|
| **Variable**         | Lowercase letters, words separated by underscores (snake_case)      | `site_name` instead of `sitename`                         |
| **Function/Method**  | Lowercase letters, words separated by underscores (snake_case)      | `def my_function()` instead of `def MyFunction()`          |
| **Constant**         | Uppercase letters, words separated by underscores                   | `MY_CONSTANT = 3.1416` instead of `MYCONSTANT = 3.1416`    |
| **Class**            | CamelCase                                                          | `class MyClass:` instead of `class myclass:`               |
| **Package/Module**   | No underscores or hyphens, consistent with Python standard library | `mypackage` instead of `my_package_name_with_underscores`  |
| **Type Variable**    | CamelCase with a leading capital letter                             | `Dict[int, str]` instead of `dict[int, str]`               |
| **Exception**        | Ends with “Error” suffix                                           | `class MyCustomExceptionError:` instead of `class MyCustomException:` |
| **Characters**       | Stick to ASCII characters                                          | `count = 42` instead of `ç = 42`                           |
| **Type Hints**       | Always use type hints for code readability                          | `def greet(name: str) -> str:` instead of `def greet(name):` |

## Contact
For questions or issues, please open an issue on GitHub or contact [charlene.1.mauger@kcl.ac.uk](charlene.1.mauger@kcl.ac.uk) 
