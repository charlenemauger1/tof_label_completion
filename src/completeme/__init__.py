from os.path import dirname
from pathlib import Path
DYNUNET_CHECKPOINT_DIR = Path(dirname(__file__)) / 'segmentation/checkpoints/dynUnet_segmentation'
LABEL_COMPLETION_CHECKPOINT_DIR = Path(dirname(__file__)) / 'label_completion/checkpoints/label_completion/checkpoints.pth'
DYNUNET_N_FOLDS = 5