from os.path import dirname
from pathlib import Path
DYNUNET_CHECKPOINT_DIR = Path(dirname(__file__)) / 'segmentation/checkpoints/dynUnet_segmentation'
DYNUNET_N_FOLDS = 5