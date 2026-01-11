import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed_dataset"
)

# -----------------------------
# PARAMETERS
# -----------------------------
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4

# -----------------------------
# OPTIMIZATION (PHASE 6)
# -----------------------------
PATIENCE = 7
LR_STEP = 10
LR_GAMMA = 0.5

# -----------------------------
# TASK
# -----------------------------
NUM_CLASSES = 1
LOSS_TYPE = "BCE+DICE"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
