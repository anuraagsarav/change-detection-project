import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed_dataset"
)

# -------------------------
# DATA PARAMETERS
# -------------------------
IMG_SIZE = 256
BATCH_SIZE = 4

# -------------------------
# TRAINING PARAMETERS
# -------------------------
EPOCHS = 50
LR = 1e-4

# -------------------------
# OPTIMIZATION (PHASE 6)
# -------------------------
PATIENCE = 7          # Early stopping patience
LR_STEP = 10          # StepLR step size
LR_GAMMA = 0.5        # Learning rate decay factor

# -------------------------
# TASK PARAMETERS
# -------------------------
NUM_CLASSES = 1       # Binary change detection
LOSS_TYPE = "BCE+DICE"

DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
