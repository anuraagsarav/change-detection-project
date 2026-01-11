import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed_dataset"
)

IMG_SIZE = 256          # Input image size (256x256)
BATCH_SIZE = 4          # Small batch to fit GPU / CPU
EPOCHS = 20             # More epochs than Phase 4
LR = 1e-4               # Lower learning rate for stable training

NUM_CLASSES = 1         # Binary change detection
LOSS_TYPE = "BCE"       # Binary Cross Entropy

DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
