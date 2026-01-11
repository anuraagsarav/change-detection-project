import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATASET_DIR = os.path.join(PROJECT_DIR, "data", "processed_dataset")

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
