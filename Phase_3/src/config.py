import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

RAW_DATA_DIR = os.path.join(PROJECT_DIR, "data", "LEVIR_CD_RAW")
OUTPUT_DATASET_DIR = os.path.join(PROJECT_DIR, "data", "processed_dataset")

PATCH_SIZE = 256
STRIDE = 256
MIN_CHANGE_RATIO = 0.01
