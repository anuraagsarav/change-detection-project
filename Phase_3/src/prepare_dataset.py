import os
import cv2
from tqdm import tqdm
from config import *
from dataset_utils import *
from patch_extraction import *

def prepare_split(split):
    path_A = os.path.join(RAW_DATA_DIR, split, "A")
    path_B = os.path.join(RAW_DATA_DIR, split, "B")
    path_M = os.path.join(RAW_DATA_DIR, split, "label")

    filenames = os.listdir(path_A)
    idx = 0

    for fname in tqdm(filenames, desc=f"Preparing {split}"):
        img1, img2, mask = load_triplet(path_A, path_B, path_M, fname)
        img1, img2, mask = resize_triplet(
            img1, img2, mask, (PATCH_SIZE, PATCH_SIZE)
        )

        patches = extract_patches(img1, img2, mask, PATCH_SIZE, STRIDE)

        for p1, p2, pm in patches:
            if is_valid_patch(pm, MIN_CHANGE_RATIO):
                out_dir = os.path.join(OUTPUT_DATASET_DIR, split)
                os.makedirs(os.path.join(out_dir, "t1"), exist_ok=True)
                os.makedirs(os.path.join(out_dir, "t2"), exist_ok=True)
                os.makedirs(os.path.join(out_dir, "mask"), exist_ok=True)

                cv2.imwrite(os.path.join(out_dir, "t1", f"{idx}.png"), p1)
                cv2.imwrite(os.path.join(out_dir, "t2", f"{idx}.png"), p2)
                cv2.imwrite(os.path.join(out_dir, "mask", f"{idx}.png"), pm)
                idx += 1

    print(f"{split}: {idx} patches saved")


if __name__ == "__main__":
    prepare_split("train")
    prepare_split("val")
    prepare_split("test")
