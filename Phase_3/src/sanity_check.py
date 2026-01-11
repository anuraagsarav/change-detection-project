import cv2
import matplotlib.pyplot as plt
import os
from config import OUTPUT_DATASET_DIR

sample_id = "2"

t1 = cv2.imread(os.path.join(OUTPUT_DATASET_DIR, "train", "t1", f"{sample_id}.png"))
t2 = cv2.imread(os.path.join(OUTPUT_DATASET_DIR, "train", "t2", f"{sample_id}.png"))
mask = cv2.imread(os.path.join(OUTPUT_DATASET_DIR, "train", "mask", f"{sample_id}.png"), 0)

plt.figure(figsize=(10,4))
plt.subplot(1,3,1); plt.imshow(t1); plt.title("t1")
plt.subplot(1,3,2); plt.imshow(t2); plt.title("t2")
plt.subplot(1,3,3); plt.imshow(mask, cmap="gray"); plt.title("mask")
plt.show()
