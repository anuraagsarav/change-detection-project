import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from dataset import ChangeDataset
from siamese_unet import SiameseUNet
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_sample(sample_index=0, threshold=0.5):
    # Load dataset
    dataset = ChangeDataset(DATASET_DIR, "test")

    # CHANGE IMAGE HERE
    # sample_index = 0, 1, 2, 10, 50 ...
    t1, t2, mask = dataset[sample_index]

    # Load model
    model = SiameseUNet().to(device)
    model.load_state_dict(torch.load("siamese_unet_dice.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        pred = model(
            t1.unsqueeze(0).to(device),
            t2.unsqueeze(0).to(device)
        )

    pred = pred.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()

    # Binary prediction
    pred_bin = (pred > threshold).astype(np.uint8)

    # Convert images for display
    t1_img = t1.permute(1, 2, 0).numpy()
    t2_img = t2.permute(1, 2, 0).numpy()

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.imshow(t1_img)
    plt.title("Before (t1)")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(t2_img)
    plt.title("After (t2)")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(pred, cmap="gray")
    plt.title("Predicted (Probability)")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(pred_bin, cmap="gray")
    plt.title("Predicted (Binary)")
    plt.axis("off")

    # Overlay
    overlay = t1_img.copy()
    overlay[pred_bin == 1] = [1, 0, 0]

    plt.subplot(2, 3, 6)
    plt.imshow(overlay)
    plt.title("Overlay on t1")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_sample(sample_index=3)
