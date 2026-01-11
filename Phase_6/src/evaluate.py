import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import ChangeDataset
from siamese_unet import SiameseUNet
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_iou(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    if union == 0:
        return 1.0

    return intersection / union


def evaluate(threshold=0.5):
    dataset = ChangeDataset(DATASET_DIR, "test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = SiameseUNet().to(device)
    model.load_state_dict(torch.load("siamese_unet_LR.pth", map_location=device))
    model.eval()

    iou_scores = []

    with torch.no_grad():
        for t1, t2, mask in loader:
            t1, t2 = t1.to(device), t2.to(device)

            pred = model(t1, t2)
            pred = pred.squeeze().cpu().numpy()
            mask = mask.squeeze().numpy()

            pred_bin = (pred > threshold).astype(np.uint8)
            mask_bin = (mask > 0.5).astype(np.uint8)

            iou = compute_iou(pred_bin, mask_bin)
            iou_scores.append(iou)

    mean_iou = np.mean(iou_scores)
    print(f"Mean IoU on Test Set: {mean_iou:.4f}")


if __name__ == "__main__":
    evaluate()
