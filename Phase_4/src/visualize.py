import torch
import cv2
import matplotlib.pyplot as plt
from dataset import ChangeDataset
from model import SiameseCNN
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_sample(index=0):
    dataset = ChangeDataset(DATASET_DIR, "test")
    t1, t2, mask = dataset[index]

    model = SiameseCNN().to(device)
    model.load_state_dict(torch.load("siamese_cnn.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        pred = model(
            t1.unsqueeze(0).to(device),
            t2.unsqueeze(0).to(device)
        )

    pred = pred.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(t1.permute(1, 2, 0))
    plt.title("t1")

    plt.subplot(1, 3, 2)
    plt.imshow(t2.permute(1, 2, 0))
    plt.title("t2")

    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap="gray")
    plt.title("Predicted Change")

    plt.show()


if __name__ == "__main__":
    visualize_sample(0)
