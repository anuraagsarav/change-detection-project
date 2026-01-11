import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset import ChangeDataset
from model import SiameseCNN
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    train_ds = ChangeDataset(DATASET_DIR, "train")
    val_ds = ChangeDataset(DATASET_DIR, "val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = SiameseCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for t1, t2, mask in train_loader:
            t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)

            pred = model(t1, t2)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "siamese_cnn.pth")


if __name__ == "__main__":
    train()
