import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ChangeDataset
from siamese_unet import SiameseUNet
from config import *

device = torch.device(DEVICE)

# -------------------------------------------------
# DICE LOSS (NEW — STEP 2)
# -------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


# -------------------------------------------------
# COMBINED BCE + DICE LOSS
# -------------------------------------------------
bce_loss = nn.BCELoss()
dice_loss = DiceLoss()

def combined_loss(pred, mask):
    return bce_loss(pred, mask) + dice_loss(pred, mask)


# -------------------------------------------------
# TRAINING FUNCTION
# -------------------------------------------------
def train():
    # Dataset
    train_ds = ChangeDataset(DATASET_DIR, "train")
    val_ds = ChangeDataset(DATASET_DIR, "val")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE
    )

    # Model
    model = SiameseUNet().to(device)

    # Optimizer (unchanged from Step-1)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # -------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for t1, t2, mask in train_loader:
            t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)

            pred = model(t1, t2)
            loss = combined_loss(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for t1, t2, mask in val_loader:
                t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
                pred = model(t1, t2)
                val_loss += combined_loss(pred, mask).item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    # Save model (temporary checkpoint)
    torch.save(model.state_dict(), "siamese_unet_step2.pth")


if __name__ == "__main__":
    train()