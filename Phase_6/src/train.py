import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ChangeDataset
from siamese_unet import SiameseUNet
from config import *

device = torch.device(DEVICE)

# -------------------------------------------------
# DICE LOSS
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
# COMBINED LOSS
# -------------------------------------------------
bce_loss = nn.BCELoss()
dice_loss = DiceLoss()

def combined_loss(pred, mask):
    return bce_loss(pred, mask) + dice_loss(pred, mask)


# -------------------------------------------------
# TRAINING WITH EARLY STOPPING
# -------------------------------------------------
def train():
    train_ds = ChangeDataset(DATASET_DIR, "train")
    val_ds = ChangeDataset(DATASET_DIR, "val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = SiameseUNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LR_STEP,
        gamma=LR_GAMMA
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        # ---------------- TRAIN ----------------
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

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for t1, t2, mask in val_loader:
                t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
                pred = model(t1, t2)
                val_loss += combined_loss(pred, mask).item()

        val_loss /= len(val_loader)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # ---------------- EARLY STOPPING ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "siamese_unet_ES.pth")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(
                f"\nEarly stopping triggered at epoch {epoch+1}. "
                f"Best Val Loss: {best_val_loss:.4f}"
            )
            break


if __name__ == "__main__":
    train()
