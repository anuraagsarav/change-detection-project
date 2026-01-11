import os
import cv2
import torch
from torch.utils.data import Dataset

class ChangeDataset(Dataset):
    def __init__(self, root_dir, split):
        self.t1_dir = os.path.join(root_dir, split, "t1")
        self.t2_dir = os.path.join(root_dir, split, "t2")
        self.mask_dir = os.path.join(root_dir, split, "mask")

        self.files = sorted(os.listdir(self.t1_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        t1 = cv2.imread(os.path.join(self.t1_dir, name))
        t2 = cv2.imread(os.path.join(self.t2_dir, name))
        mask = cv2.imread(os.path.join(self.mask_dir, name), 0)

        # Normalize
        t1 = torch.tensor(t1 / 255.0, dtype=torch.float32).permute(2, 0, 1)
        t2 = torch.tensor(t2 / 255.0, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask / 255.0, dtype=torch.float32).unsqueeze(0)

        return t1, t2, mask
