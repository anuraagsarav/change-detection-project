import cv2
import os

def load_triplet(path_A, path_B, path_M, filename):
    img1 = cv2.imread(os.path.join(path_A, filename))
    img2 = cv2.imread(os.path.join(path_B, filename))
    mask = cv2.imread(os.path.join(path_M, filename), cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None or mask is None:
        raise ValueError(f"Missing data for {filename}")

    return img1, img2, mask


def resize_triplet(img1, img2, mask, size):
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return img1, img2, mask
