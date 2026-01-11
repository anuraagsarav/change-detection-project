import numpy as np

def extract_patches(img1, img2, mask, patch_size, stride):
    patches = []
    h, w = img1.shape[:2]

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            p1 = img1[y:y+patch_size, x:x+patch_size]
            p2 = img2[y:y+patch_size, x:x+patch_size]
            pm = mask[y:y+patch_size, x:x+patch_size]
            patches.append((p1, p2, pm))

    return patches


def is_valid_patch(mask, min_change_ratio):
    changed = np.sum(mask == 255)
    total = mask.size
    return (changed / total) >= min_change_ratio
