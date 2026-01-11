import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# DIRECTORY SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# LOAD IMAGE
# -----------------------------
def load_image(filename):
    path = os.path.join(DATA_DIR, filename)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load {filename}")
    return img


# -----------------------------
# BASIC PREPROCESSING
# -----------------------------
def preprocess_basic(image, size=(256, 256)):
    image = cv2.resize(image, size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


# -----------------------------
# NORMALIZATION
# -----------------------------
def normalize_image(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


# -----------------------------
# HISTOGRAM MATCHING (CLAHE)
# -----------------------------
def histogram_equalization(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


# -----------------------------
# DENOISING
# -----------------------------
def denoise_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


# -----------------------------
# ECC ALIGNMENT
# -----------------------------
def align_images(reference, target):
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        2000,
        1e-5
    )

    cv2.findTransformECC(
        reference,
        target,
        warp_matrix,
        cv2.MOTION_AFFINE,
        criteria
    )

    aligned = cv2.warpAffine(
        target,
        warp_matrix,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )

    return aligned


# -----------------------------
# CREATE VALID PIXEL MASK
# -----------------------------
def create_valid_mask(img1, img2):
    mask = np.logical_and(img1 > 0, img2 > 0)
    return mask.astype(np.uint8) * 255


# -----------------------------
# CHANGE DETECTION
# -----------------------------
def detect_changes(before, after, valid_mask, threshold=30):
    diff = cv2.absdiff(before, after)
    diff_masked = cv2.bitwise_and(diff, diff, mask=valid_mask)
    _, change_mask = cv2.threshold(
        diff_masked, threshold, 255, cv2.THRESH_BINARY
    )
    return diff_masked, change_mask


# -----------------------------
# CHANGE PERCENTAGE
# -----------------------------
def calculate_change_percentage(change_mask, valid_mask):
    changed_pixels = np.sum(change_mask == 255)
    valid_pixels = np.sum(valid_mask == 255)

    if valid_pixels == 0:
        return 0.0

    return (changed_pixels / valid_pixels) * 100


# -----------------------------
# OVERLAY CHANGES + TEXT
# -----------------------------
def overlay_changes(original_gray, change_mask, change_percentage):
    overlay = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    overlay[change_mask == 255] = [0, 0, 255]  # Red

    text = f"Change: {change_percentage:.2f} %"
    cv2.putText(
        overlay,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    return overlay


# -----------------------------
# SAVE IMAGE
# -----------------------------
def save_image(folder, name, image):
    path = os.path.join(OUTPUT_DIR, folder)
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(os.path.join(path, name), image)


# -----------------------------
# DISPLAY
# -----------------------------
def show_results(images, titles):
    plt.figure(figsize=(15, 6))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    before = load_image("before.png")
    after = load_image("after.png")

    before_gray = preprocess_basic(before)
    after_gray = preprocess_basic(after)

    before_norm = normalize_image(before_gray)
    after_norm = normalize_image(after_gray)

    after_hist = histogram_equalization(after_norm)

    before_denoised = denoise_image(before_norm)
    after_denoised = denoise_image(after_hist)

    after_aligned = align_images(before_denoised, after_denoised)

    valid_mask = create_valid_mask(before_denoised, after_aligned)

    diff, change_mask = detect_changes(
        before_denoised, after_aligned, valid_mask
    )

    change_percentage = calculate_change_percentage(
        change_mask, valid_mask
    )

    print(f"Detected Change Percentage: {change_percentage:.2f} %")

    overlay = overlay_changes(
        before_denoised, change_mask, change_percentage
    )

    save_image("change_detection", "diff.png", diff)
    save_image("change_detection", "valid_mask.png", valid_mask)
    save_image("change_detection", "change_mask.png", change_mask)
    save_image("change_detection", "overlay.png", overlay)

    show_results(
        [before_denoised, after_aligned, change_mask, overlay],
        ["Before", "After (Aligned)", "Change Mask", "Overlay with %"]
    )


if __name__ == "__main__":
    main()
