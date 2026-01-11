import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_DIR = os.path.dirname(BASE_DIR)                
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

def load_image(image_path):

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    return image

def preprocess_image(image, size=(256, 256)):

    image_resized = cv2.resize(image, size)
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    return gray_image

def compute_difference(image1, image2):

    diff = cv2.absdiff(image1, image2)

    return diff

def threshold_difference(diff_image, threshold_value=30):

    _, change_mask = cv2.threshold(
        diff_image,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    return change_mask

def create_overlay(original_image, change_mask, change_percentage):

    overlay = original_image.copy()
    overlay[change_mask == 255] = [0, 0, 255]  

    text = f"Change Area: {change_percentage:.2f}%"
    cv2.putText(
        overlay,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return overlay

def save_output_images(diff, mask, overlay):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "diff.png"), diff)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "change_mask.png"), mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "overlay.png"), overlay)


def calculate_change_percentage(change_mask):

    total_pixels = change_mask.size
    changed_pixels = np.sum(change_mask == 255)
    change_percentage = (changed_pixels / total_pixels) * 100

    return change_percentage


def display_results(before, after, diff, mask, overlay):

    titles = ["Before Image", "After Image", "Difference", "Change Mask", "Overlay"]
    images = [before, after, diff, mask, overlay]

    plt.figure(figsize=(15, 8))
    for i in range(len(images)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():

    before_path = os.path.join(DATA_DIR, "before.png")
    after_path = os.path.join(DATA_DIR, "after.png")

    before = load_image(before_path)
    after = load_image(after_path)

    before_gray = preprocess_image(before)
    after_gray = preprocess_image(after)

    diff = compute_difference(before_gray, after_gray)
    change_mask = threshold_difference(diff)
    before_resized = cv2.resize(before, (256, 256))

    percentage = calculate_change_percentage(change_mask)
    print(f"Change Percentage: {percentage:.2f}%")
    overlay = create_overlay(before_resized, change_mask, percentage)

    save_output_images(diff, change_mask, overlay)
    display_results(before_gray, after_gray, diff, change_mask, overlay)

if __name__ == "__main__":
    main()