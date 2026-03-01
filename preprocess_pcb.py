# preprocess_pcb.py
"""
Preprocessing pipeline for PCB trace defect detection.

This script reads PCB images, applies a series of preprocessing steps, and saves the
processed images to a designated output directory while preserving the original
directory structure.

Usage:
    python preprocess_pcb.py --input_dir <path_to_raw_images> \
                              --output_dir <path_to_processed_images> \
                              [--visualize]

Options:
    --visualize   Show a few sample images with the applied transformations.
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path

# Optional: albumentations for augmentation (installed via pip if needed)
try:
    import albumentations as A
except ImportError:
    A = None


def parse_args():
    parser = argparse.ArgumentParser(description="PCB image preprocessing pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory containing raw PCB images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store processed images")
    parser.add_argument("--visualize", action="store_true", help="Show sample processed images")
    return parser.parse_args()


def get_image_paths(root_dir: str):
    """Recursively collect image file paths (common extensions)."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if Path(f).suffix.lower() in exts:
                paths.append(os.path.join(dirpath, f))
    return paths


def resize_image(img, size=(512, 512)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def normalize_image(img):
    # Convert to float32 and scale to [0,1]
    return img.astype(np.float32) / 255.0


def denoise_image(img):
    # Gaussian blur with kernel size 3x3
    return cv2.GaussianBlur(img, (3, 3), 0)


def enhance_contrast(img):
    # Apply CLAHE on the L channel of LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced


def extract_roi(img):
    # Simple contour based ROI extraction: assume the largest contour is the board
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img  # fallback
    # Choose largest contour by area
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    roi = img[y : y + h, x : x + w]
    # Resize ROI back to target size
    return resize_image(roi)


def augment_image(img, mask=None):
    if A is None:
        return img, mask
    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
    ])
    if mask is None:
        augmented = aug(image=img)
        return augmented["image"], None
    else:
        augmented = aug(image=img, mask=mask)
        return augmented["image"], augmented["mask"]


def process_and_save(image_path, input_root, output_root, visualize=False):
    rel_path = os.path.relpath(image_path, input_root)
    out_path = os.path.join(output_root, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: could not read {image_path}")
        return

    # Pipeline
    # Removed resize and extract_roi to ensure images perfectly map to segmentation masks spatially.
    # We will do normalization directly in the model dataloader instead.
    img = denoise_image(img)
    img = enhance_contrast(img)

    # Save image
    cv2.imwrite(out_path, img)

    if visualize:
        cv2.imshow("Processed", save_img)
        cv2.waitKey(1)


def main():
    args = parse_args()
    image_paths = get_image_paths(args.input_dir)
    print(f"Found {len(image_paths)} images to process.")
    for idx, img_path in enumerate(image_paths, 1):
        process_and_save(img_path, args.input_dir, args.output_dir, visualize=args.visualize)
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(image_paths)} images")
    if args.visualize:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
