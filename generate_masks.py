# generate_masks.py
"""
Segmentation mask generation for PCB trace defect detection.

This script parses bounding box annotations and creates segmentation masks.
The annotations are formatted as: xmin ymin xmax ymax class_id

Usage:
    python generate_masks.py --input_dir PCBData \
                             --output_dir masks_PCBData \
                             [--visualize]

Options:
    --visualize   Show sample generated masks.
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Generate segmentation masks from annotations")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory containing raw PCB images and annotations")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store generated masks")
    parser.add_argument("--visualize", action="store_true", help="Show sample generated masks overlaid on images")
    return parser.parse_args()

def parse_annotation(txt_path):
    """Parses a text file with lines: x1 y1 x2 y2 class_id."""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                x1, y1, x2, y2, class_id = map(int, parts[:5])
                boxes.append((x1, y1, x2, y2, class_id))
    return boxes

def generate_mask_for_image(img_path, txt_path, output_mask_path, visualize=False):
    if not os.path.exists(img_path):
        print(f"Skipping: Image file not found -> {img_path}")
        return
        
    # Read original image to get dimensions
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: cv2.imread failed for {img_path}")
        return

    h, w = img.shape[:2]
    
    # Create an empty mask (single channel)
    # Background is 0. Defect classes will be 1, 2, 3, etc.
    # Alternatively for just binary "defect or not", everything is 255.
    # Let's create a binary mask here where defect = 255.
    mask = np.zeros((h, w), dtype=np.uint8)
    
    boxes = parse_annotation(txt_path)
    for (x1, y1, x2, y2, cls_id) in boxes:
        # Draw filled rectangle for the mask
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
    # Ensure output directory exists for this mask
    output_dir = os.path.dirname(output_mask_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save mask; returns True if successful
    success = cv2.imwrite(output_mask_path, mask)
    if not success:
        print(f"Error: failed to write mask to {output_mask_path}")
    
    if visualize:
        # Create an overlay for visualization
        overlay = img.copy()
        # Create a red and transparent mask
        red_mask = np.zeros_like(img)
        red_mask[:, :, 2] = mask  # Red channel
        
        cv2.addWeighted(red_mask, 0.5, overlay, 0.5, 0, overlay)
        for (x1, y1, x2, y2, cls_id) in boxes:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(overlay, str(cls_id), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Mask Overlay", overlay)
        cv2.waitKey(1)

def main():
    args = parse_args()
    
    # Find all test.txt and trainval.txt to know the splits
    splits = ['trainval.txt', 'test.txt']
    
    processed_count = 0
    
    for split in splits:
        split_path = os.path.join(args.input_dir, split)
        if not os.path.exists(split_path):
            continue
            
        with open(split_path, 'r') as f:
            lines = f.readlines()
            
        for idx, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) >= 2:
                base_img_rel = parts[0]
                txt_rel = parts[1]
                
                # The actual image paths end in _test.jpg and _temp.jpg. We want the test image.
                img_rel = base_img_rel.replace('.jpg', '_test.jpg')
                
                img_path = os.path.join(args.input_dir, img_rel)
                txt_path = os.path.join(args.input_dir, txt_rel)
                
                # Derive output mask path based on the generic base name
                img_rel_normalized = base_img_rel.replace('\\', '/')
                mask_rel = os.path.splitext(img_rel_normalized)[0] + ".png"
                output_mask_path = os.path.join(args.output_dir, os.path.normpath(mask_rel))
                
                # Make sure the base output directory for this split exists
                os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
                
                generate_mask_for_image(img_path, txt_path, output_mask_path, visualize=args.visualize)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Generated offset path: {os.path.abspath(output_mask_path)}")
                    
    print(f"Finished generating {processed_count} masks.")
    
    if args.visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
