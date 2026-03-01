# Project Progress Report: AI-based PCB Inspection

## 1. Project Overview
**Title:** AI-based PCB Inspection: Automated Quality Assurance System for Mission-Critical Defense Electronics

**Abstract:** The project focuses on intelligent, automated inspection of Printed Circuit Boards (PCBs) to reliably identify fine-grained trace defects across multiple categories (shorts, opens, pinholes, mouse bites, spurs, and spurious copper) that are difficult to identify using traditional AOI systems. It adopts an AI-based computer vision methodology, utilizing labeled PCB datasets, image preprocessing pipelines, and deep learning models (CNN classifiers, Object Detectors, and Segmentation Networks) to minimize false negatives and guarantee system reliability in mission-critical applications.

---

## 2. Progress Achieved So Far
Based on the current state of the repository, significant progress has been made regarding data acquisition, pipeline formulation, and dataset preparation:

1. **Dataset Acquisition & Understanding:** 
   - Acquired the **DeepPCB dataset**, consisting of 1,500 image pairs (defect-free template vs. tested image). 
   - Annotated files containing bounding box coordinates and class IDs mapping to the 6 most common defects are fully mapped.

2. **Image Preprocessing Pipeline (`preprocess_pcb.py`):**
   - Implemented a preprocessing script that scans the raw images and applies transformations.
   - Specifically:
     - **Denoising:** Applied Gaussian Blur to remove noise.
     - **Contrast Enhancement:** Implemented CLAHE on the L channel of LAB color space to augment defect visibility against the background.
   - **Crucial Update:** Made sure to remove resizing and ROI extraction from earlier iterations to maintain perfect spatial alignment with segmentation masks. 

3. **Mask Generation (`generate_masks.py`):**
   - Successfully created an annotation-parsing script that converts `.txt` bounding boxes into segmentation masks.
   - Defect areas are rendered as white pixels (255) against a black background (0), effectively creating binary masks mapped exactly to their corresponding `_test.jpg` images.
   - Handled proper directory output generation for `trainval` and `test` splits.

4. **Evaluation Benchmark Scripts (`evaluation/`):**
   - Included robust benchmark evaluation scripts calculating mAP (mean Average Precision) and F-Score with IoU thresholds, crucial for precise and quantitative system evaluation.

## 3. Current Status
The project is currently at the boundary of **Data Preparation and Model Implementation**. The raw data has been systematically curated, pre-processed using computer vision techniques, and the ground-truth segmentation masks have been generated. The dataset is now completely modeled and structured, primed for deep learning injection.

---

## 4. What Is Left?
Moving into the core AI/ML phase, the following tasks remain:

1. **Dataloader Formulation:** Writing dataset classes (PyTorch/TensorFlow) to correctly yield pairs of `(preprocessed_image, segmentation_mask/bounding_boxes)`. 
2. **Deep Learning Model Implementation:** 
   - Setup state-of-the-art vision models. Given the micro-level nature of the defects, Object Detectors (YOLO / Faster R-CNN) and Segmentation Networks (U-Net / Mask R-CNN / YOLOv8-seg) need to be configured. 
3. **Training & Validation:** Creating the training loops, configuring optimizers (e.g., AdamW), and selecting suitable loss functions (e.g., BCE/Dice Loss for segmentation, Focal Loss to handle class imbalance).
4. **Evaluation Integration:** Utilizing the implemented `script.py` during validation to consistently print out precision, recall, and F1-score across epochs.
5. **AOI Edge-Deployment / Continuous Learning:** Setting up a final inference script, evaluating real-time FPS, and formatting the architecture to incorporate new defective samples over time.

---

## 5. Implementation Guide for the Remaining Parts

**Step 1: Build the Dataset Class**
- Create a `dataset.py` file with PyTorch's `Dataset` class to dynamically load images and masks.
- Incorporate data augmentations (e.g., albumentations—rotations, flips, noise) during training to build robustness against varying PCB environments.

**Step 2: Initialize the Deep Learning Architectures**
- **Option A (Segmentation - Precise but Slower):** Implement **U-Net** to classify every pixel (defect vs background) since the geometric nature of the traces requires high boundaries fidelity. 
- **Option B (Detection - Fast and Standard):** Use **YOLO** (e.g. YOLOv8/11) to draw bounding boxes and classify the exact type. This caters to the real-time requirements (AOI edge setups).

**Step 3: Define Loss and Train**
- Create `train.py`.
- **Loss:** Standard cross-entropy might struggle because defects are extremely small compared to the PCB background. Use **Focal loss** or **IoU/Dice loss** to heavily penalize missing tiny defects (prioritize recall).
- Run the code for multiple epochs and save the best performing weights based on validation F1-score.

**Step 4: Execute Benchmarks**
- Create `inference.py`. Run the test split through your trained model to export `.txt` coordinates formatted exactly as `validation/gt.zip` expects (`x1, y1, x2, y2, confidence, type`).
- Run `python evaluation/script.py -s=outputs.zip -g=evaluation/gt.zip` to confirm your F-Score and mAP.

**Step 5: System Integration**
- Create a user pipeline that inputs a scanned PCB image, normalizes it, runs an inference pass through the frozen PyTorch model, and visually draws red bounding boxes over the localized anomalies on the UI.
