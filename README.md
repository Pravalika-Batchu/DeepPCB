# AI-based PCB Inspection: Automated Quality Assurance System for Mission-Critical Defense Electronics

**Team:**
- 1602-23-733-109 – T. Sai Vikhil Reddy
- 1602-23-733-118 – M. Sneha Reddy
- 1602-23-733-308 – B. Pravalika
**Under Guidance:** Dr. T. Adilakshmi, Professor & HOD, Dept of CSE

---

## Abstract
Printed Circuit Boards (PCBs) are the backbone of modern electronic and defense systems, where even microscopic defects can lead to intermittent connectivity, degraded performance, or catastrophic system failure. Traditional inspection methods such as Automated Optical Inspection (AOI) and X-ray imaging are widely used to detect assembly defects, but with the increasing miniaturization and density of PCBs—especially in mission-critical applications—there is a growing need for intelligent, automated inspection systems capable of reliably identifying fine-grained defects across multiple defect categories.

In this project, the primary focus is on **trace defects** (shorts, opens, pinholes, and mouse bites), which directly affect the copper conductive paths responsible for electrical continuity and signal integrity. The work addresses the challenge of accurately detecting and localizing micro-level trace anomalies while minimizing false negatives that could compromise system reliability.

To solve this problem, the project adopts an AI-based computer vision methodology using labeled PCB datasets containing normal and defective trace samples. The pipeline includes image preprocessing techniques, mask generation, and deep learning models (CNN classifiers, Object Detectors, and Segmentation Networks) designed for integration into AOI or edge-based inspection setups with continuous learning capability.

---

## 🚀 Current Progress & Completed Phases

**1. Dataset Setup:** 
Incorporating the DeepPCB dataset containing 1,500 image pairs of normal and defective traces mapped to 6 common PCB defect classes.

**2. Image Preprocessing Pipeline (`preprocess_pcb.py`):**
- Created scripts for automated image enhancements.
- **Denoising:** Applied Gaussian Blur.
- **Contrast Enhancement:** Used CLAHE (Contrast Limited Adaptive Histogram Equalization) on the LAB color space to isolate complex and tiny features from background illumination.

**3. Dataset Mask Formulation (`generate_masks.py`):**
- Generated an annotation parser that translates bounding box coordinate values into binary segmentation masks mapping spatial defects accurately for spatial-heavy algorithms like U-Net. 

**4. Evaluation Ecosystem (`evaluation/`):**
- Setup base evaluation functions supporting mAP (mean Average Precision) and customized F-score metrics across intersection-over-union (IoU) thresholds.

---

## 🛠️ Upcoming Enhancements (To-Do)

**1. PyTorch / TensorFlow Dataloaders:**
Develop model-agnostic dataset classes to batch preprocessed images along with their respective masks or bounding boxes into tensors optimized for memory bandwidth over GPUs.

**2. Deep Learning Modeling Phase:**
Implementation of the core AI models tailored for the dataset:
- **Segmentation Strategy:** Building models like **U-Net** and **Mask R-CNN** that assign pixel-wise classes (detecting exact bounding borders around traces).
- **Detection Strategy:** Building optimized detectors like **YOLO** families to handle low latency predictions suitable for real-time Automated Optical Inspections.

**3. Pipeline Training & Convergence Logs:**
Creating training scripts handling optimizer setups (Adam/AdamW) and defining specialized loss functions suitable for highly imbalanced object detection configurations (using Dice Loss for semantic segmentation or Focal Loss to amplify hard-to-detect microscopic targets).

**4. Model Inference & User Reporting System:**
A visualization loop feeding real-world incoming PCB components into the inferencer to highlight errors interactively for end users or manufacturing automated sorting channels.

---

## About The DeepPCB Dataset (Original Source Context)
DeepPCB is a dataset containing 1,500 image pairs, each of which consists of a defect-free template image and an aligned tested image with annotations including positions of 6 most common types of PCB defects: open, short, mousebite, spur, pin hole and spurious copper.

### Image Annotation
Each annotated image owns an annotation file with the same filename. Each defect on the tested image is annotated as the format `x1, y1, x2, y2, type`, where `(x1,y1)` and `(x2,y2)` are the top-left and bottom-right corners of the bounding box. `type` is an integer ID: **1-open, 2-short, 3-mousebite, 4-spur, 5-copper, 6-pin-hole**.

### Benchmarks Available
The average precision rate and F-score are used for evaluation. A detection is considered correct if the Intersection over Union (IoU) between the detected box and ground truth is > 0.33.

*The evaluation scripts in `evaluation/` can be run against predicted boundary coordinates.*
