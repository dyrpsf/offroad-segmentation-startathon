# Offroad Semantic Scene Segmentation – Startathon 2026

Semantic segmentation model for off‑road desert environments using Duality AI’s synthetic “digital twin” dataset.  
The model classifies every pixel into 10 semantic classes and is designed to be fast enough for near real‑time use.

---

## 1. Project Overview

**Goal**

Train and evaluate a semantic segmentation model that labels each pixel of an RGB image as one of:

1. Trees  
2. Lush Bushes  
3. Dry Grass  
4. Dry Bushes  
5. Ground Clutter  
6. Flowers  
7. Logs  
8. Rocks  
9. Landscape (general ground)  
10. Sky  

**Approach**

- Backbone: **DeepLabV3+** with **ResNet‑50** (torchvision)
- Initialization: COCO pretrained weights
- Loss: **Focal Loss** on top of Cross‑Entropy to handle class imbalance
- Data Augmentation: heavy spatial and photometric augmentations (Albumentations)
- Training Resolution: **384 × 384**
- Metrics: mean Intersection‑over‑Union (**mIoU**), per‑class IoU, confusion matrix, and average inference time

Final quantitative results (mIoU, per‑class IoU, confusion matrix) are documented in `Hackathon_Report.txt`.

---

## 2. Repository Structure

```text
.
├── train_offroad.py          # Training script (DeepLabV3+)
├── test_offroad.py           # Evaluation + visualization + confusion matrix
├── best_model.pth            # Trained model weights (best val mIoU)
├── README.md                 # This file
├── README.txt                # Plain-text variant
├── Hackathon_Report.txt      # Detailed methodology & results for submission
├── inspect_mask.py (optional)# Helper to inspect raw label IDs
└── outputs/                  # Created by test_offroad.py (overlays, etc.) [not required]

```

The dataset itself is not included in this repo/zip and must be downloaded from the official hackathon links.

3. Dataset Layout & Label Mapping
The project assumes the default Duality AI folder layout (adapt paths if needed):

```text
Offroad_Segmentation_Training_Dataset/
└── Offroad_Segmentation_Training_Dataset/
    ├── train/
    │   ├── Color_Images/        # RGB images
    │   └── Segmentation/        # uint16 label masks
    └── val/
        ├── Color_Images/
        └── Segmentation/

Offroad_Segmentation_testImages/
└── Offroad_Segmentation_testImages/
    ├── Color_Images/            # Test RGB images
    └── Segmentation/            # Test masks (only for evaluation/visualization)

```
3.1 Raw Label IDs
Masks are 16‑bit single‑channel PNGs. Each pixel’s value is one of:

   Raw ID	Class Name
   100	Trees
   200	Lush Bushes
   300	Dry Grass
   500	Dry Bushes
   550	Ground Clutter
   600	Flowers
   700	Logs
   800	Rocks
   7100	Landscape
   10000	Sky

3.2 Training Label Mapping
For training and evaluation, these IDs are mapped to contiguous indices:

   Train ID	Raw ID	Class Name
   0	100	Trees
   1	200	Lush Bushes
   2	300	Dry Grass
   3	500	Dry Bushes
   4	550	Ground Clutter
   5	600	Flowers
   6	700	Logs
   7	800	Rocks
   8	7100	Landscape
   9	10000	Sky

Any pixel with a value outside this list is set to label 255 and treated as ignore_index in the loss and metrics.

The mapping is implemented in both train_offroad.py and test_offroad.py.

4. Environment & Installation
Tested on:

   Windows 11
   Python 3.x (Anaconda base environment)
   NVIDIA GPU with CUDA support

4.1 Core Dependencies
Install via pip (preferably from Anaconda Prompt):
```Bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install albumentations==1.4.3 opencv-python pillow numpy matplotlib tqdm tensorboard
```

Verify PyTorch and GPU:

```Bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Output should show CUDA available (... True) for GPU acceleration; CPU also works but training will be much slower.

5. Configuration
Both scripts use absolute Windows paths by default, for example:

```Python
TRAIN_IMG_DIR = r"E:\Startathon\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Color_Images"
TRAIN_MASK_DIR = r"E:\Startathon\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Segmentation"
VAL_IMG_DIR   = r"E:\Startathon\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val\Color_Images"
VAL_MASK_DIR  = r"E:\Startathon\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val\Segmentation"

TEST_IMG_DIR  = r"E:\Startathon\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Color_Images"
TEST_MASK_DIR = r"E:\Startathon\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Segmentation"
```

If your dataset is in a different location, update these constants at the top of train_offroad.py and test_offroad.py.

Other important hyperparameters in train_offroad.py:

```Python
IMAGE_SIZE   = (384, 384)
BATCH_SIZE   = 4
NUM_EPOCHS   = 2        # kept low due to hackathon time limit
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
```

You can adjust these for longer / higher‑quality training if you have more time and compute.

6. Training
From the project root:

```Bash
python train_offroad.py
```

What this script does:

1. Loads train/val images and masks.
2. Applies Albumentations augmentations on the training set:
   RandomResizedCrop(384×384, scale=(0.7, 1.0))
   Horizontal flip
   Random brightness/contrast
   Hue/saturation shift
   Gaussian noise
   Motion blur
   Normalization with ImageNet mean/std

3. Builds DeepLabV3+ (ResNet‑50) with 10‑class head.
4. Uses Focal Loss (with ignore_index=255) and AdamW optimizer.
5. Trains for the configured number of epochs.
6. After each epoch:

   Evaluates on the validation set
   Computes total loss, mean IoU and per‑class IoU

7. Saves the best checkpoint as:

```text
best_model.pth
```
based on highest validation mean IoU.

If GPU memory is limited, you can further reduce:

   IMAGE_SIZE (e.g., to (320, 320))
   BATCH_SIZE (e.g., 2)

7. Evaluation & Visualization

Use test_offroad.py to evaluate the trained model and generate visualizations.

7.1 Validation Performance (IoU + Confusion Matrix)

```Bash
python test_offroad.py --split val
```
This script:

1. Loads best_model.pth.
2. Runs inference on all validation images.
3. Computes:
   Mean IoU across all 10 classes
   Per‑class IoU
   A 10×10 confusion matrix (rows = ground truth, cols = predicted)
   Average inference time per image

Saves up to 30 visualizations (per split) into:

```text
outputs/val/
    000012_overlay.png   # RGB image + color mask overlay
    000012_mask.png      # Colorized prediction only
    ...
```

The console output includes the IoU table, confusion matrix, and average runtime. Final numbers used in the hackathon submission are recorded in Hackathon_Report.txt.

7.2 Unseen Test Environment (Qualitative Generalization)

To test on the separate “test” biome (no training on these images):

```Bash
python test_offroad.py --split test
```
This:

   Uses the same best_model.pth
   Runs inference on Offroad_Segmentation_testImages/...
   Saves overlays into:

```text
outputs/test/
```

This is useful for qualitative evaluation and demonstrating generalization; IoU can be computed if ground‑truth masks are available.

mportant: Test images are never used for training. They are only used for evaluation and visualizations.

8. Acknowledgements
Dataset, baseline scripts, and problem statement provided by Duality AI and Startathon Hackathon 2026 organizers.
Model implemented using PyTorch and torchvision.
Augmentations by Albumentations.
<br>
Authors: Deepak Yadav, Ashutosh Pavaiya, Soumya Gupta, Aashka Saiwal