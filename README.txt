Project: Offroad Semantic Scene Segmentation – Startathon 2026

Goal:
Train a semantic segmentation model on Duality AI’s synthetic off-road desert dataset
to classify each pixel into 10 classes: Trees, Lush Bushes, Dry Grass, Dry Bushes,
Ground Clutter, Flowers, Logs, Rocks, Landscape, and Sky.

Environment:
- Python 3.x (Anaconda base)
- PyTorch 2.7.1+cu118
- torchvision
- albumentations==1.4.3
- opencv-python
- numpy, pillow, matplotlib, tqdm, tensorboard

Dataset layout (as provided):
- Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\
    - train\Color_Images
    - train\Segmentation
    - val\Color_Images
    - val\Segmentation
- Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\
    - Color_Images
    - Segmentation  (only used for evaluation/visualization, never for training)

Raw annotation IDs in masks:
[100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
These are mapped to contiguous class indices [0..9] for training. Any other pixels
are ignored (label 255).

How to train:
1. Install dependencies:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install albumentations==1.4.3 opencv-python pillow numpy matplotlib tqdm tensorboard

2. From the project folder:
   python train_offroad.py

   This trains DeepLabV3+ (ResNet-50) for 4 epochs at 384x384 resolution and
   saves the best checkpoint as best_model.pth (by validation mIoU).

How to evaluate:
1. Ensure best_model.pth is present.
2. For validation performance:
   python test_offroad.py --split val

   This prints mean IoU, per-class IoU, average inference time, and saves overlay
   visualizations into outputs\val\.

3. (Optional) For unseen test environment:
   python test_offroad.py --split test
   Overlays are saved to outputs\test\.

Notes:
- Test images are never used for training.
- Loss: Focal Loss on top of Cross-Entropy, ignore_index=255 to handle unlabeled pixels.
- Strong data augmentation is applied to improve robustness to lighting and terrain shifts.