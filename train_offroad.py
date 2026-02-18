import os
import random
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# =========================
# CONFIG
# =========================

TRAIN_IMG_DIR = r"E:\Startathon\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Color_Images"
TRAIN_MASK_DIR = r"E:\Startathon\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Segmentation"
VAL_IMG_DIR = r"E:\Startathon\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val\Color_Images"
VAL_MASK_DIR = r"E:\Startathon\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val\Segmentation"

# Raw class IDs in masks (uint16)
CLASS_VALUES = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
CLASS_NAMES = [
    "Trees",
    "Lush Bushes",
    "Dry Grass",
    "Dry Bushes",
    "Ground Clutter",
    "Flowers",
    "Logs",
    "Rocks",
    "Landscape",
    "Sky",
]

RAW_ID_TO_TRAIN_ID = {v: i for i, v in enumerate(CLASS_VALUES)}
NUM_CLASSES = len(CLASS_VALUES)
IGNORE_INDEX = 255

IMAGE_SIZE = (384, 384)  # (H, W)
BATCH_SIZE = 2
NUM_EPOCHS = 4	
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2  # if get issues on Windows, set to 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
CHECKPOINT_PATH = "best_model.pth"


# =========================
# UTILITIES
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def map_mask(mask_raw: np.ndarray) -> np.ndarray:
    """
    Convert raw uint16 mask (values like 100, 200, ..., 10000)
    to contiguous class indices [0..NUM_CLASSES-1].
    Any value not in CLASS_VALUES becomes IGNORE_INDEX (255).
    """
    mask = np.full(mask_raw.shape, fill_value=IGNORE_INDEX, dtype=np.uint8)
    for raw_id, train_id in RAW_ID_TO_TRAIN_ID.items():
        mask[mask_raw == raw_id] = train_id
    return mask


class OffroadDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.image_paths = sorted(
            [
                os.path.join(images_dir, fname)
                for fname in os.listdir(images_dir)
                if fname.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        mask_path = os.path.join(self.masks_dir, filename)

        # Read RGB image (BGR from cv2 -> RGB)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read 16-bit mask
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        mask = map_mask(mask_raw)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()
        else:
            # Fallback: basic tensor conversion
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask


def get_transforms():
    train_transform = A.Compose(
        [
            A.RandomResizedCrop(
                height=IMAGE_SIZE[0], width=IMAGE_SIZE[1], scale=(0.7, 1.0), ratio=(0.9, 1.1)
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.2),
            A.MotionBlur(p=0.2),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )

    return train_transform, val_transform


class FocalLoss(nn.Module):
    """
    Focal Loss on top of CrossEntropy to focus on hard / rare classes.
    """

    def __init__(self, gamma=2.0, alpha=None, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: (N, C, H, W); targets: (N, H, W)
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            ignore_index=self.ignore_index,
            weight=self.alpha,
        )  # (N, H, W)

        pt = torch.exp(-ce_loss)  # prob of true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # mask out ignore_index pixels
        valid_mask = targets != self.ignore_index
        focal_loss = focal_loss[valid_mask]

        return focal_loss.mean()


def create_model(num_classes: int):
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    # Replace classifier head with correct num_classes
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def compute_iou_per_class(preds, targets, num_classes, ignore_index=IGNORE_INDEX):
    """
    preds, targets: torch tensors (N, H, W) with class indices.
    Returns list of IoU per class.
    """
    preds = preds.view(-1).cpu().numpy().astype(np.int64)
    targets = targets.view(-1).cpu().numpy().astype(np.int64)

    mask = targets != ignore_index
    preds = preds[mask]
    targets = targets[mask]

    ious = []
    for c in range(num_classes):
        tp = np.logical_and(preds == c, targets == c).sum()
        fp = np.logical_and(preds == c, targets != c).sum()
        fn = np.logical_and(preds != c, targets == c).sum()
        denom = tp + fp + fn
        if denom == 0:
            ious.append(float("nan"))
        else:
            ious.append(tp / denom)
    return ious


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Train", ncols=100)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)["out"]  # (N, C, H, W)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Val", ncols=100)
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

    avg_loss = running_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    ious = compute_iou_per_class(all_preds, all_targets, NUM_CLASSES, IGNORE_INDEX)
    valid_ious = [x for x in ious if not np.isnan(x)]
    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0

    return avg_loss, mean_iou, ious


def main():
    print("Using device:", DEVICE)
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True

    train_transform, val_transform = get_transforms()

    train_dataset = OffroadDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
    val_dataset = OffroadDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)

    print(f"Train images: {len(train_dataset)}")
    print(f"Val images:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,   
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = create_model(NUM_CLASSES).to(DEVICE)
    criterion = FocalLoss(gamma=2.0, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_miou = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, mean_iou, ious = evaluate(model, val_loader, criterion, DEVICE)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss:   {val_loss:.4f}")
        print(f"Val mIoU:   {mean_iou:.4f}")
        for idx, (name, iou) in enumerate(zip(CLASS_NAMES, ious)):
            print(f"  Class {idx} ({name}): IoU = {iou:.4f}" if not np.isnan(iou) else f"  Class {idx} ({name}): IoU = NaN")

        scheduler.step(mean_iou)

        # Save best model
        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_miou": best_miou,
                    "epoch": epoch,
                    "class_values": CLASS_VALUES,
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                },
                CHECKPOINT_PATH,
            )
            print(f"Saved new best model with mIoU {best_miou:.4f} to {CHECKPOINT_PATH}")

    print(f"\nTraining finished. Best Val mIoU = {best_miou:.4f}")


if __name__ == "__main__":
    main()