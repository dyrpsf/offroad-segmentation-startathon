import os
import time
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ------------------
# CONFIG
# ------------------

# Paths
TRAIN_IMG_DIR = r"E:\Startathon\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Color_Images"
TRAIN_MASK_DIR = r"E:\Startathon\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Segmentation"
VAL_IMG_DIR = r"E:\Startathon\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val\Color_Images"
VAL_MASK_DIR = r"E:\Startathon\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val\Segmentation"

TEST_IMG_DIR = r"E:\Startathon\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Color_Images"
TEST_MASK_DIR = r"E:\Startathon\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Segmentation"

CHECKPOINT_PATH = "best_model.pth"
OUT_DIR_BASE = "outputs"

# Class mapping (same as train script)
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

IMAGE_SIZE = (384, 384)  # must match your training config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Colors for visualization (BGR)
CLASS_COLORS = {
    0: (0, 128, 0),      # Trees - dark green
    1: (0, 255, 0),      # Lush Bushes - light green
    2: (0, 255, 255),    # Dry Grass - yellow
    3: (0, 128, 128),    # Dry Bushes - teal/olive
    4: (19, 69, 139),    # Ground Clutter - brownish
    5: (255, 0, 255),    # Flowers - magenta
    6: (45, 82, 160),    # Logs - brown
    7: (128, 128, 128),  # Rocks - gray
    8: (140, 180, 210),  # Landscape - tan/ground
    9: (235, 206, 135),  # Sky - sky-ish
}


# ------------------
# UTILS
# ------------------

def map_mask(mask_raw: np.ndarray) -> np.ndarray:
    """Map raw uint16 IDs to contiguous [0..NUM_CLASSES-1], others -> IGNORE_INDEX."""
    mask = np.full(mask_raw.shape, fill_value=IGNORE_INDEX, dtype=np.uint8)
    for raw_id, train_id in RAW_ID_TO_TRAIN_ID.items():
        mask[mask_raw == raw_id] = train_id
    return mask


def get_val_transform():
    return A.Compose(
        [
            A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )


def colorize_mask(pred_mask: np.ndarray) -> np.ndarray:
    """pred_mask: (H, W) with class indices 0..9. Returns BGR color image."""
    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        color_mask[pred_mask == cls_id] = color
    return color_mask


def compute_iou_per_class(all_preds, all_targets):
    """all_preds/all_targets: torch tensors (N,H,W) with class indices."""
    preds = all_preds.view(-1).cpu().numpy().astype(np.int64)
    targets = all_targets.view(-1).cpu().numpy().astype(np.int64)

    mask = targets != IGNORE_INDEX
    preds = preds[mask]
    targets = targets[mask]

    ious = []
    for c in range(NUM_CLASSES):
        tp = np.logical_and(preds == c, targets == c).sum()
        fp = np.logical_and(preds == c, targets != c).sum()
        fn = np.logical_and(preds != c, targets == c).sum()
        denom = tp + fp + fn
        if denom == 0:
            ious.append(float("nan"))
        else:
            ious.append(tp / denom)
    return ious


def compute_confusion_matrix(all_preds, all_targets, num_classes=NUM_CLASSES):
    """
    all_preds, all_targets: torch tensors (N, H, W) with class indices.
    Returns a (num_classes, num_classes) confusion matrix where
    rows = ground truth, cols = predictions.
    """
    preds = all_preds.view(-1).cpu().numpy().astype(np.int64)
    targets = all_targets.view(-1).cpu().numpy().astype(np.int64)

    mask = targets != IGNORE_INDEX
    preds = preds[mask]
    targets = targets[mask]

    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(conf_mat, (targets, preds), 1)
    return conf_mat


def load_model():
    print("Loading model from", CHECKPOINT_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # Build same base architecture as in training
    model = deeplabv3_resnet50(weights=None)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

    state_dict = checkpoint["model_state_dict"]

    # Load while ignoring unexpected keys like aux_classifier.*
    incompatible = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", incompatible.missing_keys)
    print("Unexpected keys:", incompatible.unexpected_keys)

    model.to(DEVICE)
    model.eval()
    return model


def evaluate_split(split: str, save_visuals: bool = True, max_visuals: int = 30):
    if split == "val":
        img_dir = VAL_IMG_DIR
        mask_dir = VAL_MASK_DIR
    elif split == "test":
        img_dir = TEST_IMG_DIR
        mask_dir = TEST_MASK_DIR
    else:
        raise ValueError("split must be 'val' or 'test'")

    out_dir = os.path.join(OUT_DIR_BASE, split)
    os.makedirs(out_dir, exist_ok=True)

    transform = get_val_transform()
    model = load_model()

    image_files = sorted(
        [
            f
            for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    all_preds = []
    all_targets = []

    total_time = 0.0
    count = 0
    vis_count = 0

    print(f"Evaluating split = {split}, num images = {len(image_files)}")

    for fname in image_files:
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        # original image in BGR for saving overlays
        orig_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if orig_bgr is None:
            print("Could not read image:", img_path)
            continue
        orig_h, orig_w = orig_bgr.shape[:2]

        # read mask as uint16 and map to train IDs
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            print("Could not read mask:", mask_path)
            continue
        mask_mapped = map_mask(mask_raw)

        # apply same val transform as training
        augmented = transform(
            image=cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB),
            mask=mask_mapped,
        )
        image_t = augmented["image"].unsqueeze(0).to(DEVICE)  # (1,3,H,W)
        mask_t = augmented["mask"].long()  # (H,W)

        # inference
        start_t = time.time()
        with torch.no_grad():
            out = model(image_t)["out"]  # (1,C,H,W)
            pred = out.argmax(1)[0].cpu()  # (H,W)
        end_t = time.time()

        total_time += (end_t - start_t)
        count += 1

        all_preds.append(pred)
        all_targets.append(mask_t.cpu())

        # visualization (limited number)
        if save_visuals and vis_count < max_visuals:
            pred_np = pred.numpy().astype(np.uint8)
            pred_resized = cv2.resize(
                pred_np,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

            color_mask = colorize_mask(pred_resized)
            overlay = cv2.addWeighted(orig_bgr, 0.5, color_mask, 0.5, 0)

            base = os.path.splitext(fname)[0]
            cv2.imwrite(os.path.join(out_dir, f"{base}_overlay.png"), overlay)
            cv2.imwrite(os.path.join(out_dir, f"{base}_mask.png"), color_mask)

            vis_count += 1

    if count == 0:
        print("No images processed!")
        return

    # Convert lists -> tensors
    all_preds = torch.stack(all_preds, dim=0)
    all_targets = torch.stack(all_targets, dim=0)

    # IoU
    ious = compute_iou_per_class(all_preds, all_targets)
    valid_ious = [x for x in ious if not np.isnan(x)]
    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0

    # Confusion matrix
    conf_mat = compute_confusion_matrix(all_preds, all_targets, NUM_CLASSES)

    print(f"\n=== {split.upper()} RESULTS ===")
    print(f"Mean IoU: {mean_iou:.4f}")
    for idx, (name, iou) in enumerate(zip(CLASS_NAMES, ious)):
        if np.isnan(iou):
            print(f"  Class {idx} ({name}): IoU = NaN")
        else:
            print(f"  Class {idx} ({name}): IoU = {iou:.4f}")

    print("\nConfusion matrix (rows = GT, cols = Pred):")
    print("Rows/cols order:", ", ".join(CLASS_NAMES))
    print(conf_mat)

    avg_time = total_time / count
    print(f"\nAverage inference time per image: {avg_time*1000:.2f} ms")
    print(f"Visualizations saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--no_visuals",
        action="store_true",
        help="Disable saving overlay images.",
    )
    args = parser.parse_args()

    evaluate_split(args.split, save_visuals=not args.no_visuals)


if __name__ == "__main__":
    main()