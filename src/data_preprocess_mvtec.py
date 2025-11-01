import os
import cv2
import glob
import random
import numpy as np
from tqdm import tqdm
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, RandomRotate90,
    RandomBrightnessContrast, HueSaturationValue, GaussianBlur,
    MotionBlur, RandomResizedCrop, CoarseDropout
)
from sklearn.model_selection import train_test_split
import shutil
import yaml

# === CONFIG ===
DATASET_DIR = "/Users/ajayyy/Desktop/Deep_Learning/Smart-Quality-Inspection-System/data/raw/MVTEC_AD/mvtec_anomaly_detection"  # Path to MVTec AD dataset root
OUT_DIR = "/Users/ajayyy/Desktop/Deep_Learning/Smart-Quality-Inspection-System/data/processed/MVTEC_AD"
TARGET_SIZE = (640, 640)
NUM_IMAGES = 10000  # total target samples
VAL_SPLIT = 0.2
random.seed(42)

# === AUGMENTATION PIPELINES ===
def strong_aug(p=1.0):
    return Compose([
        RandomResizedCrop(size=(TARGET_SIZE[0], TARGET_SIZE[1]), scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.3),
        RandomRotate90(p=0.3),
        RandomBrightnessContrast(p=0.5),
        HueSaturationValue(p=0.4),
        GaussianBlur(p=0.3),
        MotionBlur(p=0.2),
        CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5)
    ], p=p)

aug = strong_aug(p=1.0)

# === CREATE OUTPUT DIRECTORIES ===
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

# === LOAD DATA ===
categories = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
all_images = []

for cat in categories:
    train_path = os.path.join(DATASET_DIR, cat, "train")
    test_path = os.path.join(DATASET_DIR, cat, "test")
    gt_path = os.path.join(DATASET_DIR, cat, "ground_truth")

    # Non-defective (train)
    for img_path in glob.glob(f"{train_path}/*/*.png"):
        all_images.append((img_path, 0))  # class 0 = non-defective

    # Defective (test + ground truth)
    for img_path in glob.glob(f"{test_path}/*/*.png"):
        if "good" not in img_path:
            all_images.append((img_path, 1))  # class 1 = defective

random.shuffle(all_images)
if len(all_images) > NUM_IMAGES:
    all_images = all_images[:NUM_IMAGES]

# === AUGMENT AND CONVERT TO YOLO FORMAT ===
def process_image(img_path, label):
    img = cv2.imread(img_path)
    if img is None:
        return None, None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    aug_img = aug(image=img)["image"]

    # Dummy bounding box: center full image for now
    h, w, _ = aug_img.shape
    x_center, y_center, bw, bh = 0.5, 0.5, 1.0, 1.0
    label_txt = f"{label} {x_center} {y_center} {bw} {bh}\n"
    return cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR), label_txt

images = []
for img_path, label in tqdm(all_images, desc="Processing images"):
    img, lbl = process_image(img_path, label)
    if img is not None:
        images.append((img, lbl))

# === TRAIN-VAL SPLIT ===
train_data, val_data = train_test_split(images, test_size=VAL_SPLIT, random_state=42)

def save_split(data, split_name):
    for i, (img, lbl) in enumerate(data):
        img_name = f"{split_name}_{i:05d}.jpg"
        lbl_name = img_name.replace(".jpg", ".txt")

        img_path = os.path.join(OUT_DIR, f"images/{split_name}", img_name)
        lbl_path = os.path.join(OUT_DIR, f"labels/{split_name}", lbl_name)

        cv2.imwrite(img_path, img)
        with open(lbl_path, "w") as f:
            f.write(lbl)

save_split(train_data, "train")
save_split(val_data, "val")

print(f"Total processed: {len(images)}")
print(f"Train: {len(train_data)} | Val: {len(val_data)}")

# === CREATE data.yaml ===
data_yaml = {
    "train": f"{OUT_DIR}/images/train",
    "val": f"{OUT_DIR}/images/val",
    "nc": 2,
    "names": ["non_defective", "defective"]
}

with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
    yaml.dump(data_yaml, f)

print("YOLOv8 dataset ready at:", OUT_DIR)
print("data.yaml created at:", os.path.join(OUT_DIR, "data.yaml"))
