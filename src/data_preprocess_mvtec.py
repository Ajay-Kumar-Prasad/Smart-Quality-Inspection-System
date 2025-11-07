import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# ===== PATHS =====
base_dir = "/Users/ajayyy/Desktop/Deep_Learning/Smart-Quality-Inspection-System/data"
processed_dir = os.path.join(base_dir, "processed")

# Kolektor SDD directories
train_in = os.path.join(processed_dir, "images2/train")
val_in = os.path.join(processed_dir, "images2/val")
mask_in = os.path.join(processed_dir, "masks2/val")

train_out = os.path.join(processed_dir, "processed_images2/train")
val_out = os.path.join(processed_dir, "processed_images2/val")
mask_out = os.path.join(processed_dir, "processed_masks2/val")

os.makedirs(train_out, exist_ok=True)
os.makedirs(val_out, exist_ok=True)
os.makedirs(mask_out, exist_ok=True)

# ===== TRANSFORMS =====
resize_transform = A.Resize(512, 512)

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.6),
    A.RandomBrightnessContrast(p=0.4),
    A.MotionBlur(p=0.2),
    A.GaussNoise(p=0.2),
    A.RandomCrop(256, 256, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
])

normalize = A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))

# ===== HELPERS =====
def save_image(img, out_path):
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def process_train_image(img_path, out_dir):
    img = cv2.imread(img_path)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_transform(image=img)["image"]

    base_name = os.path.basename(img_path)
    save_image(img / 255.0, os.path.join(out_dir, base_name))

    for i in range(3):
        aug_img = augment(image=img)["image"]
        save_image(aug_img / 255.0, os.path.join(out_dir, f"{os.path.splitext(base_name)[0]}_aug{i+1}.png"))

def process_val_image_and_mask(img_path, mask_path, out_img_dir, out_mask_dir):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None

    if img is None:
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = resize_transform(image=img, mask=mask)
    img, mask = transformed["image"], transformed["mask"]

    img = normalize(image=img)["image"]
    img = np.clip(img, 0, 1)

    base_name = os.path.basename(img_path)
    save_image(img, os.path.join(out_img_dir, base_name))

    if mask is not None:
        cv2.imwrite(os.path.join(out_mask_dir, base_name), mask)

# ===== EXECUTION =====
print("Processing training images (with augmentation)...")
for f in tqdm(os.listdir(train_in)):
    if f.lower().endswith((".png", ".jpg", ".jpeg")):
        process_train_image(os.path.join(train_in, f), train_out)

print("Processing validation images and masks...")
for f in tqdm(os.listdir(val_in)):
    if f.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(val_in, f)
        mask_path = os.path.join(mask_in, f)  # same name for mask
        process_val_image_and_mask(img_path, mask_path, val_out, mask_out)

print("\nPreprocessing + augmentation complete!")
