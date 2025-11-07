import os
import cv2
import numpy as np
from tqdm import tqdm

# === PATHS ===
base_dir = "/Users/ajayyy/Desktop/Deep_Learning/Smart-Quality-Inspection-System/data/processed"
#mvtec_ad
# masks_dir = os.path.join(base_dir, "masks")
# images_dir = os.path.join(base_dir, "images")
# labels_dir = os.path.join(base_dir, "labels")

## kolektor SDD
masks_dir = os.path.join(base_dir, "masks2")
images_dir = os.path.join(base_dir, "images2")
labels_dir = os.path.join(base_dir, "labels2")

os.makedirs(labels_dir, exist_ok=True)

def mask_to_yolo(mask_path, image_path, label_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(image_path)
    if mask is None or img is None:
        return False  # skip if file missing or broken

    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False  # no defect found

    valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
    if not valid_contours:
        return False

    with open(label_path, "w") as f:
        for c in valid_contours:
            x, y, bw, bh = cv2.boundingRect(c)
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw /= w
            bh /= h
            f.write(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

    # check if file actually contains content
    return os.path.getsize(label_path) > 0

# === CONVERT SPLITS ===
splits = [d for d in os.listdir(masks_dir) if os.path.isdir(os.path.join(masks_dir, d))]

for split in splits:
    masks_split = os.path.join(masks_dir, split)
    imgs_split = os.path.join(images_dir, split)
    labels_split = os.path.join(labels_dir, split)
    os.makedirs(labels_split, exist_ok=True)

    mask_files = [f for f in os.listdir(masks_split) if f.lower().endswith((".png", ".jpg"))]
    print(f"\nProcessing '{split}' set: {len(mask_files)} masks found.")

    for mf in tqdm(mask_files):
        mask_path = os.path.join(masks_split, mf)
        img_name = mf.replace("_mask", "")  # adjust this if your naming differs
        img_path = os.path.join(imgs_split, img_name)
        if not os.path.exists(img_path):
            continue

        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_split, label_name)
        success = mask_to_yolo(mask_path, img_path, label_path)

        if not success and os.path.exists(label_path):
            os.remove(label_path)  # delete empty file

print("\nLabel generation complete — only defective samples retained.")
