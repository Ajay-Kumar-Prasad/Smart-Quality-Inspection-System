# src/data_preprocessing.py
from pathlib import Path
import cv2
import numpy as np
import random
from tqdm import tqdm
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, RandomRotate90,
    ShiftScaleRotate, RandomBrightnessContrast, HueSaturationValue,
    GaussianBlur, MotionBlur, GaussNoise, CLAHE, RandomGamma,
    Resize
)

# ----------------------------- AUGMENTATION ----------------------------------

def build_augmentations():
    """Define a stronger augmentation pipeline."""
    return Compose([
        Resize(640, 640),
        HorizontalFlip(p=0.6),
        VerticalFlip(p=0.3),
        RandomRotate90(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5),
        RandomBrightnessContrast(p=0.4),
        HueSaturationValue(p=0.3),
        GaussianBlur(blur_limit=3, p=0.2),
        MotionBlur(blur_limit=3, p=0.2),
        GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        CLAHE(p=0.3),
        RandomGamma(p=0.3),
    ], additional_targets={'mask': 'mask'})

# ----------------------------- UTILS -----------------------------------------

def mask_to_bboxes(mask):
    """Convert binary mask → YOLO bounding boxes."""
    if mask is None or np.count_nonzero(mask) == 0:
        return []
    contours, _ = cv2.findContours((mask > 127).astype('uint8'),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 2 and h > 2:
            boxes.append((x, y, w, h))
    return boxes


def save_yolo_label(txt_path, bboxes, img_w, img_h, class_id=0):
    """Save list of bounding boxes in YOLO format."""
    with open(str(txt_path), "w") as f:
        for x, y, w, h in bboxes:
            x_c = (x + w / 2) / img_w
            y_c = (y + h / 2) / img_h
            w_n = w / img_w
            h_n = h / img_h
            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")


def sanity_check(output_dir):
    """Check if images and labels count match."""
    for split in ['train', 'val']:
        img_count = len(list((output_dir / f"images/{split}").glob("*.png")))
        lbl_count = len(list((output_dir / f"labels/{split}").glob("*.txt")))
        print(f"[CHECK] {split.upper()}: {img_count} images, {lbl_count} labels")

# ----------------------------- MVTEC_AD PROCESSING ----------------------------------

def process_mvtec(root_dir, output_dir, aug, class_map=None):
    """Process MVTec AD dataset into YOLO format."""
    root_dir = Path(root_dir)
    output_img_train = output_dir / "images/train"
    output_img_val = output_dir / "images/val"
    output_lbl_train = output_dir / "labels/train"
    output_lbl_val = output_dir / "labels/val"

    for p in [output_img_train, output_img_val, output_lbl_train, output_lbl_val]:
        p.mkdir(parents=True, exist_ok=True)

    for category in root_dir.iterdir():
        if not category.is_dir():
            continue

        # Train - good (normal)
        train_good = category / "train" / "good"
        if train_good.exists():
            for img_path in train_good.iterdir():
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                augmented = aug(image=image, mask=image)
                image = augmented['image']
                out_img = output_img_train / f"{category.name}_{img_path.name}"
                cv2.imwrite(str(out_img), image)
                (output_lbl_train / out_img.name.replace('.png', '.txt')).write_text("")

        # Validation (defects + good)
        test_dir = category / "test"
        gt_dir = category / "ground_truth"
        if not test_dir.exists():
            continue

        for defect_type in test_dir.iterdir():
            if not defect_type.is_dir():
                continue

            class_id = class_map.get(defect_type.name, 0) if class_map else 0

            for img_path in defect_type.iterdir():
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                mask_path = gt_dir / defect_type.name / img_path.name.replace('.png', '_mask.png')
                mask = cv2.imread(str(mask_path), 0) if mask_path.exists() else np.zeros(image.shape[:2], dtype=np.uint8)

                augmented = aug(image=image, mask=mask)
                image, mask = augmented['image'], augmented['mask']

                dst_img = output_img_val / f"{category.name}_{defect_type.name}_{img_path.name}"
                cv2.imwrite(str(dst_img), image)

                bboxes = mask_to_bboxes(mask)
                label_path = output_lbl_val / dst_img.name.replace('.png', '.txt')
                if bboxes:
                    save_yolo_label(label_path, bboxes, *(image.shape[1::-1]), class_id=class_id)
                else:
                    label_path.write_text("")

    print("[INFO] MVTec AD processed and converted to YOLO format.")
    sanity_check(output_dir)

# ----------------------------- KolektorSDD PROCESSING ----------------------------------

def process_kolektor(root_dir, output_dir, aug, target_samples=500, val_split=0.2):
    """
    Process KolektorSDD into YOLO format with 1:1 defect:non-defect ratio
    using augmentations to reach ~1000 total images.
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)

    # Create dirs
    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            (output_dir / sub / split).mkdir(parents=True, exist_ok=True)

    print("[INFO] Scanning Kolektor dataset...")
    clean_imgs, defect_imgs = [], []

    # Identify defect vs clean
    for phase in ["train", "test"]:
        for img_path in (root_dir / phase).glob("*.png"):
            if img_path.name.endswith("_GT.png"):
                continue

            mask_path = img_path.parent / (img_path.stem + "_GT.png")
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), 0)
                if np.count_nonzero(mask) > 0:
                    defect_imgs.append(img_path)
                    continue
            clean_imgs.append(img_path)

    print(f"[INFO] Found {len(defect_imgs)} defect and {len(clean_imgs)} clean images.")

    # Balance dataset 1:1 using augmentations if needed
    n_def = len(defect_imgs)
    n_clean = len(clean_imgs)

    # If fewer defect images than target, augment to reach target
    if n_def < target_samples:
        aug_needed = target_samples - n_def
        print(f"[AUG] Need {aug_needed} extra defect samples via augmentation.")
        extra_defects = random.choices(defect_imgs, k=aug_needed)
        for img_path in tqdm(extra_defects, desc="Augmenting defect images"):
            image = cv2.imread(str(img_path))
            mask_path = img_path.parent / (img_path.stem + "_GT.png")
            mask = cv2.imread(str(mask_path), 0) if mask_path.exists() else np.zeros(image.shape[:2], dtype=np.uint8)
            augmented = aug(image=image, mask=mask)
            image_aug, mask_aug = augmented["image"], augmented["mask"]

            aug_name = f"{img_path.stem}_aug{random.randint(0,9999)}.png"
            aug_img_path = img_path.parent / aug_name
            aug_mask_path = img_path.parent / (aug_name.replace(".png", "_GT.png"))
            cv2.imwrite(str(aug_img_path), image_aug)
            cv2.imwrite(str(aug_mask_path), mask_aug)
            defect_imgs.append(aug_img_path)

    # Sample balanced clean images
    clean_imgs = random.sample(clean_imgs, min(len(clean_imgs), target_samples))
    defect_imgs = random.sample(defect_imgs, min(len(defect_imgs), target_samples))

    print(f"[INFO] Using {len(defect_imgs)} defect and {len(clean_imgs)} clean images for processing.")

    # Combine and split
    all_imgs = defect_imgs + clean_imgs
    random.shuffle(all_imgs)
    split_idx = int(len(all_imgs) * (1 - val_split))
    train_imgs = all_imgs[:split_idx]
    val_imgs = all_imgs[split_idx:]

    # Process images
    for split_name, split_imgs in zip(["train", "val"], [train_imgs, val_imgs]):
        for img_path in tqdm(split_imgs, desc=f"Processing {split_name}"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            mask_path = img_path.parent / (img_path.stem + "_GT.png")
            mask = cv2.imread(str(mask_path), 0) if mask_path.exists() else np.zeros(image.shape[:2], dtype=np.uint8)

            # Apply one more random augmentation pass
            augmented = aug(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

            out_img = output_dir / f"images/{split_name}/{img_path.name}"
            out_lbl = output_dir / f"labels/{split_name}/{img_path.stem}.txt"
            cv2.imwrite(str(out_img), image)

            bboxes = mask_to_bboxes(mask)
            if bboxes:
                save_yolo_label(out_lbl, bboxes, *(image.shape[1::-1]), class_id=0)
            else:
                out_lbl.write_text("")

    print(f"\n[INFO] KolektorSDD processed successfully (~{len(defect_imgs) + len(clean_imgs)} samples, 1:1 balanced).")
    sanity_check(output_dir)

# ----------------------------- MAIN ----------------------------------

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    aug = build_augmentations()

    datasets = {
        "mvtec": {
            "root": base_dir / "data/raw/MVTEC_AD/mvtec_anomaly_detection",
            "out": base_dir / "data/processed/MVTEC_AD"
        },
        "kolektor": {
            "root": base_dir / "data/raw/KolektorSDD/KolektorSDD2",
            "out": base_dir / "data/processed/KolektorSDD"
        }
    }

    # Define class mapping for known MVTEC_AD defect types
    CLASS_MAP = {
    'bent': 0,
    'bent_lead': 1,
    'bent_wire': 2,
    'broken': 3,
    'broken_large': 4,
    'broken_small': 5,
    'broken_teeth': 6,
    'cable_swap': 7,
    'color': 8,
    'combined': 9,
    'contamination': 10,
    'crack': 11,
    'cut': 12,
    'cut_inner_insulation': 13,
    'cut_lead': 14,
    'cut_outer_insulation': 15,
    'damaged_case': 16,
    'defective': 17,
    'fabric_border': 18,
    'fabric_interior': 19,
    'faulty_imprint': 20,
    'flip': 21,
    'fold': 22,
    'glue': 23,
    'glue_strip': 24,
    'gray_stroke': 25,
    'hole': 26,
    'liquid': 27,
    'manipulated_front': 28,
    'metal_contamination': 29,
    'misplaced': 30,
    'missing_cable': 31,
    'missing_wire': 32,
    'oil': 33,
    'pill_type': 34,
    'poke': 35,
    'poke_insulation': 36,
    'print': 37,
    'rough': 38,
    'scratch': 39,
    'scratch_head': 40,
    'scratch_neck': 41,
    'split_teeth': 42,
    'squeeze': 43,
    'squeezed_teeth': 44,
    'thread': 45,
    'thread_side': 46,
    'thread_top': 47
}


    # process_mvtec(datasets["mvtec"]["root"], datasets["mvtec"]["out"], aug, class_map=CLASS_MAP)
    process_kolektor(
    datasets["kolektor"]["root"],
    datasets["kolektor"]["out"],
    aug,
    target_samples=500,
    val_split=0.2
)


    print("\nAll datasets processed successfully! YOLO-ready data in 'data/processed/'")