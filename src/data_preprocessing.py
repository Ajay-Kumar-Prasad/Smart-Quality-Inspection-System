# src/data_preprocessing.py
from pathlib import Path
import cv2
import numpy as np
from albumentations import (
    Compose, RandomBrightnessContrast, HorizontalFlip, VerticalFlip,
    RandomRotate90, Blur, Resize
)

# ----------------------------- AUGMENTATION ----------------------------------

def build_augmentations():
    """Define augmentation pipeline for both image and mask."""
    return Compose([
        Resize(640, 640),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.2),
        RandomRotate90(p=0.3),
        RandomBrightnessContrast(p=0.2),
        Blur(blur_limit=3, p=0.1)
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

def process_kolektor(root_dir, output_dir, aug):
    """Process KolektorSDD (binary defect dataset) into YOLO format."""
    root_dir = Path(root_dir)
    output_img_train = output_dir / "images/train"
    output_img_val = output_dir / "images/val"
    output_lbl_train = output_dir / "labels/train"
    output_lbl_val = output_dir / "labels/val"

    for p in [output_img_train, output_img_val, output_lbl_train, output_lbl_val]:
        p.mkdir(parents=True, exist_ok=True)

    for phase in ["train", "test"]:
        img_out = output_img_train if phase == "train" else output_img_val
        lbl_out = output_lbl_train if phase == "train" else output_lbl_val

        for img_path in (root_dir / phase).glob("*.png"):
            if img_path.name.endswith("_GT.png"):
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                print(f"[WARN] Skipping unreadable image: {img_path}")
                continue

            mask_path = img_path.parent / (img_path.stem + "_GT.png")
            mask = cv2.imread(str(mask_path), 0) if mask_path.exists() else np.zeros(image.shape[:2], dtype=np.uint8)

            augmented = aug(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

            out_img = img_out / img_path.name
            cv2.imwrite(str(out_img), image)

            bboxes = mask_to_bboxes(mask)
            label_path = lbl_out / (img_path.stem + ".txt")

            if bboxes:
                save_yolo_label(label_path, bboxes, *(image.shape[1::-1]), class_id=0)
            else:
                label_path.write_text("")

    print("[INFO] KolektorSDD processed successfully.")
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


    process_mvtec(datasets["mvtec"]["root"], datasets["mvtec"]["out"], aug, class_map=CLASS_MAP)
    process_kolektor(datasets["kolektor"]["root"], datasets["kolektor"]["out"], aug)

    print("\nAll datasets processed successfully! YOLO-ready data in 'data/processed/'")
