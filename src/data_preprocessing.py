# src/data_preprocessing.py
from pathlib import Path
import cv2
import os
from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, VerticalFlip, RandomRotate90, Blur, Resize

# ----------------------------- AUGMENTATION ----------------------------------

def build_augmentations():
    return Compose([
        Resize(640, 640),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.2),
        RandomRotate90(p=0.3),
        RandomBrightnessContrast(p=0.2),
        Blur(blur_limit=3, p=0.1)
    ])

# ----------------------------- UTILS -----------------------------------------

def mask_to_bboxes(mask_path):
    mask = cv2.imread(str(mask_path), 0)
    if mask is None:
        return []
    contours, _ = cv2.findContours((mask > 127).astype('uint8'),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    return boxes


def save_yolo_label(txt_path, bboxes, img_w, img_h, class_id=0):
    with open(str(txt_path), "w") as f:
        for x, y, w, h in bboxes:
            x_c = (x + w / 2) / img_w
            y_c = (y + h / 2) / img_h
            w_n = w / img_w
            h_n = h / img_h
            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

# ----------------------------- MVTec PROCESSING ----------------------------------

def process_mvtec(root_dir, output_dir, aug):
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

        # Train - good images (no defects)
        train_good = category / "train" / "good"
        if train_good.exists():
            for img in train_good.iterdir():
                image = cv2.imread(str(img))
                if image is None:
                    continue
                image = aug(image=image)['image']
                out_img = output_img_train / f"{category.name}_{img.name}"
                cv2.imwrite(str(out_img), image)
                (output_lbl_train / out_img.name.replace('.png', '.txt')).write_text("")  # no defect

        # Validation - defects + good
        test_dir = category / "test"
        gt_dir = category / "ground_truth"

        if test_dir.exists():
            for defect_type in test_dir.iterdir():
                if not defect_type.is_dir():
                    continue
                for img in defect_type.iterdir():
                    image = cv2.imread(str(img))
                    if image is None:
                        continue
                    image = aug(image=image)['image']
                    dst_img = output_img_val / f"{category.name}_{defect_type.name}_{img.name}"
                    cv2.imwrite(str(dst_img), image)

                    # Find mask
                    mask_candidate = gt_dir / defect_type.name / img.name.replace('.png', '_mask.png')
                    if mask_candidate.exists():
                        bboxes = mask_to_bboxes(mask_candidate)
                        save_yolo_label(output_lbl_val / dst_img.name.replace('.png', '.txt'),
                                        bboxes, *(image.shape[1::-1]))
                    else:
                        (output_lbl_val / dst_img.name.replace('.png', '.txt')).write_text("")

    print("[INFO] MVTec AD processed and converted to YOLO format.")

# ----------------------------- KolektorSDD PROCESSING ----------------------------------

def process_kolektor(root_dir, output_dir, aug):
    root_dir = Path(root_dir)
    output_img_train = output_dir / "images/train"
    output_img_val = output_dir / "images/val"
    output_lbl_train = output_dir / "labels/train"
    output_lbl_val = output_dir / "labels/val"

    for p in [output_img_train, output_img_val, output_lbl_train, output_lbl_val]:
        p.mkdir(parents=True, exist_ok=True)

    # Process TRAIN data
    train_dir = root_dir / "train"
    for img_path in train_dir.glob("*.png"):
        # Skip GT masks
        if img_path.name.endswith("_GT.png"):
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[WARN] Skipping unreadable image: {img_path}")
            continue

        image = aug(image=image)['image']
        out_img = output_img_train / img_path.name
        cv2.imwrite(str(out_img), image)

        mask_path = train_dir / (img_path.stem + "_GT.png")
        label_path = output_lbl_train / (img_path.stem + ".txt")

        if mask_path.exists():
            bboxes = mask_to_bboxes(mask_path)
            save_yolo_label(label_path, bboxes, *(image.shape[1::-1]))
        else:
            label_path.write_text("")

    # Process TEST data
    test_dir = root_dir / "test"
    for img_path in test_dir.glob("*.png"):
        if img_path.name.endswith("_GT.png"):
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[WARN] Skipping unreadable image: {img_path}")
            continue

        image = aug(image=image)['image']
        out_img = output_img_val / img_path.name
        cv2.imwrite(str(out_img), image)

        mask_path = test_dir / (img_path.stem + "_GT.png")
        label_path = output_lbl_val / (img_path.stem + ".txt")

        if mask_path.exists():
            bboxes = mask_to_bboxes(mask_path)
            save_yolo_label(label_path, bboxes, *(image.shape[1::-1]))
        else:
            label_path.write_text("")

    print("[INFO] KolektorSDD dataset processed successfully with separate train/test folders.")

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

    # process_mvtec(datasets["mvtec"]["root"], datasets["mvtec"]["out"], aug)
    process_kolektor(datasets["kolektor"]["root"], datasets["kolektor"]["out"], aug)

    print("\n All datasets processed successfully! YOLO-ready data is in 'data/processed/'")
