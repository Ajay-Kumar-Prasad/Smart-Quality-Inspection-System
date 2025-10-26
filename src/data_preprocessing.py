from pathlib import Path
import cv2
from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, VerticalFlip, RandomRotate90, Blur, Resize

def build_augmentations():
    return Compose([
        Resize(640, 640),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.2),
        RandomRotate90(p=0.3),
        RandomBrightnessContrast(p=0.2),
        Blur(blur_limit=3, p=0.1)
    ])

def process_images(input_dir, output_dir):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    
    aug = build_augmentations()
    
    img_files = list(input_dir.rglob("*.[pj][pn]g"))  # jpg, jpeg, png
    if not img_files:
        print(f"No images found in {input_dir}")
        return
    
    total = 0
    for img_file in img_files:
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        augmented = aug(image=img)['image']
        
        # Keep relative folder structure
        rel_path = img_file.relative_to(input_dir)
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), augmented)
        total += 1

    print(f"Processing complete! Total images: {total}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent  # project root

    datasets = {
        base_dir / "data/raw/MVTEC_AD/mvtec_anomaly_detection": base_dir / "data/processed/MVTEC_AD",
        base_dir / "data/raw/KolektorSDD/KolektorSDD2": base_dir / "data/processed/KolektorSDD"
    }

    for inp, out in datasets.items():
        # process each **class subfolder** inside MVTEC_AD automatically
        if "MVTEC_AD" in str(inp):
            for class_folder in inp.iterdir():
                if class_folder.is_dir():
                    process_images(class_folder, out / class_folder.name)
        else:
            process_images(inp, out)
