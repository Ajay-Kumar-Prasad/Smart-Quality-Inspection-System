import os
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm

base_dir = "/Users/ajayyy/Desktop/Deep_Learning/Smart-Quality-Inspection-System/data/raw/NEU-DET"
splits = ["train", "validation"]

class_map = {
    "crazing": 0,
    "inclusion": 1,
    "pitted_surface": 2,
    "patches": 3,
    "scratches": 4,
    "rolled-in_scale": 5
}

for split in splits:
    images_root = os.path.join(base_dir, split, "images")
    ann_dir = os.path.join(base_dir, split, "annotations")
    labels_dir = os.path.join(base_dir, "labels", split)
    os.makedirs(labels_dir, exist_ok=True)

    xml_files = [f for f in os.listdir(ann_dir) if f.endswith(".xml")]
    print(f"Processing {split} split: {len(xml_files)} XML files found.")

    for xml_file in tqdm(xml_files):
        xml_path = os.path.join(ann_dir, xml_file)

        # parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # get class name from XML
        obj = root.find("object")
        if obj is None:
            print(f" No object in {xml_file}")
            continue

        cls_name = obj.find("name").text.strip()
        cls_id = class_map.get(cls_name)
        if cls_id is None:
            print(f" Unknown class {cls_name} in {xml_file}")
            continue

        # locate image
        img_name = xml_file.replace(".xml", ".jpg")  
        img_path = os.path.join(images_root, cls_name, img_name)
        if not os.path.exists(img_path):
            print(f" Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read image: {img_path}")
            continue
        h, w = img.shape[:2]

        # extract bounding box
        yolo_lines = []
        for obj in root.findall("object"):
            cls_name_obj = obj.find("name").text.strip()
            cls_id_obj = class_map.get(cls_name_obj)
            if cls_id_obj is None:
                continue

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h

            yolo_lines.append(f"{cls_id_obj} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if yolo_lines:
            txt_path = os.path.join(labels_dir, img_name.replace(".jpg", ".txt"))
            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_lines))
