from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import torch
import cv2
import os
import numpy as np

# ===== PATH SETUP =====
yolo_model_path = "/content/best.pt"
sam_checkpoint = "/content/sam_vit_b.pth"
input_dir = "/content/MVTEC_AD2/images/val/"
output_dir = "/content/segment_outputs/"
os.makedirs(output_dir, exist_ok=True)

# ===== MODEL LOAD =====
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO(yolo_model_path)
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

print(f"Running on {device.upper()} ...")

# ===== PROCESS IMAGES =====
for file in os.listdir(input_dir):
    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(input_dir, file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Skipping unreadable image: {file}")
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run YOLO detection
    results = yolo_model.predict(source=img_rgb, conf=0.4, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        print(f"No detections in {file}")
        continue

    predictor.set_image(img_rgb)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        box_np = np.array([[x1, y1, x2, y2]])

        # Predict segmentation mask for each box
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_np[0],
            multimask_output=False
        )

        mask = masks[0]
        color = (0, 255, 0)

        # Overlay mask
        overlay = img.copy()
        overlay[mask] = color
        alpha = 0.5
        output = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Save files
        overlay_path = os.path.join(output_dir, f"mask_overlay_{i}_{file}")
        mask_path = os.path.join(output_dir, f"binary_mask_{i}_{os.path.splitext(file)[0]}.png")

        cv2.imwrite(overlay_path, output)
        cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)

    torch.cuda.empty_cache()
    print(f"Processed {file} — {len(boxes)} detections saved.")

# ===== METRIC CALCULATION =====
gt_path = os.path.join("/content/MVTEC_AD2/masks/val/", f"{os.path.splitext(file)[0]}.png")

if os.path.exists(gt_path):
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    pred_mask = mask.astype(np.uint8)

    # Resize ground truth to match prediction, if needed
    if gt_mask.shape != pred_mask.shape:
        gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert to torch tensors
    gt_mask = torch.tensor(gt_mask, dtype=torch.bool)
    pred_mask = torch.tensor(pred_mask, dtype=torch.bool)

    # Compute IoU and Dice
    intersection = torch.logical_and(pred_mask, gt_mask)
    union = torch.logical_or(pred_mask, gt_mask)

    iou = intersection.sum().float() / union.sum().float()
    dice = 2 * intersection.sum().float() / (pred_mask.sum().float() + gt_mask.sum().float())

    print(f"{file} | IoU: {iou:.4f} | Dice: {dice:.4f}")
else:
    print(f"⚠️ No ground truth found for {file}")



'''
1.Loads finetuned YOLOv8 model.
2.Loads SAM ViT-B checkpoint.
3.Uses YOLO's bounding boxes as SAM prompts.
4.Saves both mask overlays and binary masks.
5.Clears GPU memory each loop to avoid OOM errors.
'''