import argparse
from pathlib import Path
import random
import numpy as np
import torch
import mlflow
from datetime import datetime
from ultralytics import YOLO


def set_seed(seed=42):
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train YOLOv8m for defect detection with MLflow tracking")
    p.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    p.add_argument("--device", type=str, default="auto", help="'0' for GPU, 'cpu' for CPU, or 'auto'")
    p.add_argument("--project", type=str, default="runs/yolov8", help="Ultralytics project output folder")
    p.add_argument("--name", type=str, default="defect_detector_yolov8m", help="Name for this training run")
    p.add_argument("--mlflow_dir", type=str, default="deployment/mlflow_tracking", help="MLflow tracking directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    mlflow_dir = Path(args.mlflow_dir).resolve()
    mlflow_dir.mkdir(parents=True, exist_ok=True)

    # Auto-select device
    if args.device == "auto":
        args.device = "0" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")

    # Setup MLflow
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    mlflow.set_experiment("YOLOv8_Defect_Detection")

    run_name = f"{args.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        # Log hyperparameters
        mlflow.log_params({
            "data_yaml": str(data_yaml),
            "base_model": "yolov8m.pt",
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "seed": args.seed
        })

        # Load YOLOv8m model
        model = YOLO("yolov8m.pt")

        # Train model
        print(f"Starting training on dataset: {data_yaml}")
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
            device=args.device,
            exist_ok=True
        )

        # Log metrics safely
        if hasattr(results, "metrics"):
            for k, v in results.metrics.items():
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    pass

        # Log best weights
        weights_dir = Path(args.project) / args.name / "weights"
        best_path = weights_dir / "best.pt"
        last_path = weights_dir / "last.pt"

        if best_path.exists():
            mlflow.log_artifact(str(best_path), artifact_path="yolo_weights")
        elif last_path.exists():
            mlflow.log_artifact(str(last_path), artifact_path="yolo_weights")

        # Log full run directory
        run_folder = Path(args.project) / args.name
        if run_folder.exists():
            mlflow.log_artifact(str(run_folder), artifact_path="yolo_run_artifacts")

        print(f"\n✅ Training completed successfully.")
        print(f"📦 MLflow Run ID: {run.info.run_id}")
        print(f"🏋️ Best weights saved to: {best_path if best_path.exists() else last_path}")


if __name__ == "__main__":
    main()

'''
1.Reads config from CLI
2.Sets random seeds for reproducibility
3.Starts an MLflow run for tracking
4.Loads YOLOv8 pretrained model
5.Trains it on defect dataset
6.Logs metrics + artifacts to MLflow
7.Saves weights in runs/yolov8/.../weights/best.pt
'''