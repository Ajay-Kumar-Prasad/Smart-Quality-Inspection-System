import argparse
from pathlib import Path
import random
import numpy as np
import torch
import mlflow
from datetime import datetime
from ultralytics import YOLO

def  set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Base pretrained model (yolov8n.pt/yolov8s.pt/...)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="0", help="'0' or 'cpu'")
    p.add_argument("--project", type=str, default="runs/yolov8", help="Ultralytics project output")
    p.add_argument("--name", type=str, default="defect_detector", help="run name")
    p.add_argument("--mlflow_dir", type=str, default="deployment/mlflow_tracking", help="MLflow tracking dir")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    data_yaml = Path(args.data)
    assert data_yaml.exists(), f"data.yaml not found: {data_yaml}"

    #Setup MLflow
    mlflow_dir = Path(args.mlflow_dir)
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir.resolve()}")
    mlflow.set_experiment("YOLOv8_Defect_Detection")

    run_name =f"{args.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        #Log hyperparams
        mlflow.log_param("data_yaml",str(data_yaml))
        mlflow.log_param("base_model",args.model)
        mlflow.log_param("epochs",args.epochs)
        mlflow.log_param("batch", args.batch)
        mlflow.log_param("imgsz", args.imgsz)
        mlflow.log_param("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
        mlflow.log_param("seed",args.seed)

        # Load Model
        model = YOLO(args.model) 

        # Train Model
        print(f"Starting training : model={args.model} data={data_yaml} epochs={args.epochs}")
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

        try:
            # Newer Ultralytics returns a Results object with metrics attr
            metrics = results.metrics
            if hasattr(metrics, 'mAP50'):
                mlflow.log_metric("mAP50", float(metrics.mAP50))
            # sometimes metrics fields are accessible under dict
        except Exception:
            pass

        # Try alternate access
        try:
            rd = getattr(results, "results_dict", None)
            if rd:
                for k, v in rd.items():
                    # Log important metrics if present
                    if "mAP50" in k or "mAP" in k:
                        try:
                            mlflow.log_metric(k, float(v))
                        except Exception:
                            pass
        except Exception:
            pass

        # Log model artifacts: best weights and results folder
        weights_path = Path(args.project) / args.name /"weights" / "best.pt"
        last_path = Path(args.project) / args.name / "weights" / "last.pt"
        run_folder = Path(args.project) / args.name

        if weights_path.exists():
            mlflow.log_artifact(str(weights_path), artifact_path="yolo_weights")
        elif last_path.exists():
            mlflow.log_artifact(str(last_path), artifact_path="yolo_weights")

        # Log the whole run folder as artifact
        if run_folder.exists():
            mlflow.log_artifact(str(run_folder), artifact_path="yolo_run_artifacts")

        print(f"Training finished. Run ID: {run.info.run_id}")
        print("Best weights saved to:", weights_path if weights_path.exists() else last_path)

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