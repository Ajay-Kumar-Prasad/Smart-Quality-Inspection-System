## File Structure
```
smart-quality-inspection-system/
│
├── data/
│   ├── raw/                 # Original images
│   ├── processed/           # Cleaned, augmented images
│   └── annotations/         # Labels, bounding boxes, segmentation masks
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── inference_demo.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── inference.py
│   ├── utils/
│   │   ├── augmentations.py
│   │   └── metrics.py
│
├── backend/
│   ├── app.py               # FastAPI app
│   ├── routes/
│   ├── models/
│   └── database/
│       └── schema.sql
│
├── dashboard/
│   ├── streamlit_app.py     # or React frontend
│
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── mlflow_tracking/
│   └── aws_s3_setup/
│
├── tests/
│   ├── test_model.py
│   ├── test_api.py
│
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```