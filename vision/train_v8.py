from ultralytics import YOLO
import os
from roboflow import Roboflow
import subprocess
import matplotlib.pyplot as plt

# Run YOLOv8 training
print("Starting YOLOv8 training...")
model = YOLO("yolov8n.pt")  # load a pretrained model
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=400,
    imgsz=640,
    cache=True,
    device="0",
    batch=0.85
)
