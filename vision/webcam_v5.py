import cv2
import torch
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/jonathan.pt')
# Set confidence threshold
model.conf = 0.25  # Confidence threshold
model.iou = 0.45   # IoU threshold

cap = cv2.VideoCapture(0)
# Set the resolution to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to continuously get frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Make predictions
    results = model(frame)
    
    # Render results on the frame
    rendered_frame = results.render()[0]  # Get the rendered frame with annotations
    
    # Display the resulting frame
    cv2.imshow('YOLOv5 Detection', rendered_frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord('q'):  # Increased delay for key detection
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
