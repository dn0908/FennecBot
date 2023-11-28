import cv2
import numpy as np
import torch
import yaml
import sys

# Modify sys.path
# sys.path.insert(0, "/home/sjbuhan/workspace/hyundai_bsr/fennecbot_v05_yolov5_proto_yonsei/yolov5/")

# Now import attempt_load
from yolov5.models.experimental import attempt_load

# Load the "custom" YOLOv5 model
model = attempt_load('/home/smi/FennecBot/1106_2_best.pt')
model_input_size = 640  # Model's expected input size

# Initialize the webcam capture
RTSP_URL = "rtsp:/192.168.2.2:8554/raw"
webcam_cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

# Load class names from data.yaml
with open('/home/smi/FennecBot/data.yaml', 'r') as yaml_file:
    data = yaml.safe_load(yaml_file)
    class_names = data['names']

while True:
    # Capture the webcam frame
    ret, webcam_frame = webcam_cap.read()
    
    # Check if frame read is valid
    if not ret:
        print("Failed to grab frame.")
        continue
    
    # Record original frame size
    original_frame_height, original_frame_width = webcam_frame.shape[:2]

    # Resize frame to model input size
    resized_frame = cv2.resize(webcam_frame, (model_input_size, model_input_size))

    # Convert the resized frame from BGR to RGB and reshape for model input
    # img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(resized_frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Pass the frame through the YOLOv5 model
    results = model(img_tensor)

    # Extract tensor from results tuple
    detections = results[0]

    # Confidence threshold to apply
    conf_thresh = 0.15

    # Use the confidence score to filter out weak detections
    mask = detections[0, :, 4] > conf_thresh

    # Extract the boxes, scores, and classes from the detections
    boxes = detections[0, mask, :4].cpu().numpy()
    scores = detections[0, mask, 4].cpu().numpy()
    classes = detections[0, mask, 5].cpu().numpy().astype(np.int32)

    # Draw the bounding boxes and labels on the original frame
    for box, score, class_idx in zip(boxes, scores, classes):
        # Scale the bounding box coordinates back to the original frame's size
        x1, y1, x2, y2 = box
        x1 = x1 * original_frame_width / model_input_size
        x2 = x2 * original_frame_width / model_input_size
        y1 = y1 * original_frame_height / model_input_size
        y2 = y2 * original_frame_height / model_input_size

        # Convert to integer values
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        class_name = class_names[class_idx]
        cv2.rectangle(webcam_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(webcam_frame, f"{class_name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Print the coordinates of the detected object
        print(f"{class_name} coordinates: ({x1}, {y1}), ({x2}, {y2})")

    # Display the frame on the screen
    cv2.imshow('Webcam Capture', webcam_frame)

    # Exit the program if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
webcam_cap.release()
cv2.destroyAllWindows()