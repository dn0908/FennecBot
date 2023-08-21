import cv2
import numpy as np
import torch
import yaml
import sys
import time

fpsLimit = 1 # limitq
startTime = time.time()


# Modify sys.path
sys.path.insert(0, "/home/smi/FennecBot/fennecbot_v05_yolov5_proto_yonsei/yolov5")

# Now import attempt_load
from yolov5.models.experimental import attempt_load

# Load the "custom" YOLOv5 model
# ** 0813 모델: 0812 데이터로 학습, 크기: 640x640
model = attempt_load('/home/smi/FennecBot/fennecbot_v05_yolov5_proto_yonsei/best.pt')

# Initialize the webcam capture
# webcam_cap = cv2.VideoCapture(0)
webcam_cap = cv2.VideoCapture("rtsp:/192.168.2.2:8554/raw", cv2.CAP_FFMPEG)

# Load class names from data.yaml
with open('/home/smi/FennecBot/fennecbot_v05_yolov5_proto_yonsei/yolov5/FennecBot_0812-3/data.yaml', 'r') as yaml_file:
    data = yaml.safe_load(yaml_file)
    class_names = data['names']

while True:
    # Capture the webcam frame
    ret, webcam_frame = webcam_cap.read()
    webcam_frame = cv2.resize(webcam_frame, (640, 480))

    nowTime = time.time()
    if (nowTime - startTime) >= fpsLimit:

        # webcam_frame = cv2.resize(webcam_frame, (480, 640))
        # print(webcam_frame.shape)
        
        # Check if frame read is valid
        if not ret:
            print("Failed to grab frame.")
            continue
        
            # Convert the webcam frame from BGR to RGB and reshape for model input
        img = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Pass the frame through the YOLOv5 model
        results = model(img_tensor)

        # Extract tensor from results tuple
        detections = results[0]

        # Assuming there's a confidence threshold you want to apply
        conf_thresh = 0.10

        # Use the confidence score to filter out weak detections
        mask = detections[0, :, 4] > conf_thresh

        # Extract the boxes, scores, and classes from the detections
        boxes = detections[0, mask, :4].cpu().numpy()
        scores = detections[0, mask, 4].cpu().numpy()
        classes = detections[0, mask, 5].cpu().numpy().astype(np.int32)

        # Draw the bounding boxes and labels on the frame
        for box, score, class_idx in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[class_idx]
            cv2.rectangle(webcam_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(webcam_frame, f"{class_name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Print the coordinates of the detected object
            print(f"{class_name} coordinates: ({x1}, {y1}), ({x2}, {y2})")

        startTime = time.time() # reset time

    cv2.imshow('Batcam Capture', webcam_frame)

    # Exit the program if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
webcam_cap.release()
cv2.destroyAllWindows()