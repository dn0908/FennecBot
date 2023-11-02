import cv2
import logging
import numpy as np
from batcam.websocket_client import *

import pyzbar.pyzbar as pyzbar
from tensorflow import keras
import librosa


import websocket
import json
import threading
from threading import Thread

import wave
import time

import requests

import base64
# Use code blocks to display formatted content such as code
import rel
import pickle
import pandas as pd
import csv
import numpy as np


import torch
import yaml
import sys
import time


logging.basicConfig(level=logging.INFO)

class BatCam:
    def __init__(self):
        # BATCAM RTSP_URL = "rtsp://<username>:<password>@<ip>:8544/raw
        BATCAM_IP = '192.168.2.2'
        self.RTSP_URL = "rtsp:/192.168.2.2:8554/raw"
        # self.center_x = 800
        # self.center_y = 600 # Batcam center pixels

        ##### QR DETECTION : CODE INFO CONFIGURATION #####
        self.qr_x = 0 # QR position data
        self.qr_y = 0
        self.code_info : str= "" # QR code info

        ##### OBJECT DETECTION : CUSTOM YOLOv5 MODEL CONFIGURATION #####
        #sys.path.insert(0, "/home/smi/FennecBot/fennecbot_v05_yolov5_proto_yonsei/yolov5")
        from yolov5.models.experimental import attempt_load # Now import attempt_load
        self.yolo_model = attempt_load('/home/smi/FennecBot/best_231016.pt') # Load the "custom" YOLOv5 model
        self.x1, self.y1, self.x2, self.y2 = 0, 0, 0, 0
        self.class_name : str= ""


        # self.BF_data = []
        # self.FullScan_arr = []
        # self.frame = []
        # self.on_message = []


        ##### NOISE DETECTION : DEEP LEARNING MODEL CONFIGURATIONS #####
        self.noise_model = keras.models.load_model('./models/model.h5')
        # self.noise_model.summary()
        self.sr = 96000
        self.window_size_sec = 0.05  # Window size in seconds
        self.window_size = int(self.window_size_sec / (512 / self.sr))
        self.stride_sec = self.window_size_sec/2.0  # Stride length in seconds
        self.stride = int(self.stride_sec / (512 / self.sr))
        self.noise_detection = 0


    def read_QRcodes(self, frame):
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # equalized = clahe.apply(gray)
        codes = pyzbar.decode(frame)
        for code in codes:
            x, y , w, h = code.rect
            self.qr_x = (2*x+w)/2
            self.qr_y = (2*y+h)/2
            self.code_info = code.data.decode('utf-8')
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, self.code_info, (x + 6, y - 6), font, 2.0, (255, 255, 255), 1)
            print(f"QR code info : {self.code_info}, Center x : {self.qr_x}, Center y : {self.qr_y}")
        
        return self.code_info
        
        # codes = pyzbar.decode(gray_frame)
        # print('reading QR in frame')
        # for code in codes:
        #     x, y , w, h = code.rect
        #     self.qr_x = (2*x+w)/2
        #     self.qr_y = (2*y+h)/2
        #     self.code_info = code.data.decode('utf-8')
        #     # make bounding box around code
        #     cv2.rectangle(frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
        #     # display info text
        #     font = cv2.FONT_HERSHEY_DUPLEX
        #     cv2.putText(frame, self.code_info, (x + 6, y - 6), font, 2.0, (255, 255, 255), 1)
        #     print(f"QR code info : {self.code_info}, Center x : {self.qr_x}, Center y : {self.qr_y}")
        
        # return self.code_info
        
    def yolo_detection(self, webcam_frame):
        # Convert the webcam frame from BGR to RGB and reshape for model input
        img = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Pass the frame through the YOLOv5 model   
        results = self.yolo_model(img_tensor)
        # Extract tensor from results tuple
        detections = results[0]

        # Assuming there's a conf9idence threshold you want to apply
        conf_thresh = 0.10
        # Use the confidence score to filter out weak detections
        mask = detections[0, :, 4] > conf_thresh

        # Extract the boxes, scores, and classes from the detections
        boxes = detections[0, mask, :4].cpu().numpy()
        scores = detections[0, mask, 4].cpu().numpy()
        classes = detections[0, mask, 5].cpu().numpy().astype(np.int32)

        # Load class names from data.yaml
        with open('/home/smi/FennecBot/fennecbot_v05_yolov5_proto_yonsei/yolov5/FennecBot_0812-3/data.yaml', 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)
            self.class_names = data['names']

        # Draw the bounding boxes and labels on the frame
        for box, score, class_idx in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            self.class_name = self.class_names[class_idx]
            cv2.rectangle(webcam_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(webcam_frame, f"{self.class_name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Print the coordinates of the detected object
            print(f"{self.class_name} coordinates: ({x1}, {y1}), ({x2}, {y2})")

        return self.x1,self.y1, self.x2, self.y2, self.class_name #change to self


    def multiple_yolo_detection(self, webcam_frame):
        # Convert the webcam frame from BGR to RGB and reshape for model input
        img = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Pass the frame through the YOLOv5 model   
        results = self.yolo_model(img_tensor)
        # Extract tensor from results tuple
        detections = results[0]

        # Create a list to store the results
        self.yolo_list = []

        # Assuming there's a confidence threshold you want to apply
        conf_thresh = 0.90
        # Use the confidence score to filter out weak detections
        mask = detections[0, :, 4] > conf_thresh

        # Extract the boxes, scores, and classes from the detections
        boxes = detections[0, mask, :4].cpu().numpy()
        scores = detections[0, mask, 4].cpu().numpy()
        classes = detections[0, mask, 5].cpu().numpy().astype(np.int32)

        # Load class names from data.yaml
        with open('/home/smi/FennecBot/BATCAM-MX-Data-Labeling-1/data.yaml', 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)
            self.class_names = data['names']

        # Draw the bounding boxes and labels on the frame
        for box, score, class_idx in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            class_name = self.class_names[class_idx]

            yolo_x = (x1 + x2) / 2
            yolo_y = (y1 + y2) / 2
            result = {
                "class_name": class_name,
                "score": score,
                "x_coordinate": yolo_x,
                "y_coordinate": yolo_y
            }
            self.yolo_list.append(result)

            cv2.rectangle(webcam_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(webcam_frame, f"{self.class_name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Print the coordinates of the detected object
            print(f"{class_name} coordinates: ({x1}, {y1}), ({x2}, {y2})")

        return self.yolo_list #change to self



    def save_BF(self):
        USER = "user"
        PASSWORD = "user"
        BATCAM_IP = "192.168.2.2"
        credential = f"{USER}:{PASSWORD}"
        base64EncodedAuth = base64.b64encode(credential.encode()).decode()
        trigger_id = None
        count_num = 0
        data_list = []
        print(count_num)

        ws = websocket.WebSocketApp(f"ws://{BATCAM_IP}/ws",
                                    on_open=on_open,
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close,
                                    subprotocols=["subscribe"],
                                    header={"Authorization": f"Basic %s" % base64EncodedAuth}
                                )
        Thread(target=ws.run_forever).start()
        time.sleep(1.5)
        ws.close()
        print("WebSocket Closed, Count Num to zero...")


    # return ws
    def change_LPoint(self, new_index):
        USER = "admin"
        PASSWORD = "admin"
        credential = f"{USER}:{PASSWORD}"
        base64EncodedAuth = base64.b64encode(credential.encode()).decode()

        API_HOST = f"http://{BATCAM_IP}"
        url = API_HOST + "/beamforming/setting"
        headers = {
            "Authorization": f"Basic {base64EncodedAuth}"
        }
        camera_settings = {
            "low_cut": 2000,
            "high_cut": 45000,
            "gain": 100,
            "distance": 10,
            "x_cal": -0.02,
            "y_cal": 0.02,
            "index_l_channel": 0,
            "index_l": f"1199,{new_index}"
        }
        print("Requesting body: " + str(camera_settings) + "\n")
        try:
            current_setting = requests.get(url, headers=headers)
            print("Current Camera setting: " + str(current_setting.json()) + "\n")

            response = requests.patch(url, headers=headers, data=camera_settings)
            print("Result: " + str(response.json()) + "\n")
        except Exception as ex:
            print(ex)

    def leakage_detection(self):
        def extract_features_v4(file_path, window_size, stride):
            data = pd.read_csv(file_path)
            audio = data.values.reshape(-1)

            # Perform STFT on the entire audio
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            print("stft size: ", stft.shape)

            # Calculate the number of time steps
            num_time_steps = stft.shape[1]

            features = []
            for i in range(0, num_time_steps, stride):
                # Determine the start and end indices for the window
                time_start = i
                time_end = i + window_size

                # Check if the window exceeds the audio signal length
                if time_end > num_time_steps:
                    break

                # Extract features for each time step
                stft_window = stft[:, time_start:time_end]

                # Transpose the features to have time as the first axis
                stft_window = np.transpose(stft_window)

                # Append the features to the list
                features.append(stft_window)

            return features
        X_train = []
        file_path = 'l_point_data.csv'
        class_features = extract_features_v4(file_path, self.window_size, self.stride)
        X_train.extend(class_features)
        X_train = np.array(X_train)
        print('Noise Input Data Shape : ',X_train.shape)

        y_pred_prob = self.noise_model.predict(X_train)
        y_pred = np.argmax(y_pred_prob, axis=1)
        print('Predicted Noise Class : ', y_pred)
        y_pred_mean = np.mean(y_pred)
        print('Predicted Noise Class MEAN : ', y_pred_mean)

        if y_pred_mean >= 0.5 :      # if detected, self.noise_detection changes to 1
            print(' ! Leakage Detected ! ')
            self.noise_detection = 1
        else :                       # if not, self.noise_detection remains 0
            print('NO Leakage Detected')
            self.noise_detection = 0
        
        return self.noise_detection  # return self.noise_detection


    def rtsp_to_opencv(self, QR_toggle = 0, yolo_toggle = 0, BF_toggle = 0):
        fpsLimit = 1 # limitq
        startTime = time.time()
        logging.info(f' Trying to connect to {self.RTSP_URL}...')

        cap = cv2.VideoCapture(self.RTSP_URL, cv2.CAP_FFMPEG)

        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 480)) #resize cap for modelq input
            #add for practice
            # self.read_QRcodes(frame)
            # frame = cv2.resize(frame, (1280, 960))
            # Check if frame read is valid
            if not ret:
                print("Failed to grab frame.")
                continue

            nowTime = time.time()
            if (nowTime - startTime) >= fpsLimit:
                self.frame = frame #?
                
                if QR_toggle != 0:
                    prev_code_info = self.code_info
                    
                    frame = cv2.resize(frame, (1280, 960))
                     
                    #add for practice
                    #self.read_QRcodes(frame)
                    self.code_info = self.read_QRcodes(frame)
                    # if self.code_info != prev_code_info:
                    #     QR_toggle = 0
                    #     break

                if yolo_toggle != 0:
                    # frame = cv2.resize(frame, (640, 480))
                    #self.x1, self.y1, self.x2, self.y2, self.class_name = self.yolo_detection(frame)
                    self.yolo_list = self.multiple_yolo_detection(frame)
                    break

                if BF_toggle != 0:
                    self.BF_data = self.save_BF()
                    BF_toggle = 0
                    break

                startTime = time.time() # reset time
                #add for practice
            #gray = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2GRAY)   
            cv2.imshow('Batcam Capture',frame)
            if cv2.waitKey(500) == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    Batcam = BatCam()
    # Batcam.save_BF()  
    # Batcam.leakage_detection()
    try:
        Batcam.rtsp_to_opencv(QR_toggle = 1, yolo_toggle = 0, BF_toggle=0)
    except Exception as error:
        logging.error(error)