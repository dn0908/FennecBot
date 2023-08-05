import cv2
import logging
import numpy as np
from batcam.websocket_client import *
import torch
import pyzbar.pyzbar as pyzbar
from tensorflow import keras
import librosa


import websocket
import json
import threading
from threading import Thread

import wave
import time


import base64
# Use code blocks to display formatted content such as code
import rel
import pickle
import pandas as pd
import csv
import numpy as np



logging.basicConfig(level=logging.INFO)

class BatCam:
    def __init__(self):
        # BATCAM RTSP_URL = "rtsp://<username>:<password>@<ip>:8544/raw
        BATCAM_IP = '192.168.2.2'
        self.RTSP_URL = "rtsp:/192.168.2.2:8554/raw"
        # Batcam center pixels
        self.center_x = 800
        self.center_y = 600

        # # QR position data
        self.qr_x = 0
        self.qr_y = 0

        # Load the "custom" YOLOv5 model
        # self.model = torch.hub.load('./yolo_v5', 'custom', path='./yolo_v5/yolov5s.pt', source='local')
        self.x1, self.y1, self.x2, self.y2 = 0, 0, 0, 0
        # self.BF_data = []
        self.code_info : str= ""
        self.FullScan_arr = []
        self.frame = []
        # self.on_message = []

        # Noise Model
        self.noise_model = keras.models.load_model('./models/model.h5')
        # self.noise_model.summary()
        self.sr = 96000
        self.window_size_sec = 0.05  # Window size in seconds
        self.window_size = int(self.window_size_sec / (512 / self.sr))
        self.stride_sec = self.window_size_sec/2.0  # Stride length in seconds
        self.stride = int(self.stride_sec / (512 / self.sr))
        self.noise_detection = 0

    def read_QRcodes(self, frame):
        codes = pyzbar.decode(frame)
        for code in codes:
            x, y , w, h = code.rect
            self.qr_x = (2*x+w)/2
            self.qr_y = (2*y+h)/2
            self.code_info = code.data.decode('utf-8')
            # # make bounding box around code
            # cv2.rectangle(frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
            # # display info text
            # font = cv2.FONT_HERSHEY_DUPLEX
            # cv2.putText(frame, self.code_info, (x + 6, y - 6), font, 2.0, (255, 255, 255), 1)
            print(f"QR code info : {self.code_info}, Center x : {self.qr_x}, Center y : {self.qr_y}")
        
        return self.code_info
        
    # def yolo_detection(self, webcam_frame):
    #     # Pass the frame through the YOLOv5 model
    #     results = self.model(webcam_frame)

    #     # Extract the boxes, scores, and classes from the results
    #     boxes = results.xyxy[0].cpu().numpy()[:, :4]
    #     scores = results.xyxy[0].cpu().numpy()[:, 4]
    #     classes = results.xyxy[0].cpu().numpy()[:, 5].astype(np.int32)

    #     # Load the class names from the file
    #     with open('./yolo_v5/custom_classes.txt', 'r') as lf:
    #         class_names = [cname.strip() for cname in lf.readlines()]
        
    #     # Draw the bounding boxes and labels on the frame
    #     for box, score, class_idx in zip(boxes, scores, classes):
    #         self.x1, self.y1, self.x2, self.y2 = map(int, box)
    #         class_name = class_names[class_idx]
    #         cv2.rectangle(webcam_frame, (self.x1,self.y1), (self.x2, self.y2), (0, 255, 0), 2)
    #         cv2.putText(webcam_frame, f"{class_name}: {score:.2f}", (self.x1, self.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    #         # Print the coordinates of the detected object
    #         print(f"{class_name} coordinates: ({self.x1}, {self.y1}), ({self.x2}, {self.y2})")

    #     return self.x1,self.y1, self.x2, self.y2 #change to self

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
        logging.info(f' Trying to connect to {self.RTSP_URL}...')
        # Connect to RTSP URL
        cap = cv2.VideoCapture(self.RTSP_URL, cv2.CAP_FFMPEG)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret : continue
            # w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # print('width', w, 'height',h)
            self.frame = frame
            
            if not ret:
                print("ERROR !")
                break
            
            if QR_toggle != 0:
                prev_code_info = self.code_info
                self.code_info = self.read_QRcodes(frame)
                if self.code_info != prev_code_info:
                    QR_toggle = 0
                    break
            if yolo_toggle != 0:
                self.x1, self.y1, self.x2, self.y2 = self.yolo_detection(frame)
            if BF_toggle != 0:
                self.BF_data = self.save_BF()
                BF_toggle = 0
                break
                
            # cv2.imshow('test',frame)
            # if cv2.waitKey(1) == ord('q'):
            #     break
        cap.release()
        # cv2.destroyAllWindows()


# class BatcamNoise:
#     def __init__(self):
#         pass

#     def Batcam_BF(self, toggle=0):
#         if toggle == 1:
#             ws = websocket.WebSocketApp(
#                 f"ws://{BATCAM_IP}:80/ws",
#                 on_open=on_open,
#                 on_message=on_message,
#                 on_error=on_error,
#                 on_close=on_close,
#                 subprotocols=["subscribe"],
#                 header={"Authorization": f"Basic %s" % base64EncodedAuth}
#             )
#             ws.run_forever(dispatcher=rel, reconnect=5)
#             rel.signal(2, rel.abort)
#             rel.dispatch()


if __name__ == "__main__":
    Batcam = BatCam()
    # Batcam.save_BF()  
    # Batcam.leakage_detection()
    try:
        Batcam.rtsp_to_opencv(QR_toggle = 0, yolo_toggle=0, BF_toggle=0)
    except Exception as error:
        logging.error(error)