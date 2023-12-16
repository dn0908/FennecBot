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

import glob
import os

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
        # from yolov5.models.experimental import attempt_load # Now import attempt_load
        # self.yolo_model = attempt_load('/home/smi/FennecBot/1106_2_best.pt') # Load the "custom" YOLOv5 model
        self.x1, self.y1, self.x2, self.y2 = 0, 0, 0, 0
        self.class_name : str= ""
        self.yolo_model =  torch.hub.load('/home/smi/FennecBot', 'custom', source ='local', path='1106_2_best.pt',force_reload=True) ### The repo is stored locally
        self.class_names = self.yolo_model.names ### class names in string

        self.central_point = {
            'Flange': [],
            'Flush Ring': [],
            'GasRegulator': [],
            'Nuts': [],
            'Piston Valve': [],
            'Pressure Gage': []
            }

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
        # frame = cv2.resize(frame, (1600, 1200))
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

    # WORKING ON ........

    # def plot_boxes(self, results, frame):
    #     # Storing results
    #     self.yolo_list = []

    #     labels, cord = results
    #     n = len(labels)
    #     x_shape, y_shape = frame.shape[1], frame.shape[0]

    #     print(f"[INFO] Total {n} detections. . . ")
    #     buffer = []
    #     for i in range(n):
    #         row = cord[i]
    #         if row[4] >= 0.25:  # Confidence threshold
    #             x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
    #             text_d = self.class_names[int(labels[i])]

    #             color = (0, 255, 0)  # Default color
    #             if text_d == 'Flange':
    #                 color = (0, 255, 0)
    #             elif text_d == 'Flush Ring':
    #                 color = (0, 0, 255)
    #             elif text_d == 'GasRegulator':
    #                 color = (0, 0, 255)
    #             elif text_d == 'Nuts':
    #                 color = (0, 255, 255)
    #             elif text_d == 'Piston Valve':
    #                 color = (255, 255, 0)
    #             elif text_d == 'Pressure Gage':
    #                 color = (255, 0, 255)

    #             # Draw rectangles and text
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    #             cv2.rectangle(frame, (x1, y1-20), (x2, y1), color, -1)
    #             cv2.putText(frame, f"{text_d} {row[4]:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    #             x_mean = (x1+x2)/2
    #             y_mean = (y1+y2)/2
    #             buffer.append((x_mean, y_mean))
    #             self.central_point[text_d].append(buffer)


    #         labels, cord = results
    #         n = len(labels)
    #         x_shape, y_shape = frame.shape[1], frame.shape[0]

    #         print(f"[INFO] Total {n} detections. . . ")
    #         buffer = []
    #         for i in range(n):
    #             row = cord[i]
    #             if row[4] >= 0.25:  # Confidence threshold
    #                 x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
    #                 text_d = self.class_names[int(labels[i])]
            
    #                 x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
    #                 class_name = self.class_names[int(labels[i])]
    #                 yolo_x = (x1 + x2) / 2
    #                 yolo_y = (y1 + y2) / 2

    #                 result = {
    #                     "class_name": class_name,
    #                     "x_coordinate": yolo_x,
    #                     "y_coordinate": yolo_y
    #                 }
    #                 self.yolo_list.append(result)

    #                 # # Drawing bounding boxes (optional)
    #                 # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #                 # cv2.putText(frame, f"{class_name}: {coord[4]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    #     return frame, self.central_point
        




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

        # Dynamically get the latest CSV file in the specified folder
        folder_path = '/home/smi/FennecBot/'
        # file_path = 'l_point_data.csv'
        file_path = glob.glob(f'{folder_path}/*.csv')
        file_path = max(file_path, key= os.path.getmtime)
        print("✅ Loading Model..... Reading file", file_path)
        
        class_features = extract_features_v4(file_path, self.window_size, self.stride)
        X_train.extend(class_features)
        X_train = np.array(X_train)
        # print('Noise Input Data Shape : ',X_train.shape)

        y_pred_prob = self.noise_model.predict(X_train) # Predict probabilities on test data
        y_pred = np.argmax(y_pred_prob, axis=1) # Convert probabilities to class labels
        # print('Predicted Noise Class : ', y_pred)
        y_pred_mean = np.mean(y_pred)
        # print('Predicted Noise Class MEAN : ', y_pred_mean)

        if y_pred_mean >= 0.3 :      # if detected, self.noise_detection changes to 1
            print('⚠ Leakage Detected ! @', file_path, 'score :', y_pred_mean)
            self.noise_detection = 1
        else :                       # if not, self.noise_detection remains 0
            print('NO Leakage Detected @', file_path)
            self.noise_detection = 0
        
        return self.noise_detection  # return self.noise_detection


    def rtsp_to_opencv(self, QR_toggle = 0, yolo_toggle = 0, BF_toggle = 0):
        fpsLimit = 1 # limitq
        startTime = time.time()
        logging.info(f' Trying to connect to {self.RTSP_URL}...')

        cap = cv2.VideoCapture(self.RTSP_URL, cv2.CAP_FFMPEG)

        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 480)) #resize cap for model input
            

            if not ret:
                print("Failed to grab frame.")
                continue

            nowTime = time.time()
            if (nowTime - startTime) >= fpsLimit:
                self.frame = frame #?
                
                if QR_toggle != 0:
                    frame = cv2.resize(frame, (640, 480))
                    prev_code_info = self.code_info
                    self.code_info = self.read_QRcodes(frame)
                    if self.code_info != prev_code_info:
                        QR_toggle = 0
                        break

                if yolo_toggle != 0:
                    frame = cv2.resize(frame, (640, 480))
                    # self.x1, self.y1, self.x2, self.y2, self.class_name = self.yolo_detection(frame)
                    # frame = self.yolo_detection(frame)
                    # self.yolo_list = self.multiple_yolo_detection(frame)
                    results = self.detect_(frame)
                    frame = self.plot_boxes(results, frame)

                    # break

                if BF_toggle != 0:
                    self.BF_data = self.save_BF()
                    self.leakage_detection()
                    BF_toggle = 0
                    break

                startTime = time.time() # reset time

              
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
        Batcam.rtsp_to_opencv(QR_toggle = 0, yolo_toggle = 0, BF_toggle=1)
    except Exception as error:
        logging.error(error)