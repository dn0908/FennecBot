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
        ##### BATCAM RTSP_URL = "rtsp://<username>:<password>@<ip>:8544/raw
        self.BATCAM_IP = '192.168.2.2'
        self.RTSP_URL = "rtsp:/192.168.2.2:8554/raw"

        ##### QR DETECTION : CODE INFO CONFIGURATION #####
        self.qr_x = 0 # QR position data
        self.qr_y = 0
        self.code_info : str= "" # QR code info

        ##### OBJECT DETECTION : CUSTOM YOLOv5 MODEL CONFIGURATION #####
        self.yolo_model =  torch.hub.load('/home/smi/FennecBot', 'custom', source ='local', path='1106_2_best.pt',force_reload=True) ### The repo is stored locally
        self.classes = self.yolo_model.names ### class names in string
        self.lpt_idx = 0

        ##### NOISE DETECTION : DEEP LEARNING MODEL CONFIGURATIONS #####
        self.noise_model = keras.models.load_model('./models/model.h5')
        # self.noise_model.summary()
        self.sr = 96000
        self.window_size_sec = 0.05  # Window size in seconds
        self.window_size = int(self.window_size_sec / (512 / self.sr))
        self.stride_sec = self.window_size_sec/2.0  # Stride length in seconds
        self.stride = int(self.stride_sec / (512 / self.sr))
        self.noise_detection = 0
        self.predicted_prob = 0

    def read_QRcodes(self, frame):
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


    # Yolo Detection. Detect multiple bboxes in frame, return self.yolo_list
    def yolo_detect(self, frame):
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = [frame]
        results = self.yolo_model(frame)
        labels, coordinates = results.xyxyn[0][:,-1], results.xyxyn[0][:,:-1]
        return labels, coordinates
    
    def yolo_plot(self, results, frame):
        # Storing results
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        self.yolo_list = []

        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        print(f"[INFO] Total {n} detections. . . ")


        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:  # Confidence threshold
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                class_name = self.classes[int(labels[i])]

                yolo_x = (x1 + x2) / 2
                yolo_y = (y1 + y2) / 2
                
                # # For Bounding box visualization
                # # Un Comment for visualization
                # if class_name == 'Flange':
                #     color = (255, 0, 0)
                #     # Draw rectangles and text
                #     cv2.rectangle(frame, (x1, y1), (x2, y2),color, 2)
                #     cv2.rectangle(frame, (x1, y1-20), (x2, y1), color, -1)
                #     cv2.putText(frame, class_name + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
                # elif class_name == 'Flush Ring':
                #     color = (0, 255, 0)
                #     # Draw rectangles and text
                #     cv2.rectangle(frame, (x1, y1), (x2, y2),color, 2)
                #     cv2.rectangle(frame, (x1, y1-20), (x2, y1), color, -1)
                #     cv2.putText(frame, class_name + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
                # elif class_name == 'GasRegulator':
                #     color = (0, 0, 255)
                #     # Draw rectangles and text
                #     cv2.rectangle(frame, (x1, y1), (x2, y2),color, 2)
                #     cv2.rectangle(frame, (x1, y1-20), (x2, y1), color, -1)
                #     cv2.putText(frame, class_name + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
                # elif class_name == 'Nuts':
                #     color = (0, 255, 255)
                #     # Draw rectangles and text
                #     cv2.rectangle(frame, (x1, y1), (x2, y2),color, 2)
                #     cv2.rectangle(frame, (x1, y1-20), (x2, y1), color, -1)
                #     cv2.putText(frame, class_name + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
                # elif class_name == 'Piston Valve':
                #     color = (255, 255, 0)
                #     # Draw rectangles and text
                #     cv2.rectangle(frame, (x1, y1), (x2, y2),color, 2)
                #     cv2.rectangle(frame, (x1, y1-20), (x2, y1), color, -1)
                #     cv2.putText(frame, class_name + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
                # elif class_name == 'Pressure Gage':
                #     color = (255, 0, 255)
                #     # Draw rectangles and text
                #     cv2.rectangle(frame, (x1, y1), (x2, y2),color, 2)
                #     cv2.rectangle(frame, (x1, y1-20), (x2, y1), color, -1)
                #     cv2.putText(frame, class_name + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

                result = {
                    "class_name": class_name,
                    "x_coordinate": yolo_x,
                    "y_coordinate": yolo_y
                }
                self.yolo_list.append(result)

        self.yolo_list = sorted(self.yolo_list, key=lambda item: (item["x_coordinate"], item["y_coordinate"])) # sort yolo list before return
        return frame, self.yolo_list


    # Save Listening Point Data for 32 counts to CSV file
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

        # originally 1.5 sec for 32 counts.
        # BUT occasionally error due to BATCAM. so .4 sec for buffer.
        # Reduce buffer if more than one csv saved
        time.sleep(1.9)
        count_num = 0
        ws.close()
        print("WebSocket Closed, Count Num to zero...")


    def calc_l_point(self, x_cent, y_cent):
        """
        Change 640 x 480 frame coordinate to 40x30 L point map coordinate

        Args:
        x_center: frame point x coordinate (0 to 639)
        y_center: frame point y coordinate (0 to 479)

        Returns:
        l_point_index : 40x30 L point map coordinate (0 to 1199)

        """
        # Mapping frame to L point map
        Lmap_x = int((x_cent + 1) / 16)   # 1 to 40
        Lmap_y = int((y_cent + 1) / 16)      # 1 to 30
        # print('Lmap X', Lmap_x, 'Lmap Y', Lmap_y)

        # Mapping L point map to L point index
        if Lmap_y == 0:
            if Lmap_x == 0:
                self.lpt_indx = Lmap_x
            else:
                self.lpt_indx = abs(Lmap_x - 1)
        elif Lmap_y > 0 :
            self.lpt_indx = (int(Lmap_y-1)*40) + int(Lmap_x-1)

        return self.lpt_indx
    
    def calc_l_map(self, lpt):
        """
        Change L point index to 40x30 L point map x,y coordinates

        Args:
        lpoint_index : L point index (0 to 1199)

        Returns:
        lmap_x : 40x30 L point map X coordinate (0 to 40)
        lmap_y : 40x30 L point map Y coordinate (0 to 40)

        """
        if lpt < 40:
            lmap_y = 1
            lmap_x = lpt + 1

        elif lpt >= 40 :
            lmap_x = (lpt - (40 * int(lpt/40))) + 1
            lmap_y = int(lpt/40) + 1

        return lmap_x, lmap_y


    # Change Listening Point by Index
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


    # Leakage Detection (Deep Learning Model, Predict)
    def leakage_detection(self):
        def extract_features_v4(file_path, window_size, stride):
            data = pd.read_csv(file_path)
            audio = data.values.reshape(-1)

            # Perform STFT on the entire audio
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            print("stft size: ", stft.shape)
            num_time_steps = stft.shape[1] # Calculate the number of time steps

            features = []
            for i in range(0, num_time_steps, stride):
                # Determine the start and end indices for the window
                time_start = i
                time_end = i + window_size

                # Check if the window exceeds the audio signal length
                if time_end > num_time_steps:
                    break

                stft_window = stft[:, time_start:time_end] # Extract features for each time step
                stft_window = np.transpose(stft_window) # Transpose the features to have time as the first axis
                features.append(stft_window) # Append the features to the list

            return features

        X_train = []
        folder_path = '.'
        file_path = glob.glob(f'{folder_path}/*.csv')
        file_path = max(file_path, key= os.path.getmtime) # get the latest csv file
        print("✅ Loading Model..... Reading file", file_path)
        
        class_features = extract_features_v4(file_path, self.window_size, self.stride)
        X_train.extend(class_features)
        X_train = np.array(X_train)

        y_pred_prob = self.noise_model.predict(X_train) # Predict probabilities on test data
        # print('y_pred_prob : ', y_pred_prob)
        y_pred = np.argmax(y_pred_prob, axis=1) # Convert probabilities to class labels
        # print("y_pred :",y_pred)
        y_pred_mean = np.mean(y_pred)
        # print("y_pred_mean : ", y_pred_mean)

        prob_0 = float(np.mean(y_pred_prob[:, 0]))
        prob_1 = float(np.mean(y_pred_prob[:, 1]))
        prob_2 = float(np.mean(y_pred_prob[:, 2]))

        # current y_pred_mean thresh : 0.5
        if y_pred_mean >= 0.5 :      # if detected, self.noise_detection changes to 1
            print('⚠ Leakage Detected ! @', file_path, 'score :', y_pred_mean)
            self.predicted_prob = prob_1 + prob_2 # probability for leakage
            self.noise_detection = 1
        else :                       # if not, self.noise_detection remains 0
            print('NO Leakage Detected @', file_path)
            # self.predicted_prob = prob_0
            self.predicted_prob = 0 # probability 0 for no leakage
            self.noise_detection = 0
        
        return self.noise_detection, self.predicted_prob


    def rtsp_to_opencv(self, QR_toggle = 0, yolo_toggle = 0, BF_toggle = 0):
        fpsLimit = 1 # limitq
        startTime = time.time()
        logging.info(f' Trying to connect to {self.RTSP_URL}...')

        cap = cv2.VideoCapture(self.RTSP_URL, cv2.CAP_FFMPEG)

        while True:
            ret, frame = cap.read()
            if np.any(frame) == None : # for batcam error...
                print('frame NONE but continue')
                pass
            
            frame = cv2.resize(frame, (640, 480)) #resize cap for model input

            if not ret:
                print("Failed to grab frame.")
                continue

            nowTime = time.time()
            if (nowTime - startTime) >= fpsLimit:
                self.frame = frame #?
                
                if QR_toggle != 0:
                    prev_code_info = self.code_info
                    self.code_info = self.read_QRcodes(frame)
                    if self.code_info != prev_code_info:
                        QR_toggle = 0
                        break

                if yolo_toggle != 0:
                    yolo_results = self.yolo_detect(frame)
                    frame, yolo_list_ = self.yolo_plot(yolo_results, frame)
                    break

                if BF_toggle != 0:
                    self.BF_data = self.save_BF()
                    # self.leakage_detection()
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

    try:
        Batcam.rtsp_to_opencv(QR_toggle = 0, yolo_toggle = 0, BF_toggle=1)
    except Exception as error:
        logging.error(error)