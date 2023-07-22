import cv2
import logging
import numpy as np
from websocket_client import *
# import torch
# import pyzbar.pyzbar as pyzbar

import subprocess
from datetime import datetime

import requests
import json
import base64

logging.basicConfig(level=logging.INFO)

class BatCam:
    def __init__(self):
        # BATCAM RTSP_URL = "rtsp://<username>:<password>@<ip>:8544/raw
        self.RTSP_URL = "rtsp://admin:admin@192.168.1.3:8554/raw"

        # Load the "custom" YOLOv5 model
        self.model = torch.hub.load('./temp_object_detection/yolov5', 'custom', path='./temp_object_detection/yolov5/runs/train/exp2/weights/best.pt', source='local')
        self.x1, self.y1, self.x2, self.y2 = 0, 0, 0, 0
        self.BF_data = []
        self.code_info : str= ""
        self.FullScan_arr = []
        self.frame = []
        # self.on_message = []

    def read_QRcodes(self, frame):
        codes = pyzbar.decode(frame)
        for code in codes:
            x, y , w, h = code.rect
            self.code_info = code.data.decode('utf-8')
            # make bounding box around code
            cv2.rectangle(frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
            # display info text
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, self.code_info, (x + 6, y - 6), font, 2.0, (255, 255, 255), 1)
            # print(self.code_info)
        
        return self.code_info
        
    def yolo_detection(self, webcam_frame):
        # Pass the frame through the YOLOv5 model
        results = self.model(webcam_frame)

        # Extract the boxes, scores, and classes from the results
        boxes = results.xyxy[0].cpu().numpy()[:, :4]
        scores = results.xyxy[0].cpu().numpy()[:, 4]
        classes = results.xyxy[0].cpu().numpy()[:, 5].astype(np.int32)

        # Load the class names from the file
        with open('./temp_object_detection/custom_classes.txt', 'r') as lf:
            class_names = [cname.strip() for cname in lf.readlines()]
        
        # Draw the bounding boxes and labels on the frame
        for box, score, class_idx in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[class_idx]
            cv2.rectangle(webcam_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(webcam_frame, f"{class_name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Print the coordinates of the detected object
            print(f"{class_name} coordinates: ({x1}, {y1}), ({x2}, {y2})")

        return x1, y1, x2, y2 #change to self
        
    
    def show_BF(self):
        ws = websocket.WebSocketApp(
            f"ws://{CAMERA_IP}:80/ws",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            subprotocols=["subscribe"],
            header={"Authorization": f"Basic %s" % base64EncodedAuth}
        )
        ws.run_forever(dispatcher=rel, reconnect=5)
        rel.signal(2, rel.abort)
        rel.dispatch()

        return ws
            
    def rtsp_to_opencv(self, QR_toggle = 0, yolo_toggle = 0, BF_toggle = 0):
        logging.info(f' Trying to connect to {self.RTSP_URL}...')
        # Connect to RTSP URL
        cap = cv2.VideoCapture(self.RTSP_URL)

        while cap.isOpened():
            ret, frame = cap.read()
            self.frame = frame
            
            if not ret:
                print("ERROR !")
                break
            
            if QR_toggle != 0:
                self.code_info = self.read_QRcodes(frame)
            if yolo_toggle != 0:
                self.x1, self.y1, self.x2, self.y2 = self.yolo_detection(frame)
            if BF_toggle != 0:
                self.BF_data = self.show_BF()
                
                
            cv2.imshow('test',frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


class BatCamNoise:
    def __init__(self):
        # BATCAM RTSP_URL = "rtsp://<username>:<password>@<ip>:8544/raw
        self.RTSP_URL = "rtsp://admin:admin@192.168.1.3:8554/raw"

        # output file name = current date & time
        # self.noise_output_file = "output_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"

        # command = ["ffmpeg", "-i", self.RTSP_URL, "-vn", "-acodec", "pcm_s16le", "-ar", "44100",
        #            "-ac", "2", "-filter_complex", "[0:a]channelsplit=channel_layout=stereo:channels=FL[right]"
        #              "-map", "[right]", "-t", "5", self.noise_output_file]  # set sampling time as 5 seconds
        # subprocess.call(command)
    
    def save_noise_raw(self):
        # SAVING 2 CHANNELS BOTH
        self.output_file_ch1 = "output_ch1_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
        self.output_file_ch2 = "output_ch2_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"

        command = ["ffmpeg", "-i", self.RTSP_URL, "-vn", "-acodec", "pcm_s16le",
                   "-ar", "44100",
                   "-ac", "2",
                    "-filter_complex",
                    "[0:a]channelsplit=channel_layout=stereo:channels=FL[left][right]", # split channel left & right
                    "-map", "[left]", "-t", "5", self.output_file_ch1,
                    "-map", "[right]", "-t", "5", self.output_file_ch2]  # set sampling time as 5 seconds (both channels)
        
        subprocess.call(command)

class ChangeLPoint:
    def __init__(self, user, password, camera_ip):
        self.user = user
        self.password = password
        self.camera_ip = camera_ip # enter !!! each specifications!!!

    def get_base64_encoded_auth(self):
        credential = f"{self.user}:{self.password}"
        return base64.b64encode(credential.encode()).decode()

    def get_api_host(self):
        return f"http://{self.camera_ip}"

    def get_url(self):
        return self.get_api_host() + "/beamforming/setting"

    def get_headers(self):
        base64EncodedAuth = self.get_base64_encoded_auth()
        return {
            "Authorization": f"Basic {base64EncodedAuth}"
        }

    def change_index_l(self, section_val):
        global index_l
        '''
        sections for listening points (40 x 30)
        ----------------------------
        |  1  |  2   |  3   |  4   |
        ----------------------------
        |  5  |  6   |  7   |  8   |
        ----------------------------
        |  9  |  10  |  11  |  12  |
        ----------------------------
        '''
        if section_val == "1":
            index_l = 205
            print("section 1 selected to listen !")
        elif section_val == "2":
            index_l = 216
            print("section 2 selected to listen !")
        elif section_val == "3":
            index_l = 227
            print("section 3 selected to listen !")
        elif section_val == "4":
            index_l = 238
            print("section 4 selected to listen !")
        ########################################
        elif section_val == "5":
            index_l = 645
            print("section 5 selected to listen !")
        elif section_val == "6":
            index_l = 656
            print("section 6 selected to listen !")
        elif section_val == "7":
            index_l = 667
            print("section 7 selected to listen !")
        elif section_val == "8":
            index_l = 678
            print("section 8 selected to listen !")
        ########################################
        elif section_val == "9":
            index_l = 1085
            print("section 9 selected to listen !")
        elif section_val == "10":
            index_l = 1096
            print("section 10 selected to listen !")
        elif section_val == "11":
            index_l = 1107
            print("section 11 selected to listen !")
        elif section_val == "12":
            index_l = 1119
            print("section 12 selected to listen !")

        camera_settings = {
            "low_cut": 2000,
            "high_cut": 45000,
            "gain": 100,
            "distance": 10,
            "x_cal": -0.02,
            "y_cal": 0.02,
            "index_l_channel": 0, # This could be 0 or 1
            "index_l": index_l
            # Two points of Listening Point, 
            # if you send index_l_channel to 0, you can get sound of 20th point,
            # if you send index_l_channel to 1, you can get sound of 580th point.
        }
        print("Requesting body: " + str(camera_settings) + "\n") 
        try:
            current_setting = requests.get(self.get_url(), headers=self.get_headers())
            print("Current Camera setting: " + str(current_setting.json()) + "\n")
            response = requests.patch(self.get_url(), headers=self.get_headers(), data=camera_settings)
            print("Result: " + str(response.json()) + "\n")
        except Exception as ex:
            print(ex)

        # FOR USE
        # user = "admin"
        # password = "admin"
        # camera_ip = "192.168.2.2"

        # camera = ChangeLPoint(user, password, camera_ip)

        # input_value = input("Enter input value (A/B/C): ")
        # camera.change_index_l(1) # for test


    # original file to import data from websocket
    # def Batcam_BF(self, toggle=0):
    #     if toggle == 1:
    #         ws = websocket.WebSocketApp(
    #             f"ws://{CAMERA_IP}:80/ws",
    #             on_open=on_open,
    #             on_message=on_message,
    #             on_error=on_error,
    #             on_close=on_close,
    #             subprotocols=["subscribe"],
    #             header={"Authorization": f"Basic %s" % base64EncodedAuth}
    #         )
    #         ws.run_forever(dispatcher=rel, reconnect=5)
    #         rel.signal(2, rel.abort)
    #         rel.dispatch()


if __name__ == "__main__":
    # Batcam = BatCam()/
    BatcamNoise = BatCamNoise()
    # try:
    #     Batcam.rtsp_to_opencv(QR_toggle = 0, yolo_toggle=0, BF_toggle=0)
    # except Exception as error:
    #     logging.error(error)

    # FOR USE
    user = "admin"
    password = "admin"
    camera_ip = "192.168.2.2"
    index_l = 0

    camera = ChangeLPoint(user, password, camera_ip)

    section_val = input("Enter section value : ")
    camera.change_index_l(1) # for test