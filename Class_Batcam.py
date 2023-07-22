import cv2
import logging
import numpy as np
from batcam.websocket_new import *
import torch
import pyzbar.pyzbar as pyzbar

logging.basicConfig(level=logging.INFO)

class BatCam:
    def __init__(self):
        # BATCAM RTSP_URL = "rtsp://<username>:<password>@<ip>:8544/raw
        self.RTSP_URL = "rtsp://admin:admin@{BATCAM_IP}:8554/raw"
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
    
    # def show_BF(self):
    #     ws = websocket.WebSocketApp(
    #         f"ws://{BATCAM_IP}:80/ws",
    #         on_open=on_open,
    #         on_message=on_message,
    #         on_error=on_error,
    #         on_close=on_close,
    #         subprotocols=["subscribe"],
    #         header={"Authorization": f"Basic %s" % base64EncodedAuth}
    #     )
    #     ws.run_forever(dispatcher=rel, reconnect=5)
    #     rel.signal(2, rel.abort)
    #     rel.dispatch()

    #     return ws

    def save_BF(self):
        ws = websocket.WebSocketApp(
            f"ws://{BATCAM_IP}:80/ws",
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
                self.BF_data = self.save_BF()
                
                
            cv2.imshow('test',frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


class BatcamNoise:
    def __init__(self):
        pass

    def Batcam_BF(self, toggle=0):
        if toggle == 1:
            ws = websocket.WebSocketApp(
                f"ws://{BATCAM_IP}:80/ws",
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


# if __name__ == "__main__":
#     Batcam = BatCam()
#     try:
#         Batcam.rtsp_to_opencv(QR_toggle = 0, yolo_toggle=0, BF_toggle=0)
#     except Exception as error:
#         logging.error(error)