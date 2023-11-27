import cv2
import torch
import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5.utils.torch_utils import select_device, smart_inference_mode

# from ultralytics.utils.plotting import Annotator, colors

# from yolov5.models.common import DetectMultiBackend
# from yolov5.utils.general import select_device, smart_inference_mode
# from yolov5.utils.torch_utils import scale_boxes
# from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
#                            increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
import time

class YoloDetector:
    def __init__(self, weights='1106_2_best.pt', device='', dnn=False, data='data.yaml', fp16=False,
                 imgsz=[640], conf_thres=0.5, iou_thres=0.45, max_det=1000,
                 line_thickness=3, hide_labels=False, hide_conf=False, half=False, vid_stride=1):
        self.weights = weights
        self.device = device
        self.dnn = dnn
        self.data = data
        self.fp16 = fp16
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half
        self.vid_stride = vid_stride

        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data, fp16=fp16)

    @smart_inference_mode()
    def run(self, frame):
        im0 = frame.copy()

        # Inference
        im = torch.from_numpy(frame).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        # Run inference
        pred = self.model(im)

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=self.max_det)

        # Process predictions
        bounding_box_frame = im0.copy()
        bounding_box_center_coordinates_list = []

        for det in pred[0]:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, _, cls in reversed(det):
                    c = int(cls)
                    label = None if self.hide_labels else (self.model.names[c] if self.hide_conf else f'{self.model.names[c]}')
                    annotator = Annotator(bounding_box_frame, line_width=self.line_thickness, example=str(self.model.names))
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    # Calculate and store bounding box center coordinates
                    center_x = (xyxy[0] + xyxy[2]) / 2
                    center_y = (xyxy[1] + xyxy[3]) / 2
                    bounding_box_center_coordinates_list.append((center_x, center_y))

        return bounding_box_frame, bounding_box_center_coordinates_list

def main():
    detector = YoloDetector()

    RTSP_URL = "rtsp:/192.168.2.2:8554/raw"
    fpsLimit = 1 # limitq
    startTime = time.time()
    # logging.info(f' Trying to connect to {self.RTSP_URL}...')

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    while True:
        ret, frame = cap.read()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.transpose((2, 0, 1))
        frame = cv2.resize(frame, (640, 480)) #resize cap for model input
        if not ret:
            print("Failed to grab frame.")
            continue

        nowTime = time.time()
        if (nowTime - startTime) >= fpsLimit:

            bounding_box_frame, bounding_box_centers = detector.run(frame)
            print(bounding_box_centers)

            startTime = time.time() # reset time

            
        cv2.imshow('Batcam Capture',bounding_box_frame)
        if cv2.waitKey(500) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
