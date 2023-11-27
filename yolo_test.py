import cv2
import torch
from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.general import select_device, smart_inference_mode
from utils.torch_utils import scale_boxes

class YoloDetector:
    def __init__(self, weights='yolov5s.pt', device='', dnn=False, data='coco128.yaml', fp16=False,
                 imgsz=[640], conf_thres=0.25, iou_thres=0.45, max_det=1000,
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

    # Example usage with a video file
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bounding_box_frame, bounding_box_centers = detector.run(frame)


if __name__ == '__main__':
    main()
