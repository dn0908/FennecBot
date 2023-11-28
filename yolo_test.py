import torch
import cv2
import time


def detect_(frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

def plot_boxes(results, frame,classes):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")


    ### loop through detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55: ### threshold
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) #coordinates
            text_d = classes[int(labels[i])]


            if text_d == 'Flange':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1)
                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

            elif text_d == 'Flush Ring':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0,255), 2)
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 0,255), -1)
                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

            elif text_d == 'GasRegulator':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0,255), 2)
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 0,255), -1)
                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
            
            elif text_d == 'Nuts':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255,255), 2)
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,255), -1)
                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
            
            elif text_d == 'Piston Valve':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255,0), 2)
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (255, 255,0), -1)
                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

            elif text_d == 'Pressure Gage':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0,255), 2)
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (255, 0,255), -1)
                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

            ## print(row[4], type(row[4]),int(row[4]), len(text_d))

    return frame


print(f"[INFO] Loading model... ")
## loading the custom trained model
# model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection
model =  torch.hub.load('/home/smi/FennecBot', 'custom', source ='local', path='1106_2_best.pt',force_reload=True) ### The repo is stored locally

classes = model.names ### class names in string


RTSP_URL = "rtsp:/192.168.2.2:8554/raw"
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

while True:
    # start_time = time.time()
    ret, frame = cap.read()
    if ret :
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = detect_(frame, model = model)
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        frame = plot_boxes(results, frame,classes = classes)
        
        cv2.imshow("Batcam Yolo", frame)

        if cv2.waitKey(500) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
