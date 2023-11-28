### importing required libraries
import torch
import cv2
import time


### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame,classes):

    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels

    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")


    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55: ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]


            if text_d == 'Flange':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

            elif text_d == 'Nuts':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

            elif text_d == 'Flush Ring':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0,255), 2) ## BBox
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 0,255), -1) ## for text label background
                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

            elif text_d == 'GasRegulator':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0,255), 2) ## BBox
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 0,255), -1) ## for text label background
                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
            ## print(row[4], type(row[4]),int(row[4]), len(text_d))

    return frame


print(f"[INFO] Loading model... ")
## loading the custom trained model
# model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection
model =  torch.hub.load('/home/smi/FennecBot', 'custom', source ='local', path='1106_2_best.pt',force_reload=True) ### The repo is stored locally

classes = model.names ### class names in string format


# Initialize the webcam capture
RTSP_URL = "rtsp:/192.168.2.2:8554/raw"
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)




# assert cap.isOpened()
frame_no = 1

cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
while True:
    # start_time = time.time()
    ret, frame = cap.read()
    if ret :
        print(f"[INFO] Working with frame {frame_no} ")

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = detectx(frame, model = model)
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        frame = plot_boxes(results, frame,classes = classes)
        
        cv2.imshow("vid_out", frame)
        # if vid_out:
        #     print(f"[INFO] Saving output video. . . ")
        #     out.write(frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break
        frame_no += 1

print(f"[INFO] Clening up. . . ")
### releaseing the writer

## closing all windows
cv2.destroyAllWindows()
