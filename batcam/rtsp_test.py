import cv2
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

#RTSP_URL = "rtsp://<username>:<password>@<ip>/axis-media/media.amp"
# RTSP_URL = "rtsp://user:user@192.168.0.1:8554/raw"
# RTSP_URL = "rtsp://user:user@127.0.0.53:8554/raw"
RTSP_URL = "rtsp://user:user@169.254.77.125:8554/raw"

def rtsp_to_opencv(url):
    logging.info(f' Try to connect to {url}')
    # Connect to RTSP URL
    
    cap = cv2.VideoCapture(url)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("err")
            break
        cv2.imshow('test',frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        rtsp_to_opencv(RTSP_URL)
    except Exception as error:
        logging.error(error)