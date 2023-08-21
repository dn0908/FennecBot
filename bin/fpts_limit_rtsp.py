import time
import cv2
import pyzbar.pyzbar as pyzbar
import torch

# model = torch.load('./yolo_v5/best.pt')
# device = torch.device('cpu')
# print(device)
# model.to(device)

code_info : str= ""
fpsLimit = 0 # limitq
startTime = time.time()
cap = cv2.VideoCapture("rtsp:/192.168.2.2:8554/raw", cv2.CAP_FFMPEG)
while True:
    ret, frame = cap.read()
    nowTime = time.time()
    if (nowTime - startTime) >= fpsLimit:
        


        # READ QR
        prev_code_info = code_info
        codes = pyzbar.decode(frame)
        print(codes)
        for code in codes:
            code_info = code.data.decode('utf-8')
            print("QR code info : ", code_info)
        if code_info != prev_code_info:
            QR_toggle = 0
            break

        startTime = time.time() # reset time
    
    cv2.imshow('batcam',frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()