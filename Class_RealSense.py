import cv2
import json
import numpy as np
import pyrealsense2 as rs 
import pyzbar.pyzbar as pyzbar
# from ar_markers import detect_markers


class RealSense:
    def __init__(self):
        self.code_info : str= ""
        # Set task flag
        self.flag = 'None'
        # Bring up realsense
        self.pipeline = rs.pipeline() 
        self.config = rs.config() 
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) 
        self.pipeline.start(self.config)
        # Config Camera
        self.lower_hsv = np.array([0, 115, 115]) # lower bound for detecting red objects
        self.upper_hsv = np.array([90, 255, 255]) # upper bound for detecting red objects
        self.camera_width = 640 
        self.camera_height = 480 
        self.camera_center_x = self.camera_width / 2 
        self.camera_center_y = self.camera_height / 2
        # Config Motor control PID gain
        self.Kp = 0.01 
        self.Ki = 0.0 
        self.Kd = 0.0 
        self.error_sum = 0 
        self.error_prev = 0

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
    
    def cal_Linetracing(self, contours, frame):
        linear_velocity, angular_velocity = 0, 0

        if len(contours) > 0: # if there are any contours found
            largest_contour = max(contours, key=cv2.contourArea) # find the largest contour by area
            moments = cv2.moments(largest_contour) # calculate the moments of the largest contour

            if moments["m00"] != 0: # if the area of the largest contour is not zero

                # calculate the y coordinate of the centroid of the largest contour
                centroid_x = int(moments["m10"] / moments["m00"]) 
                centroid_y = int(moments["m01"] / moments["m00"])

                # draw a green circle at the centroid on the frame
                cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1) 

                # calculate the error between the centroid x and the camera center x
                error = centroid_x - self.camera_center_x 
                error_deriv = error - self.error_prev 
                self.error_sum += error
                self.error_prev = error 
                pid_output = self.Kp * error + self.Ki * self.error_sum + self.Kd * error_deriv 
                pid_output = max(-1, min(1, pid_output)) # limit the PID output to [-1, 1]

                linear_velocity = 0.1 
                angular_velocity = pid_output * (-0.1) 

                # print("P: ", error, " AngVel: ", angular_velocity)
                # print("P: ", error, " I: ", error_sum, " D: ", error_deriv, " AngVel: ", angular_velocity)

            else: # if contour is found but is zero
                linear_velocity, angular_velocity = 0, 0
                print('Robot stop! No line detected') 
        else: # if there are no contours
            linear_velocity, angular_velocity = 0, 0
            print('Robot stop! No contours found')

        return linear_velocity, angular_velocity

    def stream(self):
        while True:
            frames = self.pipeline.wait_for_frames() 
            frame = np.asanyarray(frames.get_color_frame().get_data()) 
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv) 
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.imshow('RealSense', mask)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break

        self.pipeline.stop()
        cv2.destroyAllWindows()

if __name__=="__main__":
    Realsense = RealSense()
    Realsense.stream()
    # a, b = Realsense.cal_Linetracing()
    