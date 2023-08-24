# # For us, For every boot
# rosrun scout_bringup bringup_can2usb.bash

# # CHECK CAN comm
# candump can0

# # TO ALLOW DYNAMIXEL port ttyUSB0
# sudo chmod 666 /dev/ttyUSB0

from Class_RealSense import *
from Class_ScoutMini import *
from Class_PanTilt import *
from Class_Batcam import *

class MainController:
    def __init__(self):
        self.task = 'A'
        self.Scoutmini = ScoutMini()
        self.Realsense = RealSense()
        self.Pantilt = PanTilt()
        self.Batcam = BatCam()

    def main_action(self):
        while True:
            frames = self.Realsense.pipeline.wait_for_frames() 
            frame = np.asanyarray(frames.get_color_frame().get_data()) 
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.Realsense.lower_hsv, self.Realsense.upper_hsv) 
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            
            # self.flag = self.Realsense.detect_ARmarker(frame) # Detect AR marker
            
            # QRinfo = self.Realsense.read_QRcodes(frame) # Detect and Read QR code

            # linear_velocity, angular_velocity = self.Realsense.cal_Linetracing(contours, frame) # Calculate target velocity from Line
            
            # self.Scoutmini.move(linear_velocity, angular_velocity) # Move scoutmini

            # self.Pantilt.Move2Target(frame) # Move PanTilt
        
            self.Batcam.rtsp_to_opencv(yolo_toggle=1, BF_toggle=0, QR_toggle=1)



            # print("lin, Ang velocity: ", linear_velocity, " ", angular_velocity)
            # print("flag: ", self.flag, "QRinfo: ", QRinfo, "lin, Ang velocity: ", linear_velocity, " ", angular_velocity)
            
            cv2.imshow('RealSense', frame)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break

        self.pipeline.stop()
        cv2.destroyAllWindows()

    def main_task(self):
        '''
        * task D & E -> task specifications meeting needed
        * scan all -> in frame or by pantilt?
        
        |    TASK    |  Linetracking  |  AR  |  PanTilt |  QR  |  Yolo  | scan all |  BF  |
        ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
        |      A     |       x        |   x  |     x    |  x   |   x    |    x     |   o  | Local_hard
        |      B     |       o        |   o  |     o    |  o   |   x    |    x     |   o  | Local_QR
        |      C     |       o        |   o  |     o    |  x   |   o    |    x     |   o  | Local_soft
        |      D     |       o        |   o  |     o    |  x   |   o    |    o     |   o  | Local_fullscan
        |      E     |       o        |   o  |     o    |  o   |   o    |    o     |   o  | Full
        
        '''
        Task = 0
        
        while True:
            frames = self.Realsense.pipeline.wait_for_frames() 
            frame = np.asanyarray(frames.get_color_frame().get_data()) 
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.Realsense.lower_hsv, self.Realsense.upper_hsv) 
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            
            if self.task == 'A': # Local_hard
                if Task == 0: # Move to A position
                    self.Pantilt.Turn(dir = "front")
                    print(f"Task {self.task} - subtask {Task} ongoing")
                    self.Scoutmini.move_hard(0.1, 0, sleep = 5)
                    Task = 1
                elif Task == 1: # Find target point
                    print(f"Task {self.task} - subtask {Task} ongoing")
                    self.Pantilt.Turn(dir = 'left')
                    Task = 2
                elif Task == 2: # Collect BF data
                    print(f"Task {self.task} - subtask {Task} ongoing")
                    self.Batcam.rtsp_to_opencv(BF_toggle=1)
                    Task = 0
                    self.task = 'B'
                    self.Pantilt.Turn(dir = 'front') # Batcam to front


            if self.task == 'B': # Local_QR
                if Task == 0: # Move to B position
                    print(f"Task {self.task} - subtask {Task} ongoing")
                    linear_velocity, angular_velocity = self.Realsense.cal_Linetracing(contours, frame) 
                    self.Scoutmini.move(linear_velocity, angular_velocity)
                    if self.Realsense.read_QRcodes(frame) == 'B':
                        self.Scoutmini.move(0, 0)
                        self.Pantilt.Turn(dir = 'left')
                        Task = 1
                elif Task == 1: # Find target point
                    print(f"Task {self.task} - subtask {Task} ongoing")
                    self.Batcam.rtsp_to_opencv(QR_toggle=1)
                    self.Pantilt.Move2Target(self.Batcam.qr_x, self.Batcam.qr_y)
                    
                    
                    '''
                    pix2ang_constant = 340 / 800
                    error_x = (self.Pantilt.center_x - self.Batcam.qr_x)*pix2ang_constant
                    error_y = (self.Pantilt.center_y - self.Batcam.qr_y)*pix2ang_constant
                    step_size = 10
                    while(1):
                        pan_present_position = self.Pantilt.read_present_position(self.Pantilt.PAN_ID)
                        tilt_present_position = self.Pantilt.read_present_position(self.Pantilt.TILT_ID)
                        if (abs(pan_present_position-error_x) > self.Pantilt.error_threshold/3) or (abs(tilt_present_position-error_y) > self.Pantilt.error_threshold/3):
                            if error_x > 0: pan_present_position += step_size
                            else: pan_present_position -= step_size

                            if error_y > 0: tilt_present_position += step_size
                            else: tilt_present_position -= step_size
                            
                            self.Pantilt.Move2Target(pan_present_position, tilt_present_position)
                    '''
                    
                    if self.Batcam.code_info == 'B':
                        Task = 2
                elif Task == 2: # Collect BF data
                    print(f"Task {self.task} - subtask {Task} ongoing")
                    self.Batcam.rtsp_to_opencv(BF_toggle=1)
                    Task = 0
                    self.task = 'C'
                    self.Pantilt.Turn(dir = 'front') # Batcam to front


            if self.task == 'C': # Local_soft
                if Task == 0: # Move to C position
                    print(f"Task {self.task} - subtask {Task} ongoing")
                    break
                    linear_velocity, angular_velocity = self.Realsense.cal_Linetracing(contours, frame) 
                    self.Scoutmini.move(linear_velocity, angular_velocity)
                    if self.Realsense.read_QRcodes(frame) == 'C':
                        self.Scoutmini.move(0, 0)
                        # self.Pantilt.MotorController(pan_angle= 0, tilt_angle= 0)
                        Task = 1
                        
                elif Task == 1: # Find target point
                    print(f"Task {self.task} - subtask {Task} ongoing")
                    self.Batcam.rtsp_to_opencv(yolo_toggle=1)
                    if self.Batcam.x1 != 0:
                        target_pos = [self.Batcam.x1, self.Batcam.y1, self.Batcam.x2, self.Batcam.y2]
                        # self.Pantilt.Move2Target(self.Batcam.frame)
                        Task = 2
                elif Task == 2: # Collect BF data
                    print(f"Task {self.task} - subtask {Task} ongoing")
                    self.Batcam.rtsp_to_opencv(BF_toggle=1)
                    Task = 0
                    self.task = 'D'
                    

            if self.task == 'D': # Local_fullscan
                if Task == 0: # Move to D position
                    linear_velocity, angular_velocity = self.Realsense.cal_Linetracing(contours, frame) 
                    self.Scoutmini.move(linear_velocity, angular_velocity)
                    
                    if self.Realsense.read_QRcodes(frame) == 'D':
                        self.Scoutmini.move(0, 0)
                        # self.Pantilt.MotorController(pan_angle= 0, tilt_angle= 0)
                        Task = 1
                        
                elif Task == 1: # Find target point full scan
                    for i in range(0,100):
                        for j in range(0,50):
                            # self.Pantilt.MotorController(i, j)
                            self.Batcam.rtsp_to_opencv(yolo_toggle=1)
                            if self.Batcam.x1 != 0:
                                target_pos = [self.Batcam.x1, self.Batcam.y1, self.Batcam.x2, self.Batcam.y2]                        
                                self.Batcam.FullScan_arr.append(target_pos)
                                self.Batcam.x1, self.Batcam.y1, self.Batcam.x2, self.Batcam.y2 = 0,0,0,0
                    Task = 2
                    
                elif Task == 2: # Collect BF data
                    for target in self.Batcam.FullScan_arr:
                        # self.Pantilt.MotorController(pan_angle=target[0], tilt_angle=target[1])
                        self.Batcam.rtsp_to_opencv(BF_toggle=1)
                    
                    Task = 0
                    self.task = 'E'
                    
            if self.task == 'E':
                if Task == 0:
                    linear_velocity, angular_velocity = self.Realsense.cal_Linetracing(contours, frame) 
                    self.Scoutmini.move(linear_velocity, angular_velocity)
                    
                    if self.Realsense.read_QRcodes(frame) == 'E':
                        self.Scoutmini.move(0, 0)
                        # self.Pantilt.MotorController(pan_angle= 0, tilt_angle= 0)
                        Task = 1
                elif Task == 1:
                    


                    pass


            # cv2.imshow('RealSense', mask)
            # key = cv2.waitKey(10) & 0xFF
            # if key == ord('q'):
            #     break

        self.Realsense.pipeline.stop()
        # cv2.destroyAllWindows()


if __name__=="__main__":
    Maincontroller = MainController()
    # Maincontroller.main_action()
    Maincontroller.main_task()