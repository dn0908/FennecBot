# # For us, For every boot
# rosrun scout_bringup bringup_can2usb.bash

# # check CAN comm
# candump can0

# dynamicel ttyUSB0 allow
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
        # self.Pantilt = PanTilt()
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
                    self.Scoutmini.move_hard(0.1, 0, sleep = 5)
                    Task = 1
                elif Task == 1: # Find target point
                    # self.Pantilt.MotorController(pan_angle= 0, tilt_angle= 0)
                    Task = 2
                elif Task == 2: # Collect BF data
                    # self.Batcam.rtsp_to_opencv(BF_toggle=1)
                    
                    # Save data from self.Batcam.BF_data
                    # self.Batcam.save_BF(BF_toggle=1)
                    
                    Task = 0
                    self.Task = 'B'
                
            if self.task == 'B': # Local_QR
                if Task == 0: # Move to B position
                    linear_velocity, angular_velocity = self.Realsense.cal_Linetracing(contours, frame) 
                    self.Scoutmini.move(linear_velocity, angular_velocity)
                    
                    if self.Realsense.read_QRcodes(frame) == 'B':
                        self.Scoutmini.move(0, 0)
                        # self.Pantilt.MotorController(pan_angle= 0, tilt_angle= 0)
                        Task = 1
                        
                elif Task == 1: # Find target point
                    self.Batcam.rtsp_to_opencv(QR_toggle=1)
                    # self.Pantilt.Move2Target(self.Batcam.frame)
                    if self.Batcam.code_info == 'something':
                        Task = 2
                elif Task == 2: # Collect BF data
                    # self.Batcam.rtsp_to_opencv(BF_toggle=1)

                    # Save data from self.Batcam.BF_data
                    # self.Batcam.save_BF(BF_toggle=1)
                    
                    Task = 0
                    self.Task = 'C'

            if self.task == 'C': # Local_soft
                if Task == 0: # Move to C position
                    linear_velocity, angular_velocity = self.Realsense.cal_Linetracing(contours, frame) 
                    self.Scoutmini.move(linear_velocity, angular_velocity)
                    
                    if self.Realsense.read_QRcodes(frame) == 'C':
                        self.Scoutmini.move(0, 0)
                        # self.Pantilt.MotorController(pan_angle= 0, tilt_angle= 0)
                        Task = 1
                        
                elif Task == 1: # Find target point
                    self.Batcam.rtsp_to_opencv(yolo_toggle=1)
                    if self.Batcam.x1 != 0:
                        target_pos = [self.Batcam.x1, self.Batcam.y1, self.Batcam.x2, self.Batcam.y2]
                        # self.Pantilt.Move2Target(self.Batcam.frame)
                        Task = 2
                elif Task == 2: # Collect BF data
                    self.Batcam.rtsp_to_opencv(BF_toggle=1)

                    # Save data from self.Batcam.BF_data
                    # self.Batcam.save_BF(BF_toggle=1)
                    
                    Task = 0
                    self.Task = 'D'
                    
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
                        
                        # Save data from self.Batcam.BF_data
                        # self.Batcam.save_BF(BF_toggle=1)
                    
                    Task = 0
                    self.Task = 'E'
                    
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


            cv2.imshow('RealSense', mask)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break

        self.pipeline.stop()
        cv2.destroyAllWindows()


if __name__=="__main__":
    Maincontroller = MainController()
    Maincontroller.main_action()
    # Maincontroller.main_task()


    