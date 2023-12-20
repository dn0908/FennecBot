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
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import interpolate

class MainController:
    def __init__(self):
        self.task = 'D' # init task set to 'A'
        self.Scoutmini = ScoutMini()
        self.Realsense = RealSense()
        self.Pantilt = PanTilt()
        self.Batcam = BatCam()

        self.full_scan_data = []

    def main_action(self):
        while True:
            frames = self.Realsense.pipeline.wait_for_frames() 
            frame = np.asanyarray(frames.get_color_frame().get_data()) 
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.Realsense.lower_hsv, self.Realsense.upper_hsv) 
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
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
        Task = 1
        
        while True:
            frames = self.Realsense.pipeline.wait_for_frames() 
            frame = np.asanyarray(frames.get_color_frame().get_data()) 
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.Realsense.lower_hsv, self.Realsense.upper_hsv) 
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if self.task == 'A': # start
                if Task == 0: # Move to A position
                    print(f"Task {self.task} - subtask {Task} ongoing")
                    linear_velocity, angular_velocity = self.Realsense.cal_Linetracing(contours, frame) 
                    self.Scoutmini.move(linear_velocity, angular_velocity)
                    if self.Realsense.read_QRcodes(frame) == 'A':
                        self.Scoutmini.move(0, 0)
                        self.Pantilt.Turn(dir = 'front')
                        Task = 1
                elif Task == 1: # Find target point (in task A = front)
                    print(f"Task {self.task} - subtask {Task} ongoing")
                    self.Pantilt.Turn(dir="front")
                    Task = 2
                elif Task == 2: # Collect BF data
                    print(f"Task {self.task} - subtask {Task} ongoing")
                    self.Batcam.rtsp_to_opencv(BF_toggle=1)

                    os.chdir('/home/smi/FennecBotData/')
                    date_time_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    data_folder = "TaskA_local_" + date_time_now
                    os.makedirs(data_folder, exist_ok=True)
                    task_a_path = '/home/smi/FennecBotData/' + data_folder
                    os.chdir(task_a_path)

                    self.Batcam.rtsp_to_opencv(BF_toggle=1)
                    det_result, det_prob = self.Batcam.leakage_detection()

                    os.chdir('/home/smi/FennecBot/')

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
                        self.Pantilt.Turn(dir = 'taskB')
                        Task = 1
                elif Task == 1: # Find target point
                    print(f"Task {self.task} - subtask {Task} ongoing")
                    self.Batcam.rtsp_to_opencv(QR_toggle=1)
                    self.Pantilt.Move2Target(self.Batcam.qr_x, self.Batcam.qr_y)
                    
                    '''
                    pix2ang_constant = 340 / 640
                    error_x = (self.Pantilt.center_x - self.Batcam.qr_x)*pix2ang_constant
                    error_y = (self.Pantilt.center_y - self.Batcam.qr_y)*pix2ang_constan9494t
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
                
                elif Task == 2: # Collect BF data94
                    print(f"Task {self.task} - subtask {Task} ongoing")

                    os.chdir('/home/smi/FennecBotData/')
                    date_time_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    data_folder = "TaskB_QR_" + date_time_now
                    os.makedirs(data_folder, exist_ok=True)
                    qr_path = '/home/smi/FennecBotData/' + data_folder
                    os.chdir(qr_path)

                    self.Batcam.rtsp_to_opencv(BF_toggle=1)
                    det_result, det_prob = self.Batcam.leakage_detection()

                    os.chdir('/home/smi/FennecBot/')

                    Task = 0
                    self.task = 'C'
                    self.Pantilt.Turn(dir = 'front') # Batcam to front


            if self.task == 'C': # Local_soft
                if Task == 0: # Move to C position
                    print(f"Task {self.task} - subtask {Task} ongoing")
                    linear_velocity, angular_velocity = self.Realsense.cal_Linetracing(contours, frame) 
                    self.Scoutmini.move(linear_velocity, angular_velocity)
                    if self.Realsense.read_QRcodes(frame) == 'C':
                        self.Scoutmini.move(0, 0)
                        self.Pantilt.Turn(dir = 'front') # Batcam to front
                        # self.Pantilt.MotorController(pan_angle= 0, tilt_angle= 0)
                        Task = 1
                        
                elif Task == 1: # Find target point by YOLO
                    print(f"Task {self.task} - subtask {Task} ongoing")

                    os.chdir('/home/smi/FennecBotData/')
                    date_time_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    data_folder = "TaskC_yolo_" + date_time_now
                    os.makedirs(data_folder, exist_ok=True)
                    yolo_path = '/home/smi/FennecBotData/' + data_folder
                    os.chdir(yolo_path)

                    self.Batcam.rtsp_to_opencv(yolo_toggle=1)

                    results_length = len(self.Batcam.yolo_list)
                    print(f"[YOLOv5] 🚀 Total {results_length} boxes detected ...")

                    prev_x, prev_y = None, None  # for prev coord saving

                    for idx, result in enumerate(self.Batcam.yolo_list):
                        class_name = result["class_name"]
                        x_coordinate = result["x_coordinate"]
                        y_coordinate = result["y_coordinate"]

                        if idx == 0:
                            yolo_x = x_coordinate
                            yolo_y = y_coordinate
                            # target_pos = [x_coordinate, y_coordinate]
                            print(f"detected ID {idx + 1}: Class {class_name},X: {x_coordinate}, Y: {y_coordinate}")

                            indx = self.Batcam.calc_l_point(yolo_x, yolo_y)
                            self.Batcam.change_LPoint(indx)

                            self.Batcam.rtsp_to_opencv(BF_toggle=1) # save BF
                            det_result, det_prob = self.Batcam.leakage_detection() # leakage detection

                        else :
                            pre_x, pre_y = prev_x, prev_y
                            now_x, now_y = x_coordinate, y_coordinate
                            yolo_x = now_x - pre_x
                            yolo_y = now_y - pre_y
                            # target_pos = [yolo_x, yolo_y]
                            print(f"detected ID {idx + 1}: Class {class_name},X: {x_coordinate}, Y: {y_coordinate}")

                            indx = self.Batcam.calc_l_point(yolo_x, yolo_y)
                            self.Batcam.change_LPoint(indx)

                            self.Batcam.rtsp_to_opencv(BF_toggle=1) # save BF
                            det_result, det_prob = self.Batcam.leakage_detection() # leakage detection
                        
                        prev_x = x_coordinate
                        prev_y = y_coordinate
                    
                    os.chdir('/home/smi/FennecBot/')
                    Task = 0
                    self.task = 'D'
                    break
                    

            if self.task == 'D': # Local_fullscan
                if Task == 0: # Move to D position
                    linear_velocity, angular_velocity = self.Realsense.cal_Linetracing(contours, frame) 
                    self.Scoutmini.move(linear_velocity, angular_velocity)
                    
                    if self.Realsense.read_QRcodes(frame) == 'D':
                        self.Scoutmini.move(0, 0)
                        # self.Pantilt.MotorController(pan_angle= 0, tilt_angle= 0)
                        Task = 1
                        
                elif Task == 1: # Change L point & save csv all & predict each
                    
                    os.chdir('/home/smi/FennecBotData')

                    ##### FOR LPOINT CHANGING & FULL SCAN #####
                    date_time_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    data_folder = "TaskD_fullscan_" + date_time_now
                    os.makedirs(data_folder, exist_ok=True)
                    os.chdir(data_folder) # move into data folder
                    
                    # for point in range(1199): # for fullscan
                    for point in range(0, 1200, 70): # for scan by index step 10
                        print(f"Changing Listening Point to {point}")
                        self.Batcam.change_LPoint(point)
                        
                        time.sleep(5)  # Adjust the delay as needed

                        self.Batcam.rtsp_to_opencv(BF_toggle=1)
                        
                        lpt = point
                        Lmap_x, Lmap_y = self.Batcam.calc_l_map(lpt)
                        noise_detection, predicted_probability = self.Batcam.leakage_detection()
                        l_point_prob = {
                            "Lmap_x": Lmap_x,
                            "Lmap_y": Lmap_y,
                            "probability": predicted_probability
                        }
                        print(l_point_prob)
                        self.full_scan_data.append(l_point_prob)
                    
                    os.chdir('/home/smi/FennecBotData') # get out from data folder

                    fullscan_filename = data_folder + ".json" # dump as json file
                    with open(fullscan_filename, "w") as f:
                        json.dump(self.full_scan_data, f, indent=2)

                    os.chdir('/home/smi/FennecBot')
                    ############################################
                    Task = 2
                    

                elif Task == 2: # plot fullscan overlay map
                    # read json
                    folder_path = '/home/smi/FennecBotData/'
                    file_path = glob.glob(f'{folder_path}/*.json')
                    file_path = max(file_path, key= os.path.getmtime) # get the latest json file
                    print("✅ Loading Data..... Reading file", file_path)

                    data = json.load(open(file_path))

                    # convert data to numpy
                    x = np.array([d["Lmap_x"] for d in data])
                    y = np.array([d["Lmap_y"] for d in data])
                    p = np.array([d["probability"] for d in data])

                    # Interpolation
                    x_new = np.linspace(1, 40, 40)
                    y_new = np.linspace(1, 30, 30)
                    f = interpolate.interp2d(x, y, p, kind="linear")
                    p_new = f(x_new, y_new)

                    mapfilename = file_path.replace(".", "_")
                    mapfilename = mapfilename + '.png'

                    def on_key_press(event):
                        if event.key == "q":
                            mpimg.imsave(mapfilename, p_new, cmap='hsv')
                            plt.close()

                    plt.imshow(p_new, cmap="hsv")
                    plt.connect("key_press_event", on_key_press)
                    plt.show()
                        
                    os.chdir('/home/smi/FennecBot')

                    Task = 0
                    self.task == 'E'

                    
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