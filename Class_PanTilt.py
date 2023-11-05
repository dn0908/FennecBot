import cv2
import numpy as np
from dynamixel.DynamixelSDK.python.src.dynamixel_sdk import *  # Dynamixel SDK library (protocol 2.0)


# dynamicel ttyUSB0 allow
# sudo chmod 666 /dev/ttyUSB0


class PanTilt:
    def __init__(self):

        # # Batcam center pixels
        # self.center_x = 800
        # self.center_y = 600
        # Reshape !
        self.center_x = 320
        self.center_y = 240

        # Pan-tilt angles and control step

        # ID1 (틸트)
        # 아래 방향 1950
        # 반대 방향 3900
        # ID0 (팬)
        # 우측 -2150
        # 좌측 -90

        # init positions
        self.pan_position = 1955 #2960
        self.tilt_position = 2100 #2300

        self.step_size = 10 # step of position size needs to be 10... or vibration too high

        # Threshold error - center & max value
        self.error_threshold = 50

        ###################### DYNAMIXEL SDK INIT ######################
        
        self.MY_DXL                      = 'X_SERIES'        # OUR DYNAMIXEL : XM430-W350 -- changed
        self.BAUDRATE                    = 57600
        self.DEVICENAME                  = "/dev/ttyUSB0"    # Check your port name
        self.PROTOCOL_VERSION            = 2.0               # Use protocol 2.0 for dynamixel motors

        self.TORQUE_ENABLE               = 1                 # Value for enabling the torque
        self.TORQUE_DISABLE              = 0                 # Value for disabling the torque

        self.ADDR_TORQUE_ENABLE          = 64
        self.ADDR_GOAL_POSITION          = 116
        self.LEN_GOAL_POSITION           = 4                 # Data Byte Length
        self.ADDR_PRESENT_POSITION       = 132
        self.LEN_PRESENT_POSITION        = 4                 # Data Byte Length

        self.PAN_ID                      = 0                 # Dynamixel#1 ID : 0
        self.TILT_ID                     = 1                 # Dynamixel#1 ID : 1

        # Initialize port handler and packet handler
        self.portHandler = PortHandler(self.DEVICENAME)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        # Open port and set baudrate
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            quit()

        if self.portHandler.setBaudRate(self.BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            quit()


        # Enable torque on motors
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.PAN_ID, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE) # Add closing parenthesis and store return values
        if dxl_comm_result != COMM_SUCCESS: # Check communication result
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0: # Check error code
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.TILT_ID, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE) # Add closing parenthesis and store return values
        if dxl_comm_result != COMM_SUCCESS: # Check communication result
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0: # Check error code
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))


    # Define function to write goal position to motor
    def write_goal_position(self, id, position):
        # catch any exceptions
        try:
            self.packetHandler.write4ByteTxRx(self.portHandler, id, self.ADDR_GOAL_POSITION, position)
        except Exception as e:
            print("Error while writing goal position:", e)
        # dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, id, self.ADDR_GOAL_POSITION, position)
        # if dxl_comm_result != COMM_SUCCESS:
        #     print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        # elif dxl_error != 0:
        #     print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        # self.packetHandler.write4ByteTxRx(self.portHandler, id, self.ADDR_GOAL_POSITION, position)

    # Define function to read present position from motor
    def read_present_position(self, id):
        dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, id, self.ADDR_PRESENT_POSITION)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        return dxl_present_position
        # return self.packetHandler.read4ByteTxRx(self.portHandler, id, self.ADDR_PRESENT_POSITION)


    # Define function to control the motors using dynamixel sdk
    def MotorController(self, pan_position, tilt_position):

        # pan_position = self.pan_position
        # tilt_position = self.tilt_position

        # Write goal positions to motors
        self.write_goal_position(self.PAN_ID, pan_position)
        self.write_goal_position(self.TILT_ID, tilt_position)

        # Wait until motors reach goal positions
        while True:
            # Read present positions from motors
            pan_present_position = self.read_present_position(self.PAN_ID)
            tilt_present_position = self.read_present_position(self.TILT_ID)

            # Print current positions of the motors
            print("[ID:%03d] GoalPos:%03d PresentPos:%03d [ID:%03d] GoalPos:%03d PresentPos:%03d" \
                % (self.PAN_ID, pan_position, pan_present_position, self.TILT_ID, tilt_position, tilt_present_position))
            
            # use threshold value instead of DXL_MINIMUM_POSITION_VALUE_FOR_MOVING
            if abs(pan_present_position - pan_position) <= self.error_threshold and abs(tilt_present_position - tilt_position) <= self.error_threshold:
                break

    def Move2Target(self, target_x=0, target_y=0):
            # Read present positions from motors
            pan_present_position = self.read_present_position(self.PAN_ID)
            tilt_present_position = self.read_present_position(self.TILT_ID)

            pan_position = pan_present_position
            tilt_position = tilt_present_position

            # pix2ang_constant = 340 / 800
            pix2ang_constant = 340 / 320
            error_x = (self.center_x - target_x)*pix2ang_constant
            error_y = (self.center_y - target_y)*pix2ang_constant

            if (abs(error_x) > self.error_threshold/3) or (abs(error_y) > self.error_threshold/3):
                if error_x > 0:
                    pan_position += abs(error_x)
                else:
                    pan_position -= abs(error_x)
                if error_y > 0:
                    tilt_position += abs(error_y)
                else:
                    tilt_position -= abs(error_y)

                # Pan position limitation to [920, 3980]
                pan_position = max(920, min(3980, pan_position))

                # Tilt position limitation to [2300, 2600]
                tilt_position = max(2300, min(2600, tilt_position))

                # control motors
                self.MotorController(int(pan_position), int(tilt_position))

                

    def close_port(self):
        # disable torque for all motors
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, BROADCAST_ID, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        # Close port
        self.portHandler.closePort()
        

    def Turn(self, dir = ""):
        if dir == "front":
            # goal pos to FRONT
            pan_position = 1940 #1940
            tilt_position = 2300 #2300

            self.write_goal_position(self.PAN_ID, pan_position)
            self.write_goal_position(self.TILT_ID, tilt_position)
            while True:
                pan_present_position = self.read_present_position(self.PAN_ID)
                tilt_present_position = self.read_present_position(self.TILT_ID)
                print("[ID:%03d] GoalPos:%03d PresentPos:%03d [ID:%03d] GoalPos:%03d PresentPos:%03d" \
                    % (self.PAN_ID, pan_position, pan_present_position, self.TILT_ID, tilt_position, tilt_present_position))
                if abs(pan_present_position - pan_position) <= self.error_threshold and abs(tilt_present_position - tilt_position) <= self.error_threshold:
                    break

        if dir == "right":
            # goal pos to RIGHT
            pan_position = 1460 #920
            tilt_position = 2300 #2300

            self.write_goal_position(self.PAN_ID, pan_position)
            self.write_goal_position(self.TILT_ID, tilt_position)
            while True:
                pan_present_position = self.read_present_position(self.PAN_ID)
                tilt_present_position = self.read_present_position(self.TILT_ID)
                print("[ID:%03d] GoalPos:%03d PresentPos:%03d [ID:%03d] GoalPos:%03d PresentPos:%03d" \
                    % (self.PAN_ID, pan_position, pan_present_position, self.TILT_ID, tilt_position, tilt_present_position))
                if abs(pan_present_position - pan_position) <= self.error_threshold and abs(tilt_present_position - tilt_position) <= self.error_threshold:
                    break

        if dir == "left":
            # goal pos to LEFT
            pan_position = 2350 #3650 ,2960
            tilt_position = 2100 #2300

            self.write_goal_position(self.PAN_ID, pan_position)
            self.write_goal_position(self.TILT_ID, tilt_position)
            while True:
                pan_present_position = self.read_present_position(self.PAN_ID)
                tilt_present_position = self.read_present_position(self.TILT_ID)
                print("[ID:%03d] GoalPos:%03d PresentPos:%03d [ID:%03d] GoalPos:%03d PresentPos:%03d" \
                    % (self.PAN_ID, pan_position, pan_present_position, self.TILT_ID, tilt_position, tilt_present_position))
                if abs(pan_present_position - pan_position) <= self.error_threshold and abs(tilt_present_position - tilt_position) <= self.error_threshold:
                    break

        if dir == "taskB":
            # goal pos to LEFT
            pan_position = 1500 #3650 ,2960 #############################################
            tilt_position = 2300 #2300 #############################################

            self.write_goal_position(self.PAN_ID, pan_position)
            self.write_goal_position(self.TILT_ID, tilt_position)
            while True:
                pan_present_position = self.read_present_position(self.PAN_ID)
                tilt_present_position = self.read_present_position(self.TILT_ID)
                print("[ID:%03d] GoalPos:%03d PresentPos:%03d [ID:%03d] GoalPos:%03d PresentPos:%03d" \
                    % (self.PAN_ID, pan_position, pan_present_position, self.TILT_ID, tilt_position, tilt_present_position))
                if abs(pan_present_position - pan_position) <= self.error_threshold and abs(tilt_present_position - tilt_position) <= self.error_threshold:
                    break
        
        else :
            print("turn dir blank")

if __name__=="__main__":
    Pantilt = PanTilt()

    # PAN
    # 1940 front
    # 920  move 90 to right
    # 2960 move 90 to left
    # 3980 back

    # TILT
    # max  up  2600
    # max down 2300

    # for motor moving teset
    # Pantilt.MotorController(1800,2300)
    # Pantilt.Turn(dir = 'front')
    # time.sleep(2)
    # Pantilt.Turn(dir = 'left')
    # time.sleep(2)
    # Pantilt.Turn(dir = 'taskB')
    # time.sleep(2)
    # Pantilt.Turn(dir = 'right')
    # time.sleep(2)
    # Pantilt.Turn(dir = 'front')
    # Pantilt.MotorController(1940,2600)
    Pantilt.Move2Target(420,240)

    