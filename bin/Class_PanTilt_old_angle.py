import cv2
import numpy as np
from dynamixel.DynamixelSDK.python.src.dynamixel_sdk import *  # Dynamixel SDK library (protocol 2.0)


# dynamicel ttyUSB0 allow
# sudo chmod 666 /dev/ttyUSB0


class PanTilt:
    def __init__(self):

        # Batcam center pixels
        self.center_x = 320
        self.center_y = 240

        # Pan-tilt angles and control step

        # ID1 (틸트)
        # 아래 방향 1950
        # 반대 방향 3900
        # ID0 (팬)
        # 우측 -2150
        # 좌측 -90

        # self.pan_position = 2960
        # self.tilt_position = 2300

        self.pan_angle = ((-1500/4096)*360)
        self.tilt_angle = ((3000/4096)*360)

        # originally....
        # self.pan_angle = 90
        # self.tilt_angle = 90

        self.step_size = 0.5 # step of position size needs to be 10...

        # Threshold error - center & max value
        self.error_threshold = 10

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

    # Define function to convert angle -> position value
    def angle_to_position(self, angle):
        return int((angle/360)*4095)

    # Define function to convert position value -> angle
    def position_to_angle(self, position):
        return int(position/4095)*360


    # Define function to control the motors using dynamixel sdk
    def MotorController(self, pan_angle, tilt_angle):
        # Convert angles to position values
        pan_position = self.angle_to_position(pan_angle)
        tilt_position = self.angle_to_position(tilt_angle)

        # Write goal positions to motors
        self.write_goal_position(self.PAN_ID, pan_position)
        self.write_goal_position(self.TILT_ID, tilt_position)

        # Wait until motors reach goal positions
        while True:
            # Read present positions from motors
            pan_present_position = self.read_present_position(self.PAN_ID)
            tilt_present_position = self.read_present_position(self.TILT_ID)

            pan_present_angle = self.position_to_angle(pan_present_position)
            tilt_present_angle = self.position_to_angle(tilt_present_position)

            # Print current angles of the motors
            print("[ID:%03d] GoalAngle:%03d PresentAnlge:%03d [ID:%03d] GoalAngle:%03d PresentAngle:%03d" \
                % (self.PAN_ID, pan_angle, pan_present_angle, self.TILT_ID, tilt_angle, tilt_present_angle))
            
            # Print current angles of the motors
            # print("[ID:%03d] GoalAngle:%03d PresentAnlge:%03d [ID:%03d] GoalAngle:%03d PresentAngle:%03d" \
            #     % (self.PAN_ID, pan_angle, pan_present_angle, self.TILT_ID, tilt_angle, tilt_present_angle))

            # # Check if the motors are close enough to the goal positions
            # if not abs(pan_position - pan_present_position) > DXL_MINIMUM_POSITION_VALUE_FOR_MOVING \
            # and not abs(tilt_position - tilt_present_position) > DXL_MINIMUM_POSITION_VALUE_FOR_MOVING:
            #     break

            # use threshold value instead of DXL_MINIMUM_POSITION_VALUE_FOR_MOVING
            if abs(pan_present_position - pan_position) <= self.error_threshold and abs(tilt_present_position - tilt_position) <= self.error_threshold:
                break

    def Move2Target(self, BatCamframe):
            NoiseArr = BatCamframe
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(NoiseArr)
            cv2.circle(NoiseArr, max_loc, 10, (0, 0, 255), 2)

            # Calculate the error between the center and the maximum value
            error_x = self.center_x - max_loc[0]
            error_y = self.center_y - max_loc[1]

            if (abs(error_x) > self.error_threshold) or (abs(error_y) > self.error_threshold):
                if error_x > 0:
                    self.pan_angle += self.step_size
                else:
                    self.pan_angle -= self.step_size
                if error_y > 0:
                    self.tilt_angle += self.step_size
                else:
                    self.tilt_angle -= self.step_size

                # Pan angle limitation to [50, 130]
                self.pan_angle = max(50, min(130, self.pan_angle))
                # Limit the tilt angle to [0, 70]
                self.tilt_angle = max(0, min(70, self.tilt_angle))
                self.MotorController(self.pan_angle, self.tilt_angle)

    def close_port(self):
        # disable torque for all motors
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, BROADCAST_ID, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        # Close port
        self.portHandler.closePort()
        

if __name__=="__main__":
    Pantilt = PanTilt()

    # for motor moving teset
    
    # Pantilt.MotorController((200),((3000/4096)*360))
    # time.sleep(2)
    # Pantilt.MotorController(210,(270))
    # time.sleep(1)

#     Pantilt.Move2Target()

    