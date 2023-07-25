# FennecBot
FennecBot prototype development

### SetUP
1. For Every Boot
    * ROS run
        ```
        sudo modprobe gs_usb
        rosrun scout_bringup setup_can2usb.bash
        rosrun scout_bringup bringup_can2usb.bash
        roslaunch scout_bringup scout_robot_base.launch 
        ```
    * Check CAN comm
        ```    
        candump can0
        ```
        
2. For Dynamixel Port
    * ttyUSB0 allow
        ```
        sudo chmod 666 /dev/ttyUSB0
