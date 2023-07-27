# FennecBot
FennecBot prototype development

### SetUP
1. For Every Boot
    * CAN comm with NUC <-> Motor
        ```
        rosrun scout_bringup bringup_can2usb.bash
 
        # Check CAN comm
        candump can0
        ```
        
2. For Dynamixel Port
    * Dynamixel port ttyUSB0 allow
        ```
        sudo chmod 666 /dev/ttyUSB0
