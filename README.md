# FennecBot
FennecBot prototype development!!!
: ) !!!!!


# For us, For every boot
rosrun scout_bringup bringup_can2usb.bash

# check CAN comm
candump can0



# At first time
sudo modprobe gs_usb
rosrun scout_bringup setup_can2usb.bash
rosrun scout_bringup bringup_can2usb.bash
candump can0
roslaunch scout_bringup scout_robot_base.launch 