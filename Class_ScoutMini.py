import pyagxrobots 
import time

class ScoutMini:
    def __init__(self):
        self.scout=pyagxrobots.pysdkugv.ScoutMiniBase()
        self.scout.EnableCAN() 

    def move(self, linear_velocity, angular_velocity):
        print("Robot move", linear_velocity, angular_velocity)
        self.scout.SetMotionCommand(linear_vel=linear_velocity, angular_vel=angular_velocity) 

    def move_hard(self, linear_velocity, angular_velocity, sleep = 0):
        self.scout.SetMotionCommand(linear_vel=linear_velocity, angular_vel=angular_velocity)
        time.sleep(sleep)
        self.scout.SetMotionCommand(linear_vel=0, angular_vel=0)


# if __name__=="__main__":
#     Scoutmini = ScoutMini()
#     Scoutmini.move()

    