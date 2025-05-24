from TMMC_Wrapper import *
import rclpy
import numpy as np
import math
import time
from ultralytics import YOLO

# Variable for controlling which level of the challenge to test -- set to 0 for pure keyboard control
challengeLevel = 1

# Set to True if you want to run the simulation, False if you want to run on the real robot
is_SIM = True

# Set to True if you want to run in debug mode with extra print statements, False otherwise
Debug = False

# Initialization    
if not "robot" in globals():
    robot = Robot(IS_SIM=is_SIM, DEBUG=Debug)
    
control = Control(robot)
camera = Camera(robot)
imu = IMU(robot)
logging = Logging(robot)
lidar = Lidar(robot)

if challengeLevel <= 2:
    control.start_keyboard_control()
    rclpy.spin_once(robot, timeout_sec=0.1)


try:
    if challengeLevel == 0:
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)
            # Challenge 0 is pure keyboard control, you do not need to change this it is just for your own testing

    if challengeLevel == 1:
        distance = 0.5
        center = 0.0
        offset_angle = 10.0
        backing_up = False

        while rclpy.ok():

            scan = lidar.checkScan()  
            min_dist, _ = lidar.detect_obstacle_in_cone(
                scan,
                distance=distance,
                center=center,
                offset_angle=offset_angle
            )                          

            if min_dist != -1:
                control.set_cmd_vel(-0.2, 0.0, duration=0.5)
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)



    if challengeLevel == 2:
        def detect_stop_sign_and_handle():
            # grab the most recent image (populated by Robot.image_listener_callback)
            img = camera.rosImg_to_cv2()
            stop_detected, x1, y1, x2, y2 = camera.ML_predict_stop_sign(img)

            if stop_detected:
                # immediately zero velocities
                control.send_cmd_vel(0.0, 0.0)
                # disable user keyboard inputs so they don't override
                control.stop_keyboard_input()

                # simulate a full stop
                time.sleep(3)
                control.start_keyboard_input()
                # optional: give a tiny forward nudge so you clear the junction
                control.send_cmd_vel(0.05, 0.0)
                time.sleep(1)
                control.send_cmd_vel(0.0, 0.0)

        # main spin loop
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)

            detect_stop_sign_and_handle()
            
    if challengeLevel == 3:
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)
            

    if challengeLevel == 4:
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)
            # Write your solution here for challenge level 4

    if challengeLevel == 5:
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)
            # Write your solution here for challenge level 5
            

except KeyboardInterrupt:
    print("Keyboard interrupt received. Stopping...")

finally:
    control.stop_keyboard_control()
    robot.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
