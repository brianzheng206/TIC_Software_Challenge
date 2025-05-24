from TMMC_Wrapper import *
import rclpy
import numpy as np
import math
import time
from ultralytics import YOLO

# Variable for controlling which level of the challenge to test -- set to 0 for pure keyboard control
challengeLevel = 3

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
        offset_angle = 15.0
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

        distance_threshold = 10.0   # how far to search for any obstacle/sign
        stop_distance      = 1.0    # how close to stop
        cone_center        = 0.0    # straight ahead
        cone_half_angle    = 15.0   # ±15° vision cone
        approaching        = False

        while rclpy.ok():
            # 1) grab camera image & LIDAR scan
            img = camera.rosImg_to_cv2()
            stop_detected, x1, y1, x2, y2 = camera.ML_predict_stop_sign(img)
            scan = lidar.checkScan()
            min_dist, _ = lidar.detect_obstacle_in_cone(
                scan,
                distance=distance_threshold,
                center=cone_center,
                offset_angle=cone_half_angle
            )

            if stop_detected and not approaching:
                # sign spotted → enter “approach” mode
                print("[Level 2] Stop sign detected → driving in")
                approaching = True
                control.send_cmd_vel(0.0, 0.0)
                time.sleep(0.1)

            if approaching:
                if min_dist > stop_distance or min_dist == -1:
                    # still too far (or nothing in cone): drive forward
                    control.set_cmd_vel(0.1, 0.0, duration=0.8)
                else:
                    # within 1 m → stop and restore manual
                    control.send_cmd_vel(0.0, 0.0)
                    print(f"[Level 2] Stopped at {min_dist:.2f} m from sign.")
                    approaching = False

            # keep timers, subscribers, and keyboard thread alive
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)
            
    if challengeLevel == 3:
        print("Starting autonomous loop...")

        start_time = time.time()
        delivery_completed = False
        stop_sign_cooldown = 0  # To prevent multiple stops for same sign

        while rclpy.ok() and not delivery_completed:
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)

            # 1. LIDAR check for obstacle in front
            scan = lidar.checkScan()
            min_dist, _ = lidar.detect_obstacle_in_cone(scan, distance=0.4, center=0, offset_angle=30)
            if min_dist != -1:
                print("Obstacle detected! Avoiding...")
                control.set_cmd_vel(0.0, 0.0, 0.0)
                control.rotate(30, 1)  # turn left
                time.sleep(1)
                continue  # skip rest and spin again

            # 2. Camera check for stop sign every 0.5s
            if time.time() - stop_sign_cooldown > 5:
                img = camera.rosImg_to_cv2()
                stop_detected, x1, y1, x2, y2 = camera.ML_predict_stop_sign(img)
                if stop_detected:
                    print("Stop sign detected!")
                    control.set_cmd_vel(0.0, 0.0, 0.0)
                    time.sleep(3)
                    stop_sign_cooldown = time.time()
                    continue

            # 3. Move forward
            control.send_cmd_vel(0.1, 0.0)  # move forward at low speed

            # 4. Loop exit condition (you define: AprilTag or time-based exit)
            # For now, after 60s we assume loop done
            if time.time() - start_time > 60:
                delivery_completed = True

        control.set_cmd_vel(0.0, 0.0)
        total_time = time.time() - start_time
        print(f"✅ Delivery completed in {total_time:.2f} seconds!")
            

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
