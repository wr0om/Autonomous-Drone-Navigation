# -*- coding: utf-8 -*-
#
#  ...........       ____  _ __
#  |  ,-^-,  |      / __ )(_) /_______________ _____  ___
#  | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  | / ,..´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#     +.......   /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  MIT Licence
#
#  Copyright (C) 2023 Bitcraze AB
#

"""
file: pid_controller.py

A simple PID controller for the Crazyflie
ported from pid_controller.c in the c-based controller of the Crazyflie
in Webots
"""

import numpy as np

class VelocityHeightPIDController:
    def __init__(self):
        # Initialize error variables for velocity, altitude, pitch, and roll
        self.previous_velocity_x_error = 0.0
        self.previous_velocity_y_error = 0.0
        self.previous_altitude_error = 0.0
        self.previous_pitch_error = 0.0
        self.previous_roll_error = 0.0

        # Initialize the integral component for altitude control
        self.altitude_integral = 0.0

        # Define PID control gains for various control aspects
        self.control_gains = {
            "attitude_yaw_kp": 1, "attitude_yaw_kd": 0.5, 
            "attitude_rp_kp": 0.5, "attitude_rp_kd": 0.1,
            "velocity_xy_kp": 2, "velocity_xy_kd": 0.5, 
            "altitude_kp": 10, "altitude_ki": 5, "altitude_kd": 5
        }

    def compute_pid(self, time_delta, target_vx, target_vy, target_yaw_rate, target_altitude, 
                    current_roll, current_pitch, current_yaw_rate, current_altitude, current_vx, current_vy):
        """
        Compute PID outputs for controlling velocity and maintaining fixed height.
        """

        # Calculate the error in x-direction velocity
        error_vx = target_vx - current_vx
        # Calculate the derivative of the x-direction velocity error
        derivative_vx = (error_vx - self.previous_velocity_x_error) / time_delta

        # Calculate the error in y-direction velocity
        error_vy = target_vy - current_vy
        # Calculate the derivative of the y-direction velocity error
        derivative_vy = (error_vy - self.previous_velocity_y_error) / time_delta

        # Calculate the desired pitch using velocity error and derivative in x-direction
        pitch_target = self.control_gains["velocity_xy_kp"] * np.clip(error_vx, -1, 1) + self.control_gains["velocity_xy_kd"] * derivative_vx
        # Calculate the desired roll using velocity error and derivative in y-direction
        roll_target = -self.control_gains["velocity_xy_kp"] * np.clip(error_vy, -1, 1) - self.control_gains["velocity_xy_kd"] * derivative_vy

        # Update the previous velocity errors for the next iteration
        self.previous_velocity_x_error = error_vx
        self.previous_velocity_y_error = error_vy

        # Calculate the error in altitude
        altitude_error = target_altitude - current_altitude
        # Calculate the derivative of the altitude error
        altitude_derivative = (altitude_error - self.previous_altitude_error) / time_delta
        # Update the integral of the altitude error
        self.altitude_integral += altitude_error * time_delta
        # Calculate the altitude control output
        altitude_output = (self.control_gains["altitude_kp"] * altitude_error + 
                           self.control_gains["altitude_kd"] * altitude_derivative +
                           self.control_gains["altitude_ki"] * np.clip(self.altitude_integral, -2, 2) + 48)

        # Update the previous altitude error for the next iteration
        self.previous_altitude_error = altitude_error

        # Calculate the error in pitch
        error_pitch = pitch_target - current_pitch
        # Calculate the derivative of the pitch error
        derivative_pitch = (error_pitch - self.previous_pitch_error) / time_delta
        # Calculate the error in roll
        error_roll = roll_target - current_roll
        # Calculate the derivative of the roll error
        derivative_roll = (error_roll - self.previous_roll_error) / time_delta

        # Calculate the roll control output
        roll_output = self.control_gains["attitude_rp_kp"] * np.clip(error_roll, -1, 1) + self.control_gains["attitude_rp_kd"] * derivative_roll
        # Calculate the pitch control output
        pitch_output = self.control_gains["attitude_rp_kp"] * np.clip(error_pitch, -1, 1) + self.control_gains["attitude_rp_kd"] * derivative_pitch
        # Calculate the yaw control output
        yaw_output = self.control_gains["attitude_yaw_kp"] * np.clip(target_yaw_rate - current_yaw_rate, -1, 1)

        # Update the previous pitch and roll errors for the next iteration
        self.previous_pitch_error = error_pitch
        self.previous_roll_error = error_roll

        # Calculate motor commands based on the control outputs
        motor1 = np.clip(altitude_output - roll_output - pitch_output + yaw_output, 0, 600)
        motor2 = np.clip(altitude_output - roll_output + pitch_output - yaw_output, 0, 600)
        motor3 = np.clip(altitude_output + roll_output + pitch_output + yaw_output, 0, 600)
        motor4 = np.clip(altitude_output + roll_output - pitch_output - yaw_output, 0, 600)

        # Return the calculated motor commands
        return [motor1, motor2, motor3, motor4]
