import numpy as np
import math

class Drone:
    def __init__(self, 
                 model_name, 
                 x, y, z,
                 min_RPM, max_RPM, hover_RPM,
                 max_horizontal_speed=15.0,  
                 max_vertical_speed=5.0,
                 vertical_decel_distance=15, vertical_accel_distance=10,
                 horiz_decel_distance=15, horiz_accel_distance=10):
        
        self.model_name = model_name
        self.rpm_values = np.array([min_RPM]*4, dtype=float)
        self.position = np.array([x, y, z], dtype=float)
        self.velocity = np.zeros(3, dtype=float)

        self.min_RPM = min_RPM
        self.max_RPM = max_RPM

        self.horizontal_speed = max_horizontal_speed
        self.max_horizontal_speed = max_horizontal_speed  

        self.vertical_speed = max_vertical_speed
        self.max_vertical_speed = max_vertical_speed

        self.hover_rpm = hover_RPM
        
        self.pitch = 0.0
        self.yaw = 0.0

        self.vertical_decel_distance = vertical_decel_distance
        self.vertical_accel_distance = vertical_accel_distance

        self.horiz_decel_distance = horiz_decel_distance
        self.horiz_accel_distance = horiz_accel_distance

        self.target_history = []

    def set_position(self, x, y, z):
        self.position = np.array([x, y, z], dtype=float)
        
    def _calculate_acceleration_factors(self, target, min_vel_factor = 0.2):
        # Vertical factors
        vertical_error = target[2] - self.position[2]
        last_vertical_error = self.target_history[-1][2] - self.position[2]

        vertical_decel_factor = min((abs(vertical_error) / self.vertical_decel_distance) + min_vel_factor, 1.0)
        vertical_accel_factor = min(((abs(last_vertical_error) + 1) / self.vertical_accel_distance) + min_vel_factor, 1.0)
        vertical_total_acceleration_factor = vertical_decel_factor * vertical_accel_factor

        # Horizontal factors
        horizontal_distance = np.linalg.norm(target[:2] - self.position[:2])
        last_horizontal_distance = np.linalg.norm(self.target_history[-1][:2] - self.position[:2])

        horizontal_decel_factor = min((horizontal_distance / self.horiz_decel_distance) + min_vel_factor, 1.0)
        horizontal_accel_factor = min(((last_horizontal_distance + 1) / self.horiz_accel_distance) + min_vel_factor, 1.0)
        horizontal_total_acceleration_factor = horizontal_decel_factor * horizontal_accel_factor

        return vertical_total_acceleration_factor, horizontal_total_acceleration_factor

    def update_rpms(self, target, dt, vertical_total_acceleration_factor, horizontal_total_acceleration_factor, rpm_update_gain = 50):
        v_speed_factor = self.vertical_speed / self.max_vertical_speed
        vertical_component = self.hover_rpm + (self.max_RPM - self.hover_rpm) * vertical_total_acceleration_factor * v_speed_factor * np.sign(target[2] - self.position[2])

        h_speed_factor = self.horizontal_speed / self.max_horizontal_speed
        horizontal_component = (self.max_RPM - self.min_RPM) * horizontal_total_acceleration_factor * h_speed_factor

        desired_avg_rpm = vertical_component + horizontal_component * 0.13
        desired_avg_rpm = np.clip(desired_avg_rpm, self.min_RPM, self.max_RPM)
        
        max_rpm_change = 600 * dt

        for i in range(4):
            error = desired_avg_rpm - self.rpm_values[i]
            rpm_adjustment = rpm_update_gain * error * dt
            rpm_adjustment = np.clip(rpm_adjustment, -max_rpm_change, max_rpm_change)
            self.rpm_values[i] += rpm_adjustment
            self.rpm_values[i] = np.clip(self.rpm_values[i], self.min_RPM, self.max_RPM)
    
    def update_physics(self, target, dt, vertical_total_acceleration_factor, horizontal_total_acceleration_factor):
        horizontal_target = target[:2]
        horizontal_position = self.position[:2]
        horizontal_direction = horizontal_target - horizontal_position
        horizontal_distance = np.linalg.norm(horizontal_direction)

        if horizontal_distance > 0:
            horizontal_velocity = self.horizontal_speed * horizontal_total_acceleration_factor * (horizontal_direction / horizontal_distance)
            horizontal_step = horizontal_velocity * dt
            if np.linalg.norm(horizontal_step) > horizontal_distance:
                horizontal_position = horizontal_target
            else:
                horizontal_position += horizontal_step

            self.position[:2] = horizontal_position
            self.velocity[:2] = horizontal_velocity
        else:
            self.velocity[:2] = 0

        vertical_error = target[2] - self.position[2]
        if vertical_error != 0:
            desired_vertical_velocity = self.vertical_speed * vertical_total_acceleration_factor * np.sign(vertical_error)
            vertical_step = desired_vertical_velocity * dt

            if abs(vertical_step) > abs(vertical_error):
                self.position[2] = target[2]
                self.velocity[2] = 0
            else:
                self.position[2] += vertical_step
                self.velocity[2] = desired_vertical_velocity
        else:
            self.velocity[2] = 0

        self.position[2] = max(self.position[2], 0)
    
    def update_control(self, target, dt):
        target_position = target[:3]
        target_h_speed = target[3]
        target_v_speed = target[4]
        self.horizontal_speed = target_h_speed
        self.vertical_speed = target_v_speed

        vertical_total_acceleration_factor, horizontal_total_acceleration_factor = self._calculate_acceleration_factors(target_position)
        self.update_rpms(target_position, dt, vertical_total_acceleration_factor, horizontal_total_acceleration_factor)
        
        direction = target_position - self.position
        horizontal_distance = math.sqrt(direction[0]**2 + direction[1]**2)
        if np.linalg.norm(direction) > 0:
            self.yaw = math.atan2(direction[1], direction[0])
            self.pitch = math.atan2(-direction[2], horizontal_distance)
        
        self.update_physics(target_position, dt, vertical_total_acceleration_factor, horizontal_total_acceleration_factor)
        return self.pitch, self.yaw, self.rpm_values.copy(), self.position.copy(), self.velocity.copy()
    
    def to_dict(self):
        return {
            'model_name': self.model_name,
            'min_RPM': self.min_RPM,
            'max_RPM': self.max_RPM,
            'hover_rpm': self.hover_rpm,
            'max_horizontal_speed': self.max_horizontal_speed,
            'max_vertical_speed': self.max_vertical_speed,
            'vertical_decel_distance': self.vertical_decel_distance,
            'vertical_accel_distance': self.vertical_accel_distance,
            'horiz_decel_distance': self.horiz_decel_distance,
            'horiz_accel_distance': self.horiz_accel_distance,
        }