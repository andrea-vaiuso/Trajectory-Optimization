import numpy as np
import time
import math
import datetime
import csv
    
# ---------------- World Class ----------------

class World:
    def __init__(self, grid_size, max_world_size):
        self.grid_size = grid_size
        self.max_world_size = max_world_size
        self.grid = {}

    def get_area(self, x, y, z):
        return (x // self.grid_size, y // self.grid_size, z // self.grid_size)

    def set_area_parameters(self, x_1, x_2, y_1, y_2, parameters):
        for x in range(x_1, x_2 + 1, self.grid_size):
            for y in range(y_1, y_2 + 1, self.grid_size):
                for z in range(0, self.max_world_size * self.grid_size, self.grid_size):
                    area = self.get_area(x, y, z)
                    if area not in self.grid:
                        self.grid[area] = {}
                    self.grid[area].update(parameters)

    def get_area_parameters(self, x, y, z):
        area = self.get_area(x, y, z)
        return self.grid.get(area, {})

    def get_area_center_point(self, x, y, z):
        area = self.get_area(x, y, z)
        center = [(a * self.grid_size) + self.grid_size / 2 for a in area]
        return tuple(center)
    
    def get_areas_in_circle(self, x, y, height, radius):
        areas_in_circle = []
        parameters_in_circle = []
        radius_squared = radius ** 2
        for z in range(0, height * self.grid_size, self.grid_size):
            for i in range(max(0, x - radius), min(self.max_world_size * self.grid_size, x + radius + 1), self.grid_size):
                for j in range(max(0, y - radius), min(self.max_world_size * self.grid_size, y + radius + 1), self.grid_size):
                    if (i - x) ** 2 + (j - y) ** 2 <= radius_squared:
                        center = self.get_area_center_point(i, j, z)
                        parameters = self.get_area_parameters(i, j, z)
                        areas_in_circle.append(center)
                        parameters_in_circle.append(parameters)
        return areas_in_circle, parameters_in_circle
    
    def save_world(self, filename):
        with open(filename, 'w') as file:
            import json
            json.dump(self.grid, file)

    def load_world(self, filename):
        with open(filename, 'r') as file:
            import json
            self.grid = json.load(file)

# --------------- Simplified Drone Class ----------------
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
        
    def update_rpms(self, target, dt):
        # Calcola l'errore verticale
        vertical_error = target[2] - self.position[2]
        vertical_decel_distance = 20.0
        vertical_factor = min(abs(vertical_error) / vertical_decel_distance, 1.0)

        v_speed_factor = self.vertical_speed / self.max_vertical_speed
        vertical_component = self.hover_rpm + (self.max_RPM - self.hover_rpm) * vertical_factor * v_speed_factor * np.sign(vertical_error)

        # Componente orizzontale (considerando la velocità)
        horizontal_distance = np.linalg.norm(target[:2] - self.position[:2])
        horizontal_decel_distance = 20.0

        horizontal_factor = min(horizontal_distance / horizontal_decel_distance, 1.0)
        h_speed_factor = self.horizontal_speed / self.max_horizontal_speed
        horizontal_component = (self.max_RPM - self.min_RPM) * horizontal_factor * h_speed_factor

        # Somma le due componenti con il valore di hover per ottenere il target RPM medio
        desired_avg_rpm = vertical_component + horizontal_component * 0.13
        desired_avg_rpm = np.clip(desired_avg_rpm, self.min_RPM, self.max_RPM)

        # Aggiornamento degli RPM con tasso di variazione limitato
        rpm_update_gain = 50  # Regola questo guadagno se serve
        max_rpm_change = 600 * dt  # Limite per evitare variazioni troppo brusche

        for i in range(4):
            error = desired_avg_rpm - self.rpm_values[i]
            rpm_adjustment = rpm_update_gain * error * dt
            rpm_adjustment = np.clip(rpm_adjustment, -max_rpm_change, max_rpm_change)
            self.rpm_values[i] += rpm_adjustment
            self.rpm_values[i] = np.clip(self.rpm_values[i], self.min_RPM, self.max_RPM)
    
    
    def update_physics(self, target, dt, distance_error_threshold=1):
        # MOVIMENTO ORIZZONTALE
        horizontal_target = np.array([target[0], target[1]])
        horizontal_position = np.array([self.position[0], self.position[1]])
        horizontal_direction = horizontal_target - horizontal_position
        horizontal_distance = np.linalg.norm(horizontal_direction)

        last_horizontal_target = np.array([self.target_history[-1][0], self.target_history[-1][1]])
        last_horizontal_direction = last_horizontal_target - horizontal_position
        last_horizontal_distance = np.linalg.norm(last_horizontal_direction)

        horizontal_decel_factor = min(horizontal_distance / self.horiz_decel_distance, 1.0)
        hotizontal_accel_factor = min((last_horizontal_distance + distance_error_threshold) / self.horiz_accel_distance, 1.0)
        
        hotizontal_total_acceleration_factor = horizontal_decel_factor * hotizontal_accel_factor

        horizontal_velocity = self.horizontal_speed * hotizontal_total_acceleration_factor * (horizontal_direction / horizontal_distance)
        horizontal_step = horizontal_velocity * dt
        if np.linalg.norm(horizontal_step) > horizontal_distance:
            horizontal_position = horizontal_target.copy()
        else:
            horizontal_position += horizontal_step

            
        self.position[0] = horizontal_position[0]
        self.position[1] = horizontal_position[1]
        self.velocity[0] = horizontal_velocity[0]
        self.velocity[1] = horizontal_velocity[1]
        
        # MOVIMENTO VERTICALE
        vertical_error = target[2] - self.position[2]
        last_vertical_error = self.target_history[-1][2] - self.position[2]

       

        vertical_decel_factor = min(abs(vertical_error) / self.vertical_decel_distance , 1.0)
        vertical_accel_factor = min((abs(last_vertical_error) + distance_error_threshold) / self.vertical_accel_distance, 1.0)

        vertical_total_acceleration_factor = vertical_decel_factor * vertical_accel_factor
        # Velocità verticale desiderata: costante (massima se lontano, ridotta quando ci si avvicina)
        desired_vertical_velocity = self.vertical_speed * vertical_total_acceleration_factor * np.sign(vertical_error)
        vertical_step = desired_vertical_velocity * dt
        
        # Se il passo supera l'errore residuo, arriva esattamente a target
        if abs(vertical_step) > abs(vertical_error):
            self.position[2] = target[2]
            self.velocity[2] = 0
        else:
            self.position[2] += vertical_step
            self.velocity[2] = desired_vertical_velocity
        
        # Assicurarsi di non scendere sotto terra
        self.position[2] = max(self.position[2], 0)
    
    
    def update_control(self, target, dt):
        target_position = target[:3]
        target_h_speed = target[3]
        target_v_speed = target[4]
        self.horizontal_speed = target_h_speed
        self.vertical_speed = target_v_speed
        self.update_rpms(target_position, dt)
        direction = target_position - self.position
        horizontal_distance = math.sqrt(direction[0]**2 + direction[1]**2)
        if np.linalg.norm(direction) > 0:
            self.yaw = math.atan2(direction[1], direction[0])
            self.pitch = math.atan2(-direction[2], horizontal_distance)
        self.update_physics(target_position, dt)
        return self.pitch, self.yaw, self.rpm_values.copy(), self.position.copy(), self.velocity.copy()


# --------------- Simulation Class (with CSV Logging and Plotting) ---------------
class Simulation:
    def __init__(self, drone: Drone, world: World, noise_model: np.ndarray = None):
        self.drone = drone
        self.world = world
        self.noise_model = noise_model

    def generate_intermediate_points(self, point_a, point_b, custom_points):
        points = []
        points.append([point_a["x"],point_a["y"],point_a["z"],point_a["h_speed"],point_a["v_speed"]])
        for p in custom_points:
            points.append([p["x"],p["y"],p["z"],p["h_speed"],p["v_speed"]])
        points.append([point_b["x"],point_b["y"],point_b["z"],point_b["h_speed"],point_b["v_speed"]])
        return points

    def simulate_trajectory(self, 
                            point_a, 
                            point_b,
                            custom_points,
                            dt=0.1, 
                            horizontal_threshold=5.0, 
                            vertical_threshold=2.0,
                            print_log=True,
                            noise_annoyance_radius=100,
                            noise_rule_cost_gain=1.0,
                            altitude_rule_cost_gain=1.0,
                            time_cost_gain=1.0,
                            distance_cost_gain=1.0,
                            power_cost_gain=1.0
                            ):
        start_time = time.time()
        targets = self.generate_intermediate_points(point_a, point_b, custom_points)
        all_targets = targets.copy()
        trajectory = [self.drone.position.copy()]
        total_distance = 0.0  
        t_elapsed = 0
        log_data = []  # Log: time, pos, pitch, yaw, rpms, velocity
        noise_model_update_frequency = 3  # Update noise model

        costs = {
            "noise": 0,
            "altitude": 0,
            "time": 0,
            "distance": 0,
            "power": 0
        }

        total_avg_spl = []

        while targets:
            target = targets[0]
            horizontal_err = np.linalg.norm(self.drone.position[:2] - target[:2])
            vertical_err = abs(self.drone.position[2] - target[2])
            while horizontal_err > horizontal_threshold or vertical_err > vertical_threshold:
                previous_position = self.drone.position.copy()
                pitch, yaw, rpms, pos, vel = self.drone.update_control(target, dt)
                if self.noise_model is not None and int(t_elapsed % noise_model_update_frequency) == 0:
                    ground_areas, ground_parameters = self.world.get_areas_in_circle(int(pos[0]), int(pos[1]), 1, noise_annoyance_radius)
                    average_spl = 0
                    for i in range(len(ground_areas)):
                        x, y, _ = ground_areas[i]
                        # Compute distance between drone and area center
                        distance = np.linalg.norm(pos - np.array([x, y, 0]))
                        # Calculate radiation angle
                        zeta = np.arctan2(abs(pos[2]), distance)
                        # Calculate sound power level
                        swl_ref_rpm = self.noise_model[int(zeta * 180 / np.pi)]
                        # Set SWL depending on drone RPM
                        swl = swl_ref_rpm + 10 * np.log10(rpms[0] / self.drone.hover_rpm)
                        # Calculate sound pressure level
                        spl = swl - 10 * np.log10(4*np.pi*distance**2)
                        average_spl += spl
                        #print(spl)
                        # Check if area rules are violated
                        if ground_parameters[i] != {}:
                            pass
                    average_spl /= len(ground_areas)
                    total_avg_spl.append(average_spl)
                step_distance = np.linalg.norm(self.drone.position - previous_position)
                total_distance += step_distance
                trajectory.append(self.drone.position.copy())
                t_elapsed += dt
                horizontal_err = np.linalg.norm(self.drone.position[:2] - target[:2])
                vertical_err = abs(self.drone.position[2] - target[2])
                log_text = [round(t_elapsed, 2), pos[0], pos[1], pos[2],
                                 round(pitch, 2), round(yaw, 2),
                                 int(rpms[0]), int(rpms[1]), int(rpms[2]), int(rpms[3]),
                                 round(vel[0], 2), round(vel[1], 2), round(vel[2], 2), round(self.drone.horizontal_speed,2)]
                log_data.append(log_text)
                if print_log: print(log_text)
            self.drone.target_history.append(target)
            targets.pop(0)
        print(f"Total average noise: {np.average(total_avg_spl)} dB")
            
        
        elapsed = time.time() - start_time
        costs["distance"] += total_distance * distance_cost_gain
        costs["time"] += t_elapsed * time_cost_gain
        costs["power"] += np.average(log_data, axis=0)[6:10].sum() * t_elapsed * power_cost_gain
        costs["noise"] *= noise_rule_cost_gain * noise_model_update_frequency
        costs["altitude"] *= altitude_rule_cost_gain * noise_model_update_frequency

        total_cost = sum([v for k, v in costs.items()])

        print(f"Total cost: {total_cost:.2f}")
        print(f"Cost breakdown: {costs}")

        print(f"Simulation completed in {elapsed:.2f} seconds.")
        print(f"The drone reached all targets in {t_elapsed:.2f} seconds.")
        print(f"Total distance traveled: {total_distance:.2f} meters.")

        csv_filename = f"Logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_drone_log.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time(s)", "X", "Y", "Z", "Pitch", "Yaw", "FL", "FR", "RL", "RR", "Vx", "Vy", "Vz", "Hor_Speed"])
            writer.writerows(log_data)
        print(f"Log saved to {csv_filename}.")
        return trajectory, total_cost, log_data, all_targets