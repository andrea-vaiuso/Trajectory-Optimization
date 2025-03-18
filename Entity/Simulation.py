import numpy as np
import time
import datetime
import csv
import math
from Entity.Drone import Drone
from Entity.World import World

# --------------- Simulation Class (with CSV Logging and Plotting) ---------------
class Simulation:
    def __init__(self, drones: list, world: World, noise_model: np.ndarray = None):
        self.drones = drones
        self.world = world
        self.noise_model = noise_model

    def generate_intermediate_points(self, point_a, point_b, custom_points):
        """
        Generate the full list of waypoints: start, intermediate custom points, and destination.
        Each waypoint is a list: [x, y, z, h_speed, v_speed].
        """
        points = []
        points.append([point_a["x"], point_a["y"], point_a["z"], point_a["h_speed"], point_a["v_speed"]])
        for p in custom_points:
            points.append([p["x"], p["y"], p["z"], p["h_speed"], p["v_speed"]])
        points.append([point_b["x"], point_b["y"], point_b["z"], point_b["h_speed"], point_b["v_speed"]])
        return points

    def simulate_trajectory(self, 
                            point_a_list, 
                            point_b_list,
                            custom_points_list,
                            dt=0.1, 
                            horizontal_threshold=5.0, 
                            vertical_threshold=2.0,
                            print_log=True,
                            noise_annoyance_radius=100,
                            cost_gains_list=None,
                            time_limit_gain=10,
                            collision_distance=15.0,
                            collision_cost=9e4,
                            save_log=True,
                            save_log_folder="Logs",
                            print_info=True):
        n = len(self.drones)
        trajectories = []
        logs = []
        finished = [False] * n
        # Set initial positions and clear target history.
        for i in range(n):
            self.drones[i].set_position(point_a_list[i]["x"], point_a_list[i]["y"], point_a_list[i]["z"])
            trajectories.append([self.drones[i].position.copy()])
            logs.append([])
            self.drones[i].target_history = []
        
        # Build target lists for each drone.
        targets = []
        for i in range(n):
            t_list = self.generate_intermediate_points(point_a_list[i], point_b_list[i], custom_points_list[i])
            targets.append(t_list)
        
        # Compute individual time limits and choose the maximum.
        time_limits = []
        for i in range(n):
            A = point_a_list[i]
            B = point_b_list[i]
            distAB = np.sqrt((B["x"] - A["x"])**2 + (B["y"] - A["y"])**2 + (B["z"] - A["z"])**2) * 1.1
            maxvel = np.sqrt(self.drones[i].max_horizontal_speed**2 + self.drones[i].max_vertical_speed**2)
            time_limits.append(distAB / maxvel * time_limit_gain)
        global_time_limit = max(time_limits)
        
        # Cost accumulators for each drone.
        flight_time = [0.0] * n
        total_distance = [0.0] * n
        power_sum = [0.0] * n
        noise_costs = [0.0] * n
        altitude_costs = [0.0] * n
        
        collision_penalty_total = 0.0
        t_elapsed = 0.0
        noise_update_freq = 3  # seconds
        
        while (not all(finished)) and (t_elapsed <= global_time_limit):
            # Update each drone if not finished.
            for i in range(n):
                if finished[i]:
                    continue
                current_target = targets[i][0]  # current waypoint
                pos = self.drones[i].position
                target_pos = np.array(current_target[:3])
                pos_arr = np.array(pos)
                horizontal_err = np.linalg.norm(pos_arr[:2] - target_pos[:2])
                vertical_err = abs(pos_arr[2] - target_pos[2])
                if horizontal_err > horizontal_threshold or vertical_err > vertical_threshold:
                    prev_pos = pos_arr.copy()
                    # Update control (assumes each Drone has update_control method).
                    pitch, yaw, rpms, new_pos, vel = self.drones[i].update_control(current_target, dt)
                    self.drones[i].position = new_pos.copy()
                    trajectories[i].append(new_pos.copy())
                    step_distance = np.linalg.norm(new_pos - prev_pos)
                    total_distance[i] += step_distance
                    flight_time[i] += dt
                    power_sum[i] += sum(rpms)
                    # Noise and altitude cost calculation.
                    if self.noise_model is not None and int(t_elapsed) % noise_update_freq == 0:
                        ground_areas, ground_parameters = self.world.get_areas_in_circle(int(new_pos[0]), int(new_pos[1]), 1, noise_annoyance_radius)
                        avg_noise = 0.0
                        avg_alt = 0.0
                        for j in range(len(ground_areas)):
                            x, y, _ = ground_areas[j]
                            distance = np.linalg.norm(new_pos - np.array([x, y, 0]))
                            zeta = np.arctan2(abs(new_pos[2]), distance)
                            swl_ref_rpm = self.noise_model[min(int(zeta * 180 / np.pi), len(self.noise_model)-1)]
                            swl = swl_ref_rpm + 10 * np.log10(rpms[0] / self.drones[i].hover_rpm)
                            spl = swl - abs(10 * np.log10(1/(4 * np.pi * ((distance+1e-4)**2))))
                            avg_noise += spl
                            if ground_parameters[j] != {}:
                                avg_alt += max(new_pos[2] - ground_parameters[j]["max_altitude"], 0)
                                avg_alt += max(ground_parameters[j]["min_altitude"] - new_pos[2], 0)
                        if len(ground_areas) > 0:
                            avg_noise /= len(ground_areas)
                            avg_alt /= len(ground_areas)
                        noise_costs[i] += avg_noise * noise_update_freq
                        altitude_costs[i] += avg_alt * noise_update_freq
                    log_entry = [round(t_elapsed,2)] + list(new_pos) + [round(pitch,2), round(yaw,2)] + [int(r) for r in rpms] + list(vel) + [round(self.drones[i].horizontal_speed,2)]
                    logs[i].append(log_entry)
                    if print_log:
                        print(f"Drone {i+1} log: {log_entry}", flush=True)
                else:
                    # Target reached.
                    self.drones[i].target_history.append(current_target)
                    targets[i].pop(0)
                    if len(targets[i]) == 0:
                        finished[i] = True
            # Collision check among active drones.
            active_indices = [i for i in range(n) if not finished[i]]
            for idx1 in range(len(active_indices)):
                for idx2 in range(idx1+1, len(active_indices)):
                    i = active_indices[idx1]
                    j = active_indices[idx2]
                    pos_i = np.array(self.drones[i].position)
                    pos_j = np.array(self.drones[j].position)
                    if np.linalg.norm(pos_i - pos_j) < collision_distance:
                        collision_penalty_total += collision_cost
                        if print_info:
                            print(f"Collision detected between drone {i+1} and {j+1} at time {t_elapsed:.2f}s", flush=True)
            t_elapsed += dt
        
        simulation_completed = all(finished)
        
        # Compute individual drone costs using respective cost gains.
        total_cost = 0.0
        for i in range(n):
            noise_gain, altitude_gain, time_gain, distance_gain, power_gain = cost_gains_list[i]
            cost_distance = (total_distance[i] ** 1.3) * distance_gain
            cost_time = (flight_time[i] ** 1.3) * time_gain
            cost_power = (power_sum[i]) * flight_time[i] * power_gain
            cost_noise = noise_costs[i] * noise_gain
            cost_altitude = altitude_costs[i] * altitude_gain
            cost_i = np.sqrt(cost_distance**2 + cost_time**2 + cost_power**2 + cost_noise**2 + cost_altitude**2)
            total_cost += cost_i**2
        total_cost = np.sqrt(total_cost + collision_penalty_total**2)
        if not simulation_completed:
            total_cost = np.nan
        
        if print_info:
            print(f"Flight times: {flight_time} | Distances: {total_distance} | Collision penalty: {collision_penalty_total} | Total cost: {total_cost:.2f}")
        
        # Save log files for each drone.
        if save_log:
            for i in range(n):
                csv_filename = f"{save_log_folder}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_drone{i+1}_log.csv"
                with open(csv_filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Time(s)", "X", "Y", "Z", "Pitch", "Yaw", "FL", "FR", "RL", "RR", "Vx", "Vy", "Vz", "Hor_Speed"])
                    writer.writerows(logs[i])
                if print_info:
                    print(f"Drone {i+1} log saved to {csv_filename}.")
        
        # Instead of returning the remaining targets, return each drone's reached targets.
        all_targets = [drone.target_history for drone in self.drones]
        return trajectories, total_cost, logs, all_targets, simulation_completed

