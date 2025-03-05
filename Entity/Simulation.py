import numpy as np
import time
import datetime
import csv
import math
from Entity.Drone import Drone
from Entity.World import World

class Simulation:
    def __init__(self, drones: list, world: World, noise_model: np.ndarray = None):
        self.drones = drones
        self.world = world
        self.noise_model = noise_model

    def generate_intermediate_points_for_all(self, points_a, points_b, custom_points_all):
        # For each drone, generate list of waypoints: [start] + custom_points + [target]
        targets_all = []
        for a, b, custom_points in zip(points_a, points_b, custom_points_all):
            targets = []
            targets.append([a["x"], a["y"], a["z"], a["h_speed"], a["v_speed"]])
            for p in custom_points:
                targets.append([p["x"], p["y"], p["z"], p["h_speed"], p["v_speed"]])
            targets.append([b["x"], b["y"], b["z"], b["h_speed"], b["v_speed"]])
            targets_all.append(targets)
        return targets_all

    def simulate_trajectory(self, 
                            points_a, 
                            points_b,
                            custom_points_all,
                            dt=0.1, 
                            horizontal_threshold=5.0, 
                            vertical_threshold=2.0,
                            print_log=True,
                            noise_annoyance_radius=100,
                            noise_rule_cost_gain=1.0,
                            altitude_rule_cost_gain=1.0,
                            time_cost_gain=1.0,
                            distance_cost_gain=1.0,
                            power_cost_gain=1.0,
                            collision_threshold=2.0,
                            collision_cost_gain=1e6,
                            time_limit_gain=10,
                            save_log=True,
                            save_log_folder="Logs",
                            print_info=True
                           ):
        n_drones = len(self.drones)
        # Set initial positions for all drones
        for i, drone in enumerate(self.drones):
            a = points_a[i]
            drone.set_position(a["x"], a["y"], a["z"])
            drone.target_history = []
        
        targets_all = self.generate_intermediate_points_for_all(points_a, points_b, custom_points_all)
        # Initialize trajectories: list for each drone
        trajectories = [[drone.position.copy()] for drone in self.drones]
        # Initialize log data: list of dict entries per time step
        log_data = []
        # Initialize cost accumulators for each drone
        total_distances = [0.0 for _ in range(n_drones)]
        # Time limits per drone
        time_limits = []
        for i, drone in enumerate(self.drones):
            a = points_a[i]
            b = points_b[i]
            distAB = np.sqrt((b["x"] - a["x"])**2 + (b["y"] - a["y"])**2 + (b["z"] - a["z"])**2) * 1.1
            maxvel = np.sqrt(drone.max_horizontal_speed**2 + drone.max_vertical_speed**2)
            time_limits.append(distAB / maxvel * time_limit_gain)
        global_time_limit = max(time_limits)

        simulation_completed = True
        t_elapsed = 0.0
        # Initialize cost components for each drone
        costs = [{"noise": 0, "altitude": 0, "time": 0, "distance": 0, "power": 0} for _ in range(n_drones)]
        collision_cost_total = 0.0

        noise_model_update_frequency = 3

        # Main simulation loop, run until all drones have finished their targets or time limit exceeded
        while any(len(targets) > 1 for targets in targets_all):
            if t_elapsed > global_time_limit:
                simulation_completed = False
                break

            positions = []  # Current positions of all drones
            # For each drone, update if not reached current target
            for i, drone in enumerate(self.drones):
                targets = targets_all[i]
                # If only one target remains, drone has finished
                if len(targets) <= 1:
                    positions.append(drone.position.copy())
                    continue
                current_target = targets[0]
                horizontal_err = np.linalg.norm(drone.position[:2] - np.array(current_target[:2]))
                vertical_err = abs(drone.position[2] - current_target[2])
                if horizontal_err > horizontal_threshold or vertical_err > vertical_threshold:
                    previous_position = drone.position.copy()
                    pitch, yaw, rpms, pos, vel = drone.update_control(current_target, dt)
                    step_distance = np.linalg.norm(drone.position - previous_position)
                    total_distances[i] += step_distance
                    # Noise and altitude cost calculation
                    noise_cost = 0
                    altitude_cost = 0
                    if self.noise_model is not None and int(t_elapsed % noise_model_update_frequency) == 0:
                        ground_areas, ground_parameters = self.world.get_areas_in_circle(int(pos[0]), int(pos[1]), 1, noise_annoyance_radius)
                        avg_noise = 0
                        avg_altitude = 0
                        for j in range(len(ground_areas)):
                            x, y, _ = ground_areas[j]
                            distance = np.linalg.norm(pos - np.array([x, y, 0]))
                            zeta = np.arctan2(abs(pos[2]), distance)
                            swl_ref_rpm = self.noise_model[int(zeta * 180 / np.pi)]
                            swl = swl_ref_rpm + 10 * np.log10(rpms[0] / drone.hover_rpm)
                            spl = swl - abs(10 * np.log10(1/4*np.pi*(distance+1e-4)**2))
                            avg_noise += spl
                            if ground_parameters[j] != {}:
                                avg_altitude += max(pos[2] - ground_parameters[j]["max_altitude"], 0)
                                avg_altitude += max(ground_parameters[j]["min_altitude"] - pos[2], 0)
                        avg_noise /= len(ground_areas)
                        avg_altitude /= len(ground_areas)
                        noise_cost += avg_noise * noise_model_update_frequency
                        altitude_cost += avg_altitude * noise_model_update_frequency
                    costs[i]["distance"] = total_distances[i] ** 1.3 * distance_cost_gain
                    costs[i]["time"] = t_elapsed ** 1.3 * time_cost_gain
                    costs[i]["power"] += np.sum(rpms) * dt * power_cost_gain
                    costs[i]["noise"] += noise_cost * noise_rule_cost_gain
                    costs[i]["altitude"] += altitude_cost * altitude_rule_cost_gain

                    # Log data for this drone
                    log_entry = {
                        "time": round(t_elapsed, 2),
                        "drone_index": i,
                        "position": pos.tolist(),
                        "pitch": round(pitch, 2),
                        "yaw": round(yaw, 2),
                        "rpms": [int(r) for r in rpms],
                        "velocity": [round(v, 2) for v in vel],
                        "horizontal_speed": round(drone.horizontal_speed,2)
                    }
                    log_data.append(log_entry)
                else:
                    # Target reached, pop target and record in target history
                    drone.target_history.append(targets.pop(0))
                positions.append(drone.position.copy())
                trajectories[i].append(drone.position.copy())

            # Collision detection among drones
            for i in range(n_drones):
                for j in range(i+1, n_drones):
                    dist_between = np.linalg.norm(np.array(trajectories[i][-1]) - np.array(trajectories[j][-1]))
                    if dist_between < collision_threshold:
                        collision_cost_total += collision_cost_gain

            t_elapsed += dt

        # Aggregate total cost from all drones
        total_cost_components = np.array([sum(costs[i].values()) for i in range(n_drones)]) + collision_cost_total
        total_cost = np.sqrt(np.sum(total_cost_components**2))

        if t_elapsed > global_time_limit:
            total_cost = np.nan
            simulation_completed = False

        if print_info:
            print(f"Total simulation time: {t_elapsed:.2f}s")
            print(f"Total distances: {total_distances}")
            print(f"Collision cost: {collision_cost_total}")
            print(f"Total cost: {total_cost:.2f}")
        
        # Save log if required
        if save_log:
            csv_filename = f"{save_log_folder}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_drones_log.csv"
            with open(csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ["Time(s)"]
                for i in range(n_drones):
                    header.extend([f"Drone{i}_X", f"Drone{i}_Y", f"Drone{i}_Z", f"Drone{i}_Pitch", f"Drone{i}_Yaw", f"Drone{i}_RPMs", f"Drone{i}_Velocity", f"Drone{i}_HorSpeed"])
                writer.writerow(header)
                # Group log entries by time steps
                # This is a simple approach, not fully synchronized across drones
                for entry in log_data:
                    row = [entry["time"], entry["position"][0], entry["position"][1], entry["position"][2],
                           entry["pitch"], entry["yaw"], entry["rpms"], entry["velocity"], entry["horizontal_speed"]]
                    writer.writerow(row)
            print(f"Log saved to {csv_filename}.")

        return trajectories, total_cost, log_data, targets_all, simulation_completed
