import numpy as np
import time
import datetime
import csv
import numpy as np
import math
from Entity.Drone import Drone
from Entity.World import World


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
                            power_cost_gain=1.0,
                            time_limit_gain=10,
                            save_log=True,
                            print_info=True,
                            ):
        self.drone.set_position(point_a["x"], point_a["y"], point_a["z"])
        start_time = time.time()
        targets = self.generate_intermediate_points(point_a, point_b, custom_points)
        all_targets = targets.copy()
        trajectory = [self.drone.position.copy()]
        total_distance = 0.0  
        t_elapsed = 0
        log_data = []  # Log: time, pos, pitch, yaw, rpms, velocity
        noise_model_update_frequency = 3  # Update noise model

        distAB = np.sqrt((point_b["x"] - point_a["x"])**2 + (point_b["y"] - point_a["y"])**2 + (point_b["z"] - point_a["z"])**2) * 1.1
        time_limit = distAB / np.sqrt(self.drone.max_horizontal_speed**2 + self.drone.max_vertical_speed**2) * time_limit_gain
        if print_info: print(f"Time limit: {time_limit:.2f} seconds.")

        costs = {
            "noise": 0,
            "altitude": 0,
            "time": 0,
            "distance": 0,
            "power": 0
        }

        total_avg_spl = []
        total_avg_noise_costs = []
        total_avg_altitude_costs = []

        while targets:
            if t_elapsed > time_limit:
                break
            target = targets[0]
            horizontal_err = np.linalg.norm(self.drone.position[:2] - target[:2])
            vertical_err = abs(self.drone.position[2] - target[2])
            while horizontal_err > horizontal_threshold or vertical_err > vertical_threshold:
                previous_position = self.drone.position.copy()
                pitch, yaw, rpms, pos, vel = self.drone.update_control(target, dt)
                noise_cost = 0
                altitude_cost = 0
                if self.noise_model is not None and int(t_elapsed % noise_model_update_frequency) == 0:
                    ground_areas, ground_parameters = self.world.get_areas_in_circle(int(pos[0]), int(pos[1]), 1, noise_annoyance_radius)
                    average_spl = 0
                    average_noise_cost = 0
                    average_altitude_cost = 0
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
                        spl = swl - abs(10 * np.log10(1/4*np.pi*distance**2))
                        average_spl += spl
                        #print(spl)
                        # Check if area rules are violated
                        if ground_parameters[i] != {}:
                            average_noise_cost += spl * ground_parameters[i]["noise_penalty"]
                            average_altitude_cost += max(pos[2] - ground_parameters[i]["max_altitude"], 0)
                            average_altitude_cost += max(ground_parameters[i]["min_altitude"] - pos[2], 0)  

                    average_noise_cost /= len(ground_areas)
                    average_altitude_cost /= len(ground_areas)
                    average_spl /= len(ground_areas)
                    total_avg_spl.append(average_spl)
                    total_avg_noise_costs.append(average_noise_cost)
                    total_avg_altitude_costs.append(average_altitude_cost)
                
                noise_cost += np.average(total_avg_noise_costs) * noise_model_update_frequency
                altitude_cost += np.average(total_avg_altitude_costs) * noise_model_update_frequency

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
        if print_info: print(f"Total average noise: {np.average(total_avg_spl)} dB")
            
        
        elapsed = time.time() - start_time
        costs["distance"] += total_distance ** 1.3 * distance_cost_gain
        costs["time"] += t_elapsed ** 1.3 * time_cost_gain
        costs["power"] += np.average(log_data, axis=0)[6:10].sum() * t_elapsed * power_cost_gain
        costs["noise"] += noise_cost * noise_rule_cost_gain
        costs["altitude"] += altitude_cost * altitude_rule_cost_gain

        if t_elapsed > time_limit:
                costs["distance"] *= 9e4
                costs["time"] *= 9e4
                costs["power"] *= 9e4

        total_cost = np.sqrt(sum([v**2 for k, v in costs.items()]))

        if print_info: print(f"Total cost: {total_cost:.2f}")
        if print_info: print(f"Cost breakdown: {costs}")

        print(f"Sim_time: {elapsed:.2f}s | Flight_time: {t_elapsed:.2f}s | Dist: {total_distance:.2f}m | Cost: {total_cost:.2f}")

        if save_log:
            csv_filename = f"Logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_drone_log.csv"
            with open(csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time(s)", "X", "Y", "Z", "Pitch", "Yaw", "FL", "FR", "RL", "RR", "Vx", "Vy", "Vz", "Hor_Speed"])
                writer.writerows(log_data)
            print(f"Log saved to {csv_filename}.")

        return trajectory, total_cost, log_data, all_targets