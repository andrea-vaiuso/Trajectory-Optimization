from Entity.Simulation import Simulation
from Entity.World import World
from Entity.Drone import Drone
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real  # Per definire dimensioni discrete/continue
from utility import showPlot, plotCosts, show2DWorld
import time
import datetime
import json
import yaml
import os

iterations = 0
costs = [1e9]

def get_cost_gains(A: dict, B: dict, drone: Drone):
    distAB = np.sqrt((B["x"] - A["x"])**2 + (B["y"] - A["y"])**2 + (B["z"] - A["z"])**2) * 1.1
    maxvel = np.sqrt(drone.max_horizontal_speed**2 + drone.max_vertical_speed**2)
    noise_rule_cost_gain = 1
    altitude_rule_cost_gain = 1
    time_cost_gain = maxvel / distAB
    distance_cost_gain = 1 / distAB
    power_cost_gain = time_cost_gain / drone.hover_rpm
    return noise_rule_cost_gain, altitude_rule_cost_gain, time_cost_gain, distance_cost_gain, power_cost_gain

def execute_simulation(drone: Drone, world: World, noise_model, A, B, custom_points, cost_gains, showplots=True, interval=30, log_folder="Logs"):
    sim = Simulation(drone, world, noise_model)
    noise_gain, altitude_gain, time_gain, distance_gain, power_gain = cost_gains
    trajectory, _, log_data, all_targets, _ = sim.simulate_trajectory(
        point_a=A, point_b=B, dt=0.1,
        horizontal_threshold=5.0, vertical_threshold=2.0,
        custom_points=custom_points,
        print_log=False,
        noise_rule_cost_gain=noise_gain,
        altitude_rule_cost_gain=altitude_gain,
        time_cost_gain=time_gain,
        distance_cost_gain=distance_gain,
        power_cost_gain=power_gain,
        save_log=True,
        save_log_folder=log_folder
    )
    if showplots:
        show2DWorld(world, world.grid_size, trajectory, A, B, all_targets, save=True, save_folder=log_folder)    
        showPlot(trajectory, A, B, all_targets, world, world.grid_size, world.max_world_size, log_data, interval=interval)

def main():
    with open("optimization_params.yaml", "r") as file:
        params = yaml.safe_load(file)

    grid_size = params["grid_size"]
    max_world_size = params["max_world_size"]
    num_points = params["num_points"]
    n_iterations = params["n_iterations"]
    perturbation_factor = params["perturbation_factor"]
    grid_step = params["grid_step"]
    world_file_name = params["world_file_name"]
    A = params["A"]
    B = params["B"]

    print("Loading world...")
    world = World.load_world(world_file_name)


    print("Creating drone...")
    drone = Drone(
        model_name=params["drone_model_name"],
        x=A["x"],
        y=A["y"],
        z=A["z"],
        min_RPM=params["min_RPM"],
        max_RPM=params["max_RPM"],
        hover_RPM=params["hover_RPM"],
        max_horizontal_speed=params["max_horizontal_speed"],
        max_vertical_speed=params["max_vertical_speed"]
    )

    print("Loading noise model...")
    angle_noise_model = np.load(params["noise_model"])
    
    print("Initializing simulation...")
    sim = Simulation(drone, world, angle_noise_model)
    
    noise_gain, altitude_gain, time_gain, distance_gain, power_gain = get_cost_gains(A, B, drone)
    
    distAB = np.sqrt((B["x"] - A["x"])**2 + (B["y"] - A["y"])**2 + (B["z"] - A["z"])**2)
    max_offset = perturbation_factor * distAB

    def cost_function(params):
        global iterations, costs
        iterations += 1
        custom_points = []
        for i in range(num_points):
            t = (i + 1) / (num_points + 1)
            base_point = {
                "x": A["x"] + t * (B["x"] - A["x"]),
                "y": A["y"] + t * (B["y"] - A["y"]),
                "z": A["z"] + t * (B["z"] - A["z"])
            }
            # Discretizziamo gli offset: li arrotondiamo all'intero più vicino,
            # poi moltiplichiamo per grid_step per ottenere il valore in metri.
            offset_x = int(round(params[i*5 + 0])) * grid_step
            offset_y = int(round(params[i*5 + 1])) * grid_step
            offset_z = int(round(params[i*5 + 2])) * grid_step
            
            # Assicuriamoci che la somma base_point + offset sia nel range
            final_x = np.clip(base_point["x"] + offset_x, 0, max_world_size)
            final_y = np.clip(base_point["y"] + offset_y, 0, max_world_size)
            final_z = np.clip(base_point["z"] + offset_z, 0, max_world_size)
            
            point = {
                "x": final_x,
                "y": final_y,
                "z": final_z,
                "h_speed": params[i*5 + 3],
                "v_speed": params[i*5 + 4]
            }
            custom_points.append(point)
        print(f"Iteration ({iterations}/{n_iterations}) | Best: {min(costs):.2f} | ", end="")
        _, total_cost, _, _, simulation_completed = sim.simulate_trajectory(
            point_a=A, point_b=B, dt=1,
            horizontal_threshold=5.0, vertical_threshold=2.0,
            custom_points=custom_points,
            print_log=False,
            noise_rule_cost_gain=noise_gain,
            altitude_rule_cost_gain=altitude_gain,
            time_cost_gain=time_gain,
            distance_cost_gain=distance_gain,
            power_cost_gain=power_gain,
            print_info=False,
            save_log=False
        )
        if simulation_completed:
            costs.append(total_cost)
        else:
            costs.append(np.nan)
        return total_cost

    # Costruiamo i bounds per ciascun punto intermedio.
    # Per ogni punto, calcoliamo il base_point (interpolazione lineare tra A e B)
    # e impostiamo i bounds per gli offset in modo che base_point + offset sia in [0, max_world_size].
    # I bounds per gli offset sono definiti in "unità discrete" (multipli di grid_step).
    dimensions = []
    x0 = []  # Punto iniziale per l'ottimizzazione

    for i in range(num_points):
        t = (i + 1) / (num_points + 1)
        base_point = {
            "x": A["x"] + t * (B["x"] - A["x"]),
            "y": A["y"] + t * (B["y"] - A["y"]),
            "z": A["z"] + t * (B["z"] - A["z"])
        }
        # Calcola i limiti per ciascun offset
        lower_x = max(-max_offset, -base_point["x"])
        upper_x = min(max_offset, max_world_size - base_point["x"])
        lower_y = max(-max_offset, -base_point["y"])
        upper_y = min(max_offset, max_world_size - base_point["y"])
        lower_z = max(-max_offset, -base_point["z"])
        upper_z = min(max_offset, max_world_size - base_point["z"])

        # Convertiamo i bounds in unità discrete (multipli di grid_step)
        lx_disc = int(np.ceil(lower_x / grid_step))
        ux_disc = int(np.floor(upper_x / grid_step))
        ly_disc = int(np.ceil(lower_y / grid_step))
        uy_disc = int(np.floor(upper_y / grid_step))
        lz_disc = int(np.ceil(lower_z / grid_step))
        uz_disc = int(np.floor(upper_z / grid_step))

        dimensions.extend([
            Integer(lx_disc, ux_disc),  # offset_x (in unità di grid_step)
            Integer(ly_disc, uy_disc),  # offset_y
            Integer(lz_disc, uz_disc)   # offset_z
        ])

        dimensions.extend([
            Real(5, 20),   # h_speed
            Real(3, 8)     # v_speed
        ])

        # Punto iniziale: per offset usiamo 0 (cioè 0 * grid_step = 0)
        # per le velocità usiamo i valori di A.
        x0.extend([0, 0, 0, A["h_speed"], A["v_speed"]])

    print("Starting optimization...")
    start_time = time.time()

    result = gp_minimize(cost_function, dimensions, x0=x0, n_calls=n_iterations, random_state=params["optimization_random_state"])

    end_time = time.time()
    print("Optimal cost:", result.fun)
    print("Total optimization time: {:.2f} seconds".format(end_time - start_time))

    best_params = result.x

    print("Simulating best trajectory...")
    custom_points_best = []
    for i in range(num_points):
        t = (i + 1) / (num_points + 1)
        base_point = {
            "x": A["x"] + t * (B["x"] - A["x"]),
            "y": A["y"] + t * (B["y"] - A["y"]),
            "z": A["z"] + t * (B["z"] - A["z"])
        }
        final_x = np.clip(base_point["x"] + int(round(best_params[i*5 + 0])) * grid_step, 0, max_world_size)
        final_y = np.clip(base_point["y"] + int(round(best_params[i*5 + 1])) * grid_step, 0, max_world_size)
        final_z = np.clip(base_point["z"] + int(round(best_params[i*5 + 2])) * grid_step, 0, max_world_size)
        
        point = {
            "x": final_x,
            "y": final_y,
            "z": final_z,
            "h_speed": best_params[i*5 + 3],
            "v_speed": best_params[i*5 + 4]
        }
        custom_points_best.append(point)

    print("Best points:", custom_points_best)
    # Save the best custom points into a file
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Create the folder if it doesn't exist
    os.makedirs(f"OptimizedTrajectory/{time_str}", exist_ok=True)

    np.save(f"OptimizedTrajectory/{time_str}/bestpoints.npy", custom_points_best)
    plotCosts(costs[1:], save=True, datetime=time_str)
    # Create a dictionary to store optimization information
    optimization_info = {
        "n_iterations": int(n_iterations),
        "best_cost": float(result.fun),
        "optimization_time_seconds": float(end_time - start_time),
        "n_points": int(num_points),
        "custom_points": custom_points_best,
        "A": {k: float(v) for k, v in A.items()},
        "B": {k: float(v) for k, v in B.items()},
        "grid_size": int(grid_size),
        "max_world_size": int(max_world_size),
        "perturbation_factor": float(perturbation_factor),
        "grid_step": int(grid_step),
        "noise_rule_cost_gain": float(noise_gain),
        "altitude_rule_cost_gain": float(altitude_gain),
        "time_cost_gain": float(time_gain),
        "distance_cost_gain": float(distance_gain),
        "power_cost_gain": float(power_gain),
        "drone": drone.to_dict(),
        "angle_noise_model": "dnn_sound_model/angles_swl.npy",
        "world_file_name": world.world_name
    }

    # Save the dictionary to a JSON file
    with open(f"OptimizedTrajectory/{time_str}/optimization_info.json", "w") as json_file:
        json.dump(optimization_info, json_file, indent=4)
    
    print("Executing simulation...")
    execute_simulation(drone,
                    world,
                    angle_noise_model, 
                    A, B, 
                    custom_points_best, 
                    (noise_gain, 
                        altitude_gain, 
                        time_gain, 
                        distance_gain, 
                        power_gain),
                    showplots=True,
                    interval=30,
                    log_folder=f"OptimizedTrajectory/{time_str}")

if __name__ == "__main__":
    main()
