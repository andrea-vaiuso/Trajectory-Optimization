import os
import time
import json
import yaml
import datetime
import numpy as np
import optuna
from optuna.samplers import TPESampler
from Entity.Simulation import Simulation
from Entity.World import World
from Entity.Drone import Drone
from utility import showPlot, plotCosts, show2DWorld

# Global optimization counters
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


def compute_base_point(t, A, B):
    """Compute the base point (linear interpolation) between A and B at fraction t."""
    return {
        "x": A["x"] + t * (B["x"] - A["x"]),
        "y": A["y"] + t * (B["y"] - A["y"]),
        "z": A["z"] + t * (B["z"] - A["z"])
    }


def generate_custom_points(param_list, A, B, num_points, grid_step, max_world_size):
    """
    Given a flat parameter list (optimization parameters), generate a list of custom points.
    Each point consists of 5 values (3 offsets and 2 speeds).
    """
    custom_points = []
    for i in range(num_points):
        t = (i + 1) / (num_points + 1)
        base_point = compute_base_point(t, A, B)
        offset_x = int(round(param_list[i * 5 + 0])) * grid_step
        offset_y = int(round(param_list[i * 5 + 1])) * grid_step
        offset_z = int(round(param_list[i * 5 + 2])) * grid_step

        final_x = np.clip(base_point["x"] + offset_x, 0, max_world_size)
        final_y = np.clip(base_point["y"] + offset_y, 0, max_world_size)
        final_z = np.clip(base_point["z"] + offset_z, 0, max_world_size)

        point = {
            "x": final_x,
            "y": final_y,
            "z": final_z,
            "h_speed": param_list[i * 5 + 3],
            "v_speed": param_list[i * 5 + 4]
        }
        custom_points.append(point)
    return custom_points


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
    global iterations, costs
    with open("optimization_params.yaml", "r") as file:
        params = yaml.safe_load(file)

    # Load parameters from YAML.
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
    cost_gains = get_cost_gains(A, B, drone)
    noise_gain, altitude_gain, time_gain, distance_gain, power_gain = cost_gains

    # Pre-calculate max_offset used for defining search bounds.
    distAB = np.sqrt((B["x"] - A["x"])**2 + (B["y"] - A["y"])**2 + (B["z"] - A["z"])**2)
    max_offset = perturbation_factor * distAB

    def objective(trial: optuna.Trial):
        nonlocal sim, A, B, num_points, grid_step, max_world_size, noise_gain, altitude_gain, time_gain, distance_gain, power_gain
        global iterations, costs
        iterations += 1

        # Create a flat list of parameters for all custom points.
        params_list = []
        for i in range(num_points):
            t = (i + 1) / (num_points + 1)
            base_point = compute_base_point(t, A, B)
            # Calculate lower and upper bounds for each offset, in discrete units.
            lower_x = max(-max_offset, -base_point["x"])
            upper_x = min(max_offset, max_world_size - base_point["x"])
            lx_disc = int(np.ceil(lower_x / grid_step))
            ux_disc = int(np.floor(upper_x / grid_step))
            offset_x = trial.suggest_int(f"offset_x_{i}", lx_disc, ux_disc, step=grid_step)

            lower_y = max(-max_offset, -base_point["y"])
            upper_y = min(max_offset, max_world_size - base_point["y"])
            ly_disc = int(np.ceil(lower_y / grid_step))
            uy_disc = int(np.floor(upper_y / grid_step))
            offset_y = trial.suggest_int(f"offset_y_{i}", ly_disc, uy_disc, step=grid_step)

            lower_z = max(-max_offset, -base_point["z"])
            upper_z = min(max_offset, max_world_size - base_point["z"])
            lz_disc = int(np.ceil(lower_z / grid_step))
            uz_disc = int(np.floor(upper_z / grid_step))
            offset_z = trial.suggest_int(f"offset_z_{i}", lz_disc, uz_disc, step=grid_step)

            h_speed = trial.suggest_int(f"h_speed_{i}", 5, params["max_horizontal_speed"], step=1)
            v_speed = trial.suggest_int(f"v_speed_{i}", 3, params["max_vertical_speed"], step=1)

            params_list.extend([offset_x, offset_y, offset_z, h_speed, v_speed])

        custom_points = generate_custom_points(params_list, A, B, num_points, grid_step, max_world_size)
        print(f"Iteration ({iterations}/{n_iterations}) | Best: {min(costs):.2f} | ", end="")

        # Use a coarser dt (dt=1) for faster evaluation during optimization.
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
        # If the simulation didn't complete, assign a high cost.
        if not simulation_completed:
            total_cost = np.nan
        costs.append(total_cost)
        return total_cost

    print("Starting optimization with Optuna...")
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=params["optimization_random_state"]))
    start_time = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_iterations)
    end_time = time.time()

    best_trial = study.best_trial
    print("Optimal cost:", best_trial.value)
    print("Total optimization time: {:.2f} seconds".format(end_time - start_time))

    # Reconstruct the best parameters.
    best_params = []
    for i in range(num_points):
        offset_x = best_trial.params[f"offset_x_{i}"]
        offset_y = best_trial.params[f"offset_y_{i}"]
        offset_z = best_trial.params[f"offset_z_{i}"]
        h_speed = best_trial.params[f"h_speed_{i}"]
        v_speed = best_trial.params[f"v_speed_{i}"]
        best_params.extend([offset_x, offset_y, offset_z, h_speed, v_speed])

    custom_points_best = generate_custom_points(best_params, A, B, num_points, grid_step, max_world_size)
    print("Best points:", custom_points_best)

    # Save the best trajectory and optimization info.
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(f"OptimizedTrajectory/{time_str}", exist_ok=True)
    np.save(f"OptimizedTrajectory/{time_str}/bestpoints.npy", custom_points_best)
    plotCosts(costs[1:], save=True, datetime=time_str)

    optimization_info = {
        "n_iterations": int(n_iterations),
        "best_cost": float(best_trial.value),
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
        "angle_noise_model": params["noise_model"],
        "world_file_name": world.world_name,
        "best_trial_params": best_trial.params
    }

    with open(f"OptimizedTrajectory/{time_str}/optimization_info.json", "w") as json_file:
        json.dump(optimization_info, json_file, indent=4)
    
    print("Executing simulation with best trajectory...")
    execute_simulation(
        drone,
        world,
        angle_noise_model, 
        A, B, 
        custom_points_best, 
        cost_gains,
        showplots=True,
        interval=30,
        log_folder=f"OptimizedTrajectory/{time_str}"
    )


if __name__ == "__main__":
    main()
