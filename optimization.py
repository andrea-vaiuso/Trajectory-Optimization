import os
import time
import json
import yaml
import datetime
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real
from Entity.Simulation import Simulation
from Entity.World import World
from Entity.Drone import Drone
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from utility import showPlot, plotCosts, show2DWorld

# Global optimization counters and history
iterations = 0
costs = [1e9]
optimization_history = []  # Will store tuples: (iteration, trajectory, all_targets)

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


def build_dimensions(A, B, num_points, grid_step, max_world_size, perturbation_factor):
    dimensions = []
    x0 = []
    # Calculate overall distance between A and B for the perturbation range.
    distAB = np.sqrt((B["x"] - A["x"])**2 + (B["y"] - A["y"])**2 + (B["z"] - A["z"])**2)
    max_offset = perturbation_factor * distAB

    for i in range(num_points):
        t = (i + 1) / (num_points + 1)
        base_point = compute_base_point(t, A, B)

        # Calculate the lower and upper bounds for each offset so that (base + offset) remains in range.
        lower_x = max(-max_offset, -base_point["x"])
        upper_x = min(max_offset, max_world_size - base_point["x"])
        lower_y = max(-max_offset, -base_point["y"])
        upper_y = min(max_offset, max_world_size - base_point["y"])
        lower_z = max(-max_offset, -base_point["z"])
        upper_z = min(max_offset, max_world_size - base_point["z"])

        # Convert to discrete units.
        lx_disc = int(np.ceil(lower_x / grid_step))
        ux_disc = int(np.floor(upper_x / grid_step))
        ly_disc = int(np.ceil(lower_y / grid_step))
        uy_disc = int(np.floor(upper_y / grid_step))
        lz_disc = int(np.ceil(lower_z / grid_step))
        uz_disc = int(np.floor(upper_z / grid_step))

        dimensions.extend([
            Integer(lx_disc, ux_disc),  # offset_x (in grid_step units)
            Integer(ly_disc, uy_disc),  # offset_y
            Integer(lz_disc, uz_disc)   # offset_z
        ])
        dimensions.extend([
            Real(5, 20),   # horizontal speed
            Real(3, 8)     # vertical speed
        ])
        # Initial guess: offsets = 0, speeds from A.
        x0.extend([0, 0, 0, A["h_speed"], A["v_speed"]])
    return dimensions, x0


def execute_simulation(sim: Simulation, world: World, 
                    A, B, custom_points, cost_gains, 
                       showplots=True, interval=30, log_folder="Logs",
                       dt=0.1, print_info = False, save_log = False, print_log = False
                       ):
    noise_gain, altitude_gain, time_gain, distance_gain, power_gain = cost_gains
    trajectory, total_cost, log_data, all_targets, simulation_completed = sim.simulate_trajectory(
        point_a=A, point_b=B, dt=dt,
        horizontal_threshold=5.0, vertical_threshold=2.0,
        custom_points=custom_points,
        print_log=print_log,
        noise_rule_cost_gain=noise_gain,
        altitude_rule_cost_gain=altitude_gain,
        time_cost_gain=time_gain,
        distance_cost_gain=distance_gain,
        power_cost_gain=power_gain,
        save_log_folder=log_folder,
        print_info=print_info,
        save_log=save_log
    )
    if showplots:
        show2DWorld(world, world.grid_size, trajectory, A, B, all_targets, save=True, save_folder=log_folder)    
        showPlot(trajectory, A, B, all_targets, world, world.grid_size, world.max_world_size, log_data, interval=interval)
    return trajectory, total_cost, log_data, all_targets, simulation_completed


def animate_optimization_steps(world: World, grid_size, A, B, optimization_history, save_path):
    """
    Create an animation of the optimization steps.
    The background (grid and background image) is plotted once.
    For each frame the trajectory (and intermediate targets) is updated.
    The plot title is updated to show the current optimization step.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot static background image (if available)
    if world.background_image is not None:
        bg_img = np.array(world.background_image)
        ax.imshow(bg_img, extent=[0, world.max_world_size, 0, world.max_world_size],
                  origin='lower', alpha=0.7, zorder=-1)
    
    # Plot grid (static)
    for (x, y, z), params in world.grid.items():
        if z == 0:
            rect = plt.Rectangle((x * grid_size, y * grid_size), grid_size, grid_size,
                                 color=world.AREA_PARAMS[params]["color"],
                                 alpha=world.AREA_PARAMS[params]["alpha"])
            ax.add_patch(rect)
    
    # Plot static start and target points
    ax.scatter(A["x"], A["y"], color='green', s=50, label="A (Start)")
    ax.scatter(B["x"], B["y"], color='blue', s=50, label="B (Target)")
    
    # Create dynamic objects for intermediate targets and trajectory
    targets_scatter = ax.scatter([], [], color='red', s=30)
    traj_line, = ax.plot([], [], 'k--', lw=1.5)
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(0, world.max_world_size)
    ax.set_ylim(0, world.max_world_size)
    ax.legend(loc='upper left')
    ax.grid(True)
    
    # Precompute the data arrays for each frame
    precomputed = []
    for step, traj, all_targets in optimization_history:
        if traj and len(traj) > 0:
            traj_arr = np.array(traj)
        else:
            traj_arr = np.empty((0, 2))
        if all_targets is not None and len(all_targets) > 2:
            pts = np.array(all_targets[1:-1])[:, :2]
        else:
            pts = np.empty((0, 2))
        precomputed.append((step, traj_arr, pts))
    
    def init():
        traj_line.set_data([], [])
        targets_scatter.set_offsets(np.empty((0, 2)))
        ax.set_title("Optimization Step: 0")
        return traj_line, targets_scatter,

    def update(frame):
        step, traj_arr, pts = precomputed[frame]
        # Update trajectory line if available
        if traj_arr.size > 0:
            traj_line.set_data(traj_arr[:, 0], traj_arr[:, 1])
        else:
            traj_line.set_data([], [])
        targets_scatter.set_offsets(pts)
        ax.set_title(f"Optimization Step: {step}/{len(precomputed)}")
        return traj_line, targets_scatter,
    
    ani = animation.FuncAnimation(fig, update, frames=len(precomputed),
                                  init_func=init, blit=True, interval=500)
    
    # Optionally switch to the 'ffmpeg' writer if available for better performance:
    try:
        ani.save(save_path, writer='ffmpeg', fps=25, dpi=300)
    except Exception as e:
        # Fall back to Pillow writer if ffmpeg is not available.
        print("FFmpeg writer failed, falling back to Pillow writer:", e)
        ani.save(save_path, writer='pillow', fps=25, dpi=300)
    
    plt.close(fig)


def main():
    global iterations, costs, optimization_history
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

    dimensions, x0 = build_dimensions(A, B, num_points, grid_step, max_world_size, perturbation_factor)

    # The cost function for optimization.
    def cost_function(param_list):
        nonlocal sim, A, B, num_points, grid_step, max_world_size, noise_gain, altitude_gain, time_gain, distance_gain, power_gain
        global iterations, costs, optimization_history
        iterations += 1

        custom_points = generate_custom_points(param_list, A, B, num_points, grid_step, max_world_size)
        print(f"Iteration ({iterations}/{n_iterations}) | Best: {min(costs):.2f} | ", end="")

        trajectory, total_cost, log_data, all_targets, simulation_completed = execute_simulation(
            sim, world, A, B, custom_points, cost_gains,
            showplots=False, interval=30, log_folder="Logs",
            dt=1, print_info=False, save_log=False
        )

        if simulation_completed:
            # Save the optimization step (iteration, trajectory, and intermediate targets)
            optimization_history.append((iterations, trajectory, all_targets))
        costs.append(total_cost if simulation_completed else np.nan)
        return total_cost

    print("Starting optimization...")
    start_time = time.time()
    result = gp_minimize(cost_function, dimensions, x0=x0, n_calls=n_iterations, random_state=params["optimization_random_state"])
    end_time = time.time()

    print("Optimal cost:", result.fun)
    print("Total optimization time: {:.2f} seconds".format(end_time - start_time))
    best_params = result.x

    # Generate custom points using the best found parameters.
    custom_points_best = generate_custom_points(best_params, A, B, num_points, grid_step, max_world_size)
    print("Best points:", custom_points_best)

    # Save best trajectory information.
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_folder = f"OptimizedTrajectory/{time_str}"
    os.makedirs(save_folder, exist_ok=True)
    np.save(f"{save_folder}/bestpoints.npy", custom_points_best)
    plotCosts(costs[1:], save=True, datetime=time_str)

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
        "angle_noise_model": params["noise_model"],
        "world_file_name": world.world_name
    }

    with open(f"{save_folder}/optimization_info.json", "w") as json_file:
        json.dump(optimization_info, json_file, indent=4)
    
    create_opt_ani = False
    if create_opt_ani:
        # Create and save animation of the optimization steps.
        anim_path = f"{save_folder}/optimization_animation.gif"
        if optimization_history:
            print("Creating animation of optimization steps...")
            animate_optimization_steps(world, grid_size, A, B, optimization_history, anim_path)
            print(f"Animation saved as {anim_path}")
        else:
            print("No completed simulation steps were recorded for animation.")
    
    print("Executing simulation...")
    execute_simulation(
        sim, world, A, B, custom_points_best, cost_gains,
        showplots=True, interval=30, log_folder=save_folder,
        dt=0.1, print_info=True, save_log=True, print_log=False
    )


if __name__ == "__main__":
    main()
