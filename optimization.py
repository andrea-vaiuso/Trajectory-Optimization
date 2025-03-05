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
costs_history = [1e9]
optimization_history = []  # Will store tuples: (iteration, trajectories, all_targets)

def get_cost_gains_all(A_list, B_list, drones):
    gains_list = []
    for A, B, drone in zip(A_list, B_list, drones):
        distAB = np.sqrt((B["x"] - A["x"])**2 + (B["y"] - A["y"])**2 + (B["z"] - A["z"])**2) * 1.1
        maxvel = np.sqrt(drone.max_horizontal_speed**2 + drone.max_vertical_speed**2)
        noise_rule_cost_gain = 1
        altitude_rule_cost_gain = 1
        time_cost_gain = maxvel / distAB
        distance_cost_gain = 1 / distAB
        power_cost_gain = time_cost_gain / drone.hover_rpm
        gains_list.append((noise_rule_cost_gain, altitude_rule_cost_gain, time_cost_gain, distance_cost_gain, power_cost_gain))
    return gains_list

def compute_base_point(t, A, B):
    """Compute the base point (linear interpolation) between A and B at fraction t."""
    return {
        "x": A["x"] + t * (B["x"] - A["x"]),
        "y": A["y"] + t * (B["y"] - A["y"]),
        "z": A["z"] + t * (B["z"] - A["z"])
    }

def generate_custom_points(param_list, A_list, B_list, num_points, grid_step, max_world_size):
    """
    Given a flat parameter list, generate a list (per drone) of custom points.
    Each point consists of 5 values (3 offsets and 2 speeds).
    """
    custom_points_all = []
    offset = 0
    for A, B in zip(A_list, B_list):
        custom_points = []
        for i in range(num_points):
            t = (i + 1) / (num_points + 1)
            base_point = compute_base_point(t, A, B)
            offset_x = int(round(param_list[offset + i*5 + 0])) * grid_step
            offset_y = int(round(param_list[offset + i*5 + 1])) * grid_step
            offset_z = int(round(param_list[offset + i*5 + 2])) * grid_step

            final_x = np.clip(base_point["x"] + offset_x, 0, max_world_size)
            final_y = np.clip(base_point["y"] + offset_y, 0, max_world_size)
            final_z = np.clip(base_point["z"] + offset_z, 0, max_world_size)

            point = {
                "x": final_x,
                "y": final_y,
                "z": final_z,
                "h_speed": param_list[offset + i*5 + 3],
                "v_speed": param_list[offset + i*5 + 4]
            }
            custom_points.append(point)
        custom_points_all.append(custom_points)
        offset += num_points * 5
    return custom_points_all

def build_dimensions(A_list, B_list, num_points, grid_step, max_world_size, perturbation_factor):
    dimensions = []
    x0 = []
    for A, B in zip(A_list, B_list):
        # Calculate overall distance between A and B for the perturbation range.
        distAB = np.sqrt((B["x"] - A["x"])**2 + (B["y"] - A["y"])**2 + (B["z"] - A["z"])**2)
        max_offset = perturbation_factor * distAB

        for i in range(num_points):
            t = (i + 1) / (num_points + 1)
            base_point = compute_base_point(t, A, B)

            lower_x = max(-max_offset, -base_point["x"])
            upper_x = min(max_offset, max_world_size - base_point["x"])
            lower_y = max(-max_offset, -base_point["y"])
            upper_y = min(max_offset, max_world_size - base_point["y"])
            lower_z = max(-max_offset, -base_point["z"])
            upper_z = min(max_offset, max_world_size - base_point["z"])

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
            x0.extend([0, 0, 0, A["h_speed"], A["v_speed"]])
    return dimensions, x0

def animate_optimization_steps(world: World, grid_size, A_list, B_list, optimization_history, save_path):
    """
    Create an animation of the optimization steps for multiple drones.
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
    
    # Plot static start and target points for each drone
    colors = ['green', 'blue', 'orange', 'purple', 'cyan']
    for idx, (A, B) in enumerate(zip(A_list, B_list)):
        color = colors[idx % len(colors)]
        ax.scatter(A["x"], A["y"], color=color, s=50, label=f"Drone {idx+1} Start")
        ax.scatter(B["x"], B["y"], color=color, s=50, marker='x', label=f"Drone {idx+1} Target")
    
    # Create dynamic objects for trajectories and intermediate targets for each drone
    traj_lines = []
    targets_scatters = []
    n_drones = len(A_list)
    for i in range(n_drones):
        line, = ax.plot([], [], 'k--', lw=1.5)
        scatter = ax.scatter([], [], s=30)
        traj_lines.append(line)
        targets_scatters.append(scatter)
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(0, world.max_world_size)
    ax.set_ylim(0, world.max_world_size)
    ax.legend(loc='upper left')
    ax.grid(True)
    
    # Precompute the data arrays for each frame for each drone
    precomputed = []
    for step, trajectories, all_targets in optimization_history:
        drones_traj = []
        drones_targets = []
        for traj in trajectories:
            if traj and len(traj) > 0:
                traj_arr = np.array(traj)[:, :2]
            else:
                traj_arr = np.empty((0, 2))
            drones_traj.append(traj_arr)
        for targets in all_targets:
            if targets and len(targets) > 2:
                pts = np.array(targets[1:-1])[:, :2]
            else:
                pts = np.empty((0, 2))
            drones_targets.append(pts)
        precomputed.append((step, drones_traj, drones_targets))
    
    def init():
        for line in traj_lines:
            line.set_data([], [])
        for scatter in targets_scatters:
            scatter.set_offsets(np.empty((0, 2)))
        ax.set_title("Optimization Step: 0")
        return traj_lines + targets_scatters

    def update(frame):
        step, drones_traj, drones_targets = precomputed[frame]
        for i in range(n_drones):
            if drones_traj[i].size > 0:
                traj_lines[i].set_data(drones_traj[i][:, 0], drones_traj[i][:, 1])
            else:
                traj_lines[i].set_data([], [])
            targets_scatters[i].set_offsets(drones_targets[i])
        ax.set_title(f"Optimization Step: {step}/{len(precomputed)}")
        return traj_lines + targets_scatters
    
    ani = animation.FuncAnimation(fig, update, frames=len(precomputed),
                                  init_func=init, blit=True, interval=500)
    
    try:
        ani.save(save_path, writer='ffmpeg', fps=25, dpi=300)
    except Exception as e:
        print("FFmpeg writer failed, falling back to Pillow writer:", e)
        ani.save(save_path, writer='pillow', fps=25, dpi=300)
    
    plt.close(fig)

def execute_simulation(sim: Simulation, world: World, 
                        A_list, B_list, custom_points_all, cost_gains,
                        showplots=True, interval=30, log_folder="Logs",
                        dt=0.1, print_info = False, save_log = False, print_log = False
                       ):
    # Aggregate cost gains: For simplicity, average over drones
    # Alternatively, we could pass list to simulation.
    trajectories, total_cost, log_data, all_targets, simulation_completed = sim.simulate_trajectory(
        points_a=A_list, points_b=B_list, custom_points_all=custom_points_all, dt=dt,
        horizontal_threshold=5.0, vertical_threshold=2.0,
        print_log=print_log, noise_annoyance_radius=100,
        noise_rule_cost_gain=cost_gains[0][0],  # Assuming similar gains for all drones
        altitude_rule_cost_gain=cost_gains[0][1],
        time_cost_gain=cost_gains[0][2],
        distance_cost_gain=cost_gains[0][3],
        power_cost_gain=cost_gains[0][4],
        collision_threshold=2.0, collision_cost_gain=1e6,
        time_limit_gain=10, save_log=save_log, save_log_folder=log_folder, print_info=print_info
    )
    if showplots:
        show2DWorld(world, trajectories, A_list, B_list, all_targets, save=True, save_folder=log_folder)    
        showPlot(trajectories, A_list, B_list, all_targets, world, world.grid_size, world.max_world_size, log_data, interval=interval)
    return trajectories, total_cost, log_data, all_targets, simulation_completed

def optimize(params, world, drones):
    global iterations, costs_history, optimization_history
    grid_size = params["grid_size"]
    max_world_size = params["max_world_size"]
    num_points = params["num_points"]
    n_iterations = params["n_iterations"]
    perturbation_factor = params["perturbation_factor"]
    grid_step = params["grid_step"]
    A_list = params["A"]
    B_list = params["B"]

    print("Loading noise model...")
    angle_noise_model = np.load(params["noise_model"])
    
    print("Initializing simulation...")
    sim = Simulation(drones, world, angle_noise_model)
    cost_gains = get_cost_gains_all(A_list, B_list, drones)
    
    dimensions, x0 = build_dimensions(A_list, B_list, num_points, grid_step, max_world_size, perturbation_factor)

    def cost_function(param_list):
        nonlocal sim, A_list, B_list, num_points, grid_step, max_world_size
        global iterations, costs_history, optimization_history
        iterations += 1

        custom_points_all = generate_custom_points(param_list, A_list, B_list, num_points, grid_step, max_world_size)
        print(f"Iteration ({iterations}/{n_iterations}) | Best: {min(costs_history):.2f} | ", flush=True)

        trajectories, total_cost, log_data, all_targets, simulation_completed = execute_simulation(
            sim, world, A_list, B_list, custom_points_all, cost_gains,
            showplots=False, interval=30, log_folder="Logs",
            dt=1, print_info=False, save_log=False
        )

        if simulation_completed:
            optimization_history.append((iterations, trajectories, all_targets))
        costs_history.append(total_cost if simulation_completed else np.nan)
        return total_cost

    print("Starting optimization...")
    start_time = time.time()
    result = gp_minimize(cost_function, dimensions, x0=x0, n_calls=n_iterations, random_state=params["optimization_random_state"])
    end_time = time.time()

    print("Optimal cost:", result.fun)
    print("Total optimization time: {:.2f} seconds".format(end_time - start_time))
    best_params = result.x

    custom_points_best = generate_custom_points(best_params, A_list, B_list, num_points, grid_step, max_world_size)
    print("Best points:", custom_points_best)

    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_folder = f"OptimizedTrajectory/{time_str}"
    os.makedirs(save_folder, exist_ok=True)
    np.save(f"{save_folder}/bestpoints.npy", custom_points_best)
    plotCosts(costs_history[1:], save=True, datetime=time_str)

    optimization_info = {
        "n_iterations": int(n_iterations),
        "best_cost": float(result.fun),
        "optimization_time_seconds": float(end_time - start_time),
        "n_points": int(num_points),
        "custom_points": custom_points_best,
        "A": [ {k: float(v) for k, v in point.items()} for point in A_list ],
        "B": [ {k: float(v) for k, v in point.items()} for point in B_list ],
        "grid_size": int(grid_size),
        "max_world_size": int(max_world_size),
        "perturbation_factor": float(perturbation_factor),
        "grid_step": int(grid_step),
        "noise_rule_cost_gain": float(cost_gains[0][0]),
        "altitude_rule_cost_gain": float(cost_gains[0][1]),
        "time_cost_gain": float(cost_gains[0][2]),
        "distance_cost_gain": float(cost_gains[0][3]),
        "power_cost_gain": float(cost_gains[0][4]),
        "drones": [drone.to_dict() for drone in drones],
        "angle_noise_model": params["noise_model"],
        "world_file_name": world.world_name
    }

    with open(f"{save_folder}/optimization_info.json", "w") as json_file:
        json.dump(optimization_info, json_file, indent=4)
    
    create_opt_ani = False
    if create_opt_ani:
        anim_path = f"{save_folder}/optimization_animation.gif"
        if optimization_history:
            print("Creating animation of optimization steps...")
            animate_optimization_steps(world, grid_size, A_list, B_list, optimization_history, anim_path)
            print(f"Animation saved as {anim_path}")
        else:
            print("No completed simulation steps were recorded for animation.")
    
    return sim, world, A_list, B_list, custom_points_best, cost_gains, save_folder
