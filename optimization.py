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
optimization_history = []  # Will store tuples: (iteration, trajectories, all_targets)

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

        # Calculate lower and upper bounds for each offset.
        lower_x = max(-max_offset, -base_point["x"])
        upper_x = min(max_offset, max_world_size - base_point["x"])
        lower_y = max(-max_offset, -base_point["y"])
        upper_y = min(max_offset, max_world_size - base_point["y"])
        lower_z = max(-max_offset, -base_point["z"])
        upper_z = min(max_offset, max_world_size - base_point["z"])

        # Discrete bounds.
        lx_disc = int(np.ceil(lower_x / grid_step))
        ux_disc = int(np.floor(upper_x / grid_step))
        ly_disc = int(np.ceil(lower_y / grid_step))
        uy_disc = int(np.floor(upper_y / grid_step))
        lz_disc = int(np.ceil(lower_z / grid_step))
        uz_disc = int(np.floor(upper_z / grid_step))

        dimensions.extend([
            Integer(lx_disc, ux_disc),  # offset_x
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
                       A_list, B_list, custom_points_list, cost_gains_list, 
                       showplots=True, interval=10, log_folder="Logs",
                       dt=0.1, print_info=False, save_log=True, print_log=False,
                       collision_distance=2.0, collision_cost=1e6):
    trajectories, total_cost, log_data, all_targets, simulation_completed = sim.simulate_trajectory(
        point_a_list=A_list, point_b_list=B_list,
        custom_points_list=custom_points_list,
        dt=dt, horizontal_threshold=5.0, vertical_threshold=2.0,
        print_log=print_log, noise_annoyance_radius=100,
        cost_gains_list=cost_gains_list, time_limit_gain=10,
        collision_distance=collision_distance, collision_cost=collision_cost,
        save_log=save_log, save_log_folder=log_folder, print_info=print_info
    )
    if showplots:
        show2DWorld(world, world.grid_size, trajectories, A_list, B_list, all_targets,
                    save=True, save_folder=log_folder)    
        showPlot(trajectories, A_list, B_list, all_targets, world, world.grid_size, world.max_world_size, log_data, interval=interval)
    return trajectories, total_cost, log_data, all_targets, simulation_completed

def animate_optimization_steps(world: World, grid_size, A_list, B_list, optimization_history, save_path):
    """
    Animate the optimization steps for multiple drones.
    Each drone’s trajectory and intermediate targets are drawn in a different color.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Background and grid.
    if world.background_image is not None:
        bg_img = np.array(world.background_image)
        ax.imshow(bg_img, extent=[0, world.max_world_size, 0, world.max_world_size],
                  origin='lower', alpha=0.7, zorder=-1)
    
    for (x, y, z), params in world.grid.items():
        if z == 0:
            rect = plt.Rectangle((x * grid_size, y * grid_size), grid_size, grid_size,
                                 color=world.AREA_PARAMS[params]["color"],
                                 alpha=world.AREA_PARAMS[params]["alpha"])
            ax.add_patch(rect)
    
    # Plot start and target points for each drone.
    colors = ['green', 'blue', 'magenta', 'orange', 'cyan']
    n = len(A_list)
    for i in range(n):
        ax.scatter(A_list[i]["x"], A_list[i]["y"], color=colors[i % len(colors)], s=50, label=f"Drone {i+1} A")
        ax.scatter(B_list[i]["x"], B_list[i]["y"], color=colors[(i+1) % len(colors)], s=50, label=f"Drone {i+1} B")
    
    # Create dynamic objects for trajectories and intermediate targets.
    traj_lines = [ax.plot([], [], linestyle='--', lw=1.5, color=colors[i % len(colors)])[0] for i in range(n)]
    targets_scatter = ax.scatter([], [], color='red', s=30)
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(0, world.max_world_size)
    ax.set_ylim(0, world.max_world_size)
    ax.legend(loc='upper left')
    ax.grid(True)
    
    # Precompute frame data.
    precomputed = []
    for step, traj_list, targets_list in optimization_history:
        # For each drone, get its trajectory (projected to XY) and targets.
        trajs = []
        pts = []
        for traj in traj_list:
            if traj and len(traj) > 0:
                traj_arr = np.array(traj)[:, :2]
            else:
                traj_arr = np.empty((0, 2))
            trajs.append(traj_arr)
        for targets in targets_list:
            if targets and len(targets) > 0:
                pts.extend(np.array(targets)[1:-1, :2])
        precomputed.append((step, trajs, np.array(pts) if len(pts) > 0 else np.empty((0,2))))
    
    def init():
        for line in traj_lines:
            line.set_data([], [])
        targets_scatter.set_offsets(np.empty((0, 2)))
        ax.set_title("Optimization Step: 0")
        return traj_lines + [targets_scatter]

    def update(frame):
        step, trajs, pts = precomputed[frame]
        for i, line in enumerate(traj_lines):
            if len(trajs[i]) > 0:
                line.set_data(trajs[i][:, 0], trajs[i][:, 1])
            else:
                line.set_data([], [])
        targets_scatter.set_offsets(pts)
        ax.set_title(f"Optimization Step: {step}/{len(precomputed)}")
        return traj_lines + [targets_scatter]
    
    ani = animation.FuncAnimation(fig, update, frames=len(precomputed),
                                  init_func=init, blit=True, interval=500)
    
    try:
        ani.save(save_path, writer='ffmpeg', fps=25, dpi=300)
    except Exception as e:
        print("FFmpeg writer failed, falling back to Pillow writer:", e)
        ani.save(save_path, writer='pillow', fps=25, dpi=300)
    
    plt.close(fig)

def optimize(params, world, drones):
    global iterations, costs, optimization_history
    grid_size = params["grid_size"]
    max_world_size = params["max_world_size"]
    num_points = params["num_points"]
    n_iterations = params["n_iterations"]
    perturbation_factor = params["perturbation_factor"]
    grid_step = params["grid_step"]
    A_list = params["A"]  # Now a list of dictionaries
    B_list = params["B"]  # Now a list of dictionaries

    print("Loading noise model...")
    angle_noise_model = np.load(params["noise_model"])
    
    print("Initializing simulation...")
    sim = Simulation(drones, world, angle_noise_model)
    # Compute cost gains for each drone.
    cost_gains_list = [get_cost_gains(A_list[i], B_list[i], drones[i]) for i in range(len(drones))]

    # Build dimensions and initial guess for all drones.
    dimensions = []
    x0 = []
    for i in range(len(drones)):
        dims, guess = build_dimensions(A_list[i], B_list[i], num_points, grid_step, max_world_size, perturbation_factor)
        dimensions.extend(dims)
        x0.extend(guess)

    def cost_function(param_list):
        nonlocal sim, A_list, B_list, num_points, grid_step, max_world_size
        global iterations, costs, optimization_history
        iterations += 1

        # Split parameter list into segments for each drone.
        custom_points_list = []
        segment_size = num_points * 5
        for i in range(len(drones)):
            segment = param_list[i * segment_size : (i+1) * segment_size]
            custom_points = generate_custom_points(segment, A_list[i], B_list[i], num_points, grid_step, max_world_size)
            custom_points_list.append(custom_points)
        
        print(f"Iteration ({iterations}/{n_iterations}) | Best: {min(costs):.2f} | ", end="", flush=True)

        trajectory, total_cost, log_data, all_targets, simulation_completed = execute_simulation(
            sim, world, A_list, B_list, custom_points_list, cost_gains_list,
            showplots=False, interval=30, log_folder="Logs",
            dt=1, print_info=True, save_log=False,
            collision_distance=params.get("collision_distance", 2.0),
            collision_cost=params.get("collision_cost", 1e6)
        )

        # Instead of returning NaN if simulation is not complete, return a high penalty cost.
        if not simulation_completed:
            total_cost = 1e10

        # Save optimization history only if simulation completed.
        if simulation_completed:
            optimization_history.append((iterations, trajectory, all_targets))
        
        costs.append(total_cost)
        return total_cost


    print("Starting optimization...")
    start_time = time.time()
    result = gp_minimize(cost_function, dimensions, x0=x0, n_calls=n_iterations, random_state=params["optimization_random_state"])
    end_time = time.time()

    print("Optimal cost:", result.fun)
    print("Total optimization time: {:.2f} seconds".format(end_time - start_time))
    best_params = result.x

    # Generate custom points using the best found parameters.
    best_custom_points_list = []
    segment_size = num_points * 5
    for i in range(len(drones)):
        segment = best_params[i * segment_size : (i+1) * segment_size]
        custom_points = generate_custom_points(segment, A_list[i], B_list[i], num_points, grid_step, max_world_size)
        best_custom_points_list.append(custom_points)
    print("Best points:", best_custom_points_list)

    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_folder = f"OptimizedTrajectory/{time_str}"
    os.makedirs(save_folder, exist_ok=True)
    np.save(f"{save_folder}/bestpoints.npy", best_custom_points_list)
    plotCosts(costs[1:], save=True, datetime=time_str)

    optimization_info = {
        "n_iterations": int(n_iterations),
        "best_cost": float(result.fun),
        "optimization_time_seconds": float(end_time - start_time),
        "n_points": int(num_points),
        "custom_points": best_custom_points_list,
        "A": A_list,
        "B": B_list,
        "grid_size": int(grid_size),
        "max_world_size": int(max_world_size),
        "perturbation_factor": float(perturbation_factor),
        "grid_step": int(grid_step),
        "noise_rule_cost_gain": float(cost_gains_list[0][0]),
        "altitude_rule_cost_gain": float(cost_gains_list[0][1]),
        "time_cost_gain": float(cost_gains_list[0][2]),
        "distance_cost_gain": float(cost_gains_list[0][3]),
        "power_cost_gain": float(cost_gains_list[0][4]),
        "drones": [drone.to_dict() for drone in drones],
        "angle_noise_model": params["noise_model"],
        "world_file_name": world.world_name,
        "collision_distance": params.get("collision_distance", 2.0),
        "collision_cost": params.get("collision_cost", 1e6)
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
    
    return sim, world, A_list, B_list, best_custom_points_list, cost_gains_list, save_folder

if __name__ == "__main__":
    # For testing purposes only.
    pass
