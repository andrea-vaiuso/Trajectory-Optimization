import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Entity.World import World

def show2DWorld(world: World, grid_size, trajectories=None, A_list=None, B_list=None, all_targets=None, image_alpha=0.7, save=False, save_folder="OptimizedTrajectory"):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Background image.
    if world.background_image is not None:
        bg_img = np.array(world.background_image)
        ax.imshow(bg_img, extent=[0, world.max_world_size, 0, world.max_world_size],
                  origin='lower', alpha=image_alpha, zorder=-1)

    # Draw grid.
    for (x, y, z), params in world.grid.items():
        if z == 0:
            rect = plt.Rectangle((x * grid_size, y * grid_size), grid_size, grid_size, 
                                 color=world.AREA_PARAMS[params]["color"], 
                                 alpha=world.AREA_PARAMS[params]["alpha"])
            ax.add_patch(rect)

    # Define colors for drones.
    colors = ['green', 'blue', 'magenta', 'orange', 'cyan']
    n = len(A_list) if A_list is not None else 0

    if trajectories is not None and A_list is not None and B_list is not None:
        for i in range(n):
            # Plot starting point (A) as a circle.
            ax.scatter(A_list[i]["x"], A_list[i]["y"], color=colors[i % len(colors)], s=50, marker='o', label=f"Drone {i+1} A")
            # Plot destination (B) as a circle.
            ax.scatter(B_list[i]["x"], B_list[i]["y"], color=colors[(i+1) % len(colors)], s=50, marker='o', label=f"Drone {i+1} B")
            # Plot intermediate targets with the same color as the drone.
            if all_targets is not None and len(all_targets) > i and all_targets[i]:
                # Convert each target (assumed to be a list: [x, y]) to a numpy array.
                pts = np.array([[pt[0], pt[1]] for pt in all_targets[i]])
                if pts.size > 0:
                    ax.scatter(pts[:, 0], pts[:, 1], color=colors[i % len(colors)], s=30)
                    for j, pt in enumerate(pts, 1):
                        ax.text(pt[0] + 5, pt[1] + 5, f"{j}", color=colors[i % len(colors)], fontsize=8)
            # Plot trajectory.
            traj_arr = np.array(trajectories[i])
            ax.plot(traj_arr[:, 0], traj_arr[:, 1], linestyle='--', lw=1.5, color=colors[i % len(colors)])
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"2D World '{world.world_name}' XY")
    ax.set_xlim(0, world.max_world_size)
    ax.set_ylim(0, world.max_world_size)
    ax.legend(loc='upper left')
    ax.grid(True)
    if save:
        plt.savefig(f"{save_folder}/trajectory_{world.world_name}.png", dpi=300)
    plt.show()


def showPlot(trajectories, A_list, B_list, all_targets, world: World, grid_size, max_world_size, log_data, interval=50):
    colors = ['green', 'blue', 'magenta', 'orange', 'cyan']
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # Create line objects for each drone trajectory.
    lines = []
    for i in range(len(trajectories)):
        line, = ax.plot([], [], [], linestyle='--', lw=1.5, color=colors[i % len(colors)], label=f"Drone {i+1} Path")
        lines.append(line)
    
    # Create scatter markers for the current drone positions with triangle markers.
    current_markers = []
    for i in range(len(trajectories)):
        marker = ax.scatter([], [], [], color=colors[i % len(colors)], marker='^', s=80, label=f"Drone {i+1}")
        current_markers.append(marker)

    ax.set_xlim(0, max_world_size)
    ax.set_ylim(0, max_world_size)
    ax.set_zlim(0, max_world_size)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Drone Trajectories Animation')
    
    # Plot start and destination points with circle markers.
    n = len(A_list)
    for i in range(n):
        ax.scatter(A_list[i]["x"], A_list[i]["y"], A_list[i]["z"],
                   color=colors[i % len(colors)], s=50, marker='o', label=f"Drone {i+1} A")
        ax.scatter(B_list[i]["x"], B_list[i]["y"], B_list[i]["z"],
                   color=colors[(i+1) % len(colors)], s=50, marker='o', label=f"Drone {i+1} B")
        # Plot intermediate targets with the same color as the drone.
        if all_targets is not None and len(all_targets) > i and all_targets[i]:
            pts = np.array([[pt[0], pt[1], pt[2]] for pt in all_targets[i]])
            if pts.size > 0:
                ax.scatter(pts[:,0], pts[:,1], pts[:,2], color=colors[i % len(colors)], s=30)
                for j, pt in enumerate(pts, 1):
                    ax.text(pt[0], pt[1], pt[2], f"{j}", color=colors[i % len(colors)], fontsize=8)

    log_text = ax.text2D(0, 0.05, "", transform=ax.transAxes, fontsize=8, color='purple')
    
    # Precompute data for animation.
    traj_data = []
    for traj in trajectories:
        traj_arr = np.array(traj)
        traj_data.append(traj_arr)
    max_frames = max([arr.shape[0] for arr in traj_data]) if traj_data else 0

    # Use the first drone's log entries for annotation.
    if isinstance(log_data, list) and len(log_data) > 0 and isinstance(log_data[0], list):
         log_entries = log_data[0]
    else:
         log_entries = log_data

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        for marker in current_markers:
            marker._offsets3d = ([], [], [])
        log_text.set_text("")
        return lines + current_markers + [log_text]

    def update(frame):
        for i, line in enumerate(lines):
            arr = traj_data[i]
            if frame < arr.shape[0]:
                line.set_data(arr[:frame, 0], arr[:frame, 1])
                line.set_3d_properties(arr[:frame, 2])
                current_point = arr[frame-1] if frame > 0 else arr[0]
                current_markers[i]._offsets3d = ([current_point[0]], [current_point[1]], [current_point[2]])
            else:
                line.set_data(arr[:, 0], arr[:, 1])
                line.set_3d_properties(arr[:, 2])
                current_point = arr[-1]
                current_markers[i]._offsets3d = ([current_point[0]], [current_point[1]], [current_point[2]])
        if log_entries and frame < len(log_entries):
            ld = log_entries[frame]
            log_str = (f"Time: {ld[0]:.2f} s\n"
                       f"RPMs: [{ld[6]}, {ld[7]}, {ld[8]}, {ld[9]}]\n"
                       f"Velocity: ({ld[10]}, {ld[11]}, {ld[12]})\n"
                       f"Altitude: {ld[3]:.2f} m\n"
                       f"Pitch: {ld[4]:.2f} rad, Yaw: {ld[5]:.2f} rad")
            log_text.set_text(log_str)
        return lines + current_markers + [log_text]
    
    ani = animation.FuncAnimation(fig, update, frames=max_frames,
                                  init_func=init, interval=interval)
    plt.legend()
    plt.show()


def plotCosts(costs, save=True, datetime=None, folder="OptimizedTrajectory"):
    plt.figure()
    # If a very high penalty is present, use a logarithmic scale.
    if max(costs) > 1e4:
        plt.semilogy(costs, marker='o')
    else:
        plt.plot(costs, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Costs over iterations")
    if save:
        plt.savefig(f"{folder}/{datetime}/costs.png", dpi=300)
    plt.show()
