import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Entity.World import World

def show2DWorld(world: World, trajectories, A_list, B_list, all_targets, image_alpha=0.7, save=False, save_folder="OptimizedTrajectory"):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    if world.background_image is not None:
        bg_img = np.array(world.background_image)
        if bg_img.ndim == 3 and bg_img.shape[2] == 4:
            bg_img = bg_img[..., :3]
        if np.max(bg_img) > 1.0:
            bg_img = bg_img.astype(np.float32) / np.max(bg_img)
        ax.imshow(bg_img, extent=[0, world.max_world_size, 0, world.max_world_size],
                  origin='lower', alpha=image_alpha, zorder=-1)

    for (x, y, z), params in world.grid.items():
        if z == 0:
            rect = plt.Rectangle((x * world.grid_size, y * world.grid_size), world.grid_size, world.grid_size, 
                                 color=world.AREA_PARAMS[params]["color"], 
                                 alpha=world.AREA_PARAMS[params]["alpha"])
            ax.add_patch(rect)

    colors = ['green', 'blue', 'orange', 'purple', 'cyan']
    n_drones = len(trajectories)
    for i in range(n_drones):
        A = A_list[i]
        B = B_list[i]
        color = colors[i % len(colors)]
        ax.scatter(A["x"], A["y"], color=color, s=50, label=f"Drone {i+1} Start", zorder=5)
        ax.scatter(B["x"], B["y"], color=color, s=50, marker='x', label=f"Drone {i+1} Target", zorder=5)
        traj_arr = np.array(trajectories[i])
        ax.plot(traj_arr[:, 0], traj_arr[:, 1], '--', color=color, lw=1.5, zorder=3)
        # Plot intermediate targets with a high zorder
        targets = all_targets[i]
        for j, pt in enumerate(targets[1:-1], 1):
            ax.scatter(pt[0], pt[1], color='red', s=30, zorder=10)
            ax.text(pt[0] + 5, pt[1] + 5, f"{j}", color='red', fontsize=8, zorder=15)

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


def showPlot(trajectories, A_list, B_list, all_targets, world: World, grid_size, max_world_size, log_data, interval=1):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Prepare colors for drones
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    n_drones = len(trajectories)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # Create animated lines and points for each drone
    lines = []
    points = []
    for i in range(n_drones):
        line, = ax.plot([], [], [], '--', lw=1.5, color=colors[i % len(colors)])
        point, = ax.plot([], [], [], marker='o', color=colors[i % len(colors)], markersize=8, linestyle='None')
        lines.append(line)
        points.append(point)

    ax.set_xlim(0, max_world_size)
    ax.set_ylim(0, max_world_size)
    ax.set_zlim(0, max_world_size)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Multi-Drone Trajectory Animation')

    # Plot static start and target points for each drone
    for i in range(n_drones):
        A = A_list[i]
        B = B_list[i]
        color = colors[i % len(colors)]
        ax.scatter(A["x"], A["y"], A["z"], color=color, s=50, label=f'Drone {i+1} Start', zorder=5)
        ax.scatter(B["x"], B["y"], B["z"], color=color, s=50, marker='x', label=f'Drone {i+1} Target', zorder=5)
    
    # Plot and store the static intermediate points (waypoints) for each drone
    static_targets = []
    for i in range(n_drones):
        targets = all_targets[i]
        # Plot intermediate points (excluding start and target)
        xs = [pt[0] for pt in targets[1:-1]]
        ys = [pt[1] for pt in targets[1:-1]]
        zs = [pt[2] for pt in targets[1:-1]]
        scatter = ax.scatter(xs, ys, zs, color='black', s=30, zorder=10)
        static_targets.append(scatter)
        # Add labels to these points
        for j, pt in enumerate(targets[1:-1], 1):
            ax.text(pt[0], pt[1], pt[2], f"{j}", color='black', fontsize=8, zorder=15)

    log_text = ax.text2D(0, 0.05, "", transform=ax.transAxes, fontsize=8, color='purple')

    traj_arrays = [np.array(traj) for traj in trajectories]
    max_frames = max([len(arr) for arr in traj_arrays])

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        for point in points:
            point.set_data([], [])
            point.set_3d_properties([])
        log_text.set_text("")
        # Return animated objects; static targets are already drawn and will remain.
        return lines + points + [log_text]

    def update(frame):
        for i in range(n_drones):
            arr = traj_arrays[i]
            if frame < len(arr):
                lines[i].set_data(arr[:frame, 0], arr[:frame, 1])
                lines[i].set_3d_properties(arr[:frame, 2])
                # Show the current position as a point
                points[i].set_data(arr[frame-1:frame, 0], arr[frame-1:frame, 1])
                points[i].set_3d_properties(arr[frame-1:frame, 2])
        if frame < len(log_data):
            ld = log_data[frame]
            log_str = (f"Time: {ld['time']} s\n"
                       f"Drone {ld['drone_index']} | RPMs: {ld['rpms']}\n"
                       f"Velocity: {ld['velocity']}\n"
                       f"Altitude: {ld['position'][2]:.2f} m\n"
                       f"Pitch: {ld['pitch']} rad, Yaw: {ld['yaw']} rad")
            log_text.set_text(log_str)
        return lines + points + [log_text]

    ani = animation.FuncAnimation(fig, update, frames=max_frames,
                                  init_func=init, interval=interval, blit=True)
    plt.show()


def plotCosts(costs, save=True, datetime=None, folder="OptimizedTrajectory"):
    plt.plot(costs)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Costs over iterations")
    if save:
        plt.savefig(f"{folder}/{datetime}/costs.png", dpi=300)
    plt.show()
