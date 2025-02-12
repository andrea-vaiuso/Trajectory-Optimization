import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from World import World

def showPlot(trajectory, A, B, all_targets, world: World, grid_size, max_world_size, log_data, interval=1):
    traj_arr = np.array(trajectory)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(traj_arr[:, 0], traj_arr[:, 1], 'k-', linewidth=1.5, label="Trajectory")
    ax.scatter(A["x"], A["y"], color='green', s=50, label="A (Start)")
    ax.scatter(B["x"], B["y"], color='red', s=50, label="B (Target)")
    for i, pt in enumerate(all_targets[1:-1], 1):
        ax.scatter(pt[0], pt[1], color='blue', s=30)
        ax.text(pt[0]+5, pt[1]+5, f"{i}", color='blue', fontsize=8)
    for (x, y, z), params in world.grid.items():
        if z == 0:
            rect = plt.Rectangle((x * grid_size, y * grid_size), grid_size, grid_size, color=world.AREA_PARAMS[params]["color"], alpha=0.1)
            ax.add_patch(rect)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Drone 2D XY Trajectory with Waypoints")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.legend()
    ax.grid(True)
    plt.show()
        
    traj_arr = np.array(trajectory)
    x_data = traj_arr[:, 0]
    y_data = traj_arr[:, 1]
    z_data = traj_arr[:, 2]

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    line, = ax.plot([], [], [], 'k--', lw=1.5)
    point, = ax.plot([], [], [], 'ro', markersize=8)

    ax.set_xlim(0, max_world_size)
    ax.set_ylim(0, max_world_size)
    ax.set_zlim(0, max_world_size)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Drone Trajectory Animation')

    # Scatter marker per i waypoint
    ax.scatter(A["x"], A["y"], A["z"], color='green', s=50, label='A')
    for i, pt in enumerate(all_targets[1:-1], 1):
        ax.scatter(pt[0], pt[1], pt[2], color='blue', s=30)
        ax.text(pt[0], pt[1], pt[2], f"{i}", color='blue', fontsize=8)
    ax.scatter(B["x"], B["y"], B["z"], color='red', s=50, label='B')

    log_text = ax.text2D(0.05, 0.05, "", transform=ax.transAxes, fontsize=8, color='purple')

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        log_text.set_text("")
        return line, point, log_text

    def update(frame):
        line.set_data(x_data[:frame], y_data[:frame])
        line.set_3d_properties(z_data[:frame])
        point.set_data(x_data[frame-1:frame], y_data[frame-1:frame])
        point.set_3d_properties(z_data[frame-1:frame])
        if frame < len(log_data):
            ld = log_data[frame]
            log_str = (f"Time: {ld[0]} s\n"
                       f"RPMs: [{ld[6]}, {ld[7]}, {ld[8]}, {ld[9]}]\n"
                       f"Velocity: ({ld[10]}, {ld[11]}, {ld[12]}) - h ({round(np.sqrt(ld[10]**2+ld[11]**2),2)}, v ({round(ld[12])}))\n"
                       f"Pitch: {ld[4]} rad, Yaw: {ld[5]} rad")
            log_text.set_text(log_str)
        return line, point, log_text

    ani = animation.FuncAnimation(fig, update, frames=len(x_data),
                                  init_func=init, interval=interval, blit=True)
    plt.show()

def plotCosts(costs, save=True, datetime=None, folder="OptimizedTrajectory"):
    plt.plot(costs)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Costs over iterations")
    if save:
        plt.savefig(f"{folder}/{datetime}_costs.png")
    plt.show()