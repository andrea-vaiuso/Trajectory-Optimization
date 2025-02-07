import csv
import numpy as np
import time
import matplotlib.pyplot as plt
import math

# ---------------- World Class ----------------
class World:
    def __init__(self, grid_size=10, max_world_size=1000):
        self.grid_size = grid_size
        self.max_world_size = max_world_size
        self.grid = {}

    def get_area(self, x, y, z):
        return (x // self.grid_size, y // self.grid_size, z // self.grid_size)

    def set_area_parameters(self, x_1, x_2, y_1, y_2, parameters):
        for x in range(x_1, x_2 + 1, self.grid_size):
            for y in range(y_1, y_2 + 1, self.grid_size):
                for z in range(0, self.max_world_size * self.grid_size, self.grid_size):
                    area = self.get_area(x, y, z)
                    if area not in self.grid:
                        self.grid[area] = {}
                    self.grid[area].update(parameters)

    def get_area_parameters(self, x, y, z):
        area = self.get_area(x, y, z)
        return self.grid.get(area, {})

    def get_area_center_point(self, x, y, z):
        area = self.get_area(x, y, z)
        center = [(a * self.grid_size) + self.grid_size / 2 for a in area]
        return tuple(center)

    def save_world(self, filename):
        with open(filename, 'w') as file:
            import json
            json.dump(self.grid, file)

    def load_world(self, filename):
        with open(filename, 'r') as file:
            import json
            self.grid = json.load(file)

# --------------- Simplified Drone Class ----------------
import numpy as np
import math

class Drone:
    def __init__(self, 
                 model_name, 
                 x, y, z,
                 mass=9.0, 
                 g=9.81, 
                 base_RPM=1800,
                 max_RPM=4000,
                 max_horizontal_speed=15.0,  
                 max_vertical_speed=5.0):
        self.model_name = model_name
        self.rpm_values = np.array([base_RPM]*4, dtype=float)
        self.position = np.array([x, y, z], dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.mass = mass
        self.g = g

        self.base_RPM = base_RPM
        self.max_RPM = max_RPM

        self.max_horizontal_speed = max_horizontal_speed  
        self.max_vertical_speed = max_vertical_speed  
        self.speed = max_horizontal_speed
        
        self.pitch = 0.0
        self.yaw = 0.0

        self.min_hover_thrust = self.mass * self.g
        self.min_hover_rpm = 2000

        # Variabili per il controllo PID verticale:
        self.altitude_integral = 0.0
        self.altitude_prev_error = 0.0

    def update_rpms(self, target, dt):
        # Calcola l'errore verticale
        altitude_error = target[2] - self.position[2]
        
        # RPM ideali per l'hover
        desired_hover_rpm = 2060
        
        # Parametri PID (regolabili per ridurre oscillazioni)
        k_p = 1500  # Ridotto per smorzare la risposta
        k_i = 100    # Ridotto per evitare integrale windup
        k_d = 1500   # Ridotto per ridurre risposta troppo aggressiva

        # Aggiorna il termine integrativo
        self.altitude_integral += altitude_error * dt
        self.altitude_integral = np.clip(self.altitude_integral, -100, 100)
        
        # Calcola il termine derivativo
        d_error = (altitude_error - self.altitude_prev_error) / dt
        self.altitude_prev_error = altitude_error
        
        # Calcola la correzione PID
        pid_correction = k_p * altitude_error + k_i * self.altitude_integral + k_d * d_error

        # Componente orizzontale
        horizontal_distance = np.linalg.norm(target[:2] - self.position[:2])
        decel_distance = 20.0
        horizontal_component = (self.max_RPM - self.base_RPM) * min(horizontal_distance / decel_distance, 1.0)
        
        # Somma le componenti
        desired_avg_rpm = desired_hover_rpm + pid_correction + horizontal_component
        desired_avg_rpm = np.clip(desired_avg_rpm, self.base_RPM, self.max_RPM)
        
        # Controllo sulla velocità di variazione degli RPM
        rpm_update_gain = 50  # Ridotto per rallentare il cambiamento
        max_rpm_change = 600 * dt  # Limita il cambiamento massimo degli RPM per evitare variazioni brusche
        
        # Aggiornamento degli RPM con velocità limitata
        for i in range(4):
            error = desired_avg_rpm - self.rpm_values[i]
            rpm_adjustment = rpm_update_gain * error * dt
            rpm_adjustment = np.clip(rpm_adjustment, -max_rpm_change, max_rpm_change)  # Limita il tasso di cambiamento
            
            self.rpm_values[i] += rpm_adjustment
            self.rpm_values[i] = np.clip(self.rpm_values[i], self.base_RPM, self.max_RPM)


    def update_physics(self, target, dt):
        # MOVIMENTO ORIZZONTALE
        horizontal_target = np.array([target[0], target[1]])
        horizontal_position = np.array([self.position[0], self.position[1]])
        horizontal_direction = horizontal_target - horizontal_position
        horizontal_distance = np.linalg.norm(horizontal_direction)

        decel_distance = 20.0
        if horizontal_distance > 0:
            factor = min(horizontal_distance / decel_distance, 1.0)
            horizontal_velocity = self.speed * factor * (horizontal_direction / horizontal_distance)
            horizontal_step = horizontal_velocity * dt
            if np.linalg.norm(horizontal_step) > horizontal_distance:
                horizontal_position = horizontal_target.copy()
            else:
                horizontal_position += horizontal_step
        else:
            horizontal_velocity = np.zeros(2)

        self.position[0] = horizontal_position[0]
        self.position[1] = horizontal_position[1]

        self.velocity[0] = horizontal_velocity[0]
        self.velocity[1] = horizontal_velocity[1]

        # MOVIMENTO VERTICALE
        thrust = self.get_thrust()
        vertical_acceleration = thrust / self.mass - self.g
        self.vertical_acceleration = vertical_acceleration

        self.velocity[2] += vertical_acceleration * dt
        self.velocity[2] = np.clip(self.velocity[2], -self.max_vertical_speed, self.max_vertical_speed)

        self.position[2] += self.velocity[2] * dt
        self.position[2] = max(self.position[2], 0)  # Prevent going below ground level

    def get_thrust(self, k=7.056):
        return k* (2e-6 * np.average(self.rpm_values)**2)

    def update_control(self, target, dt):
        self.update_rpms(target, dt)
        
        direction = target - self.position
        horizontal_distance = math.sqrt(direction[0]**2 + direction[1]**2)
        if np.linalg.norm(direction) > 0:
            self.yaw = math.atan2(direction[1], direction[0])
            self.pitch = math.atan2(-direction[2], horizontal_distance)
        
        self.update_physics(target, dt)
        return self.pitch, self.yaw, self.rpm_values.copy(), self.position.copy(), self.velocity.copy()


# --------------- Simulation Class (with CSV Logging and Plotting) ---------------
class Simulation:
    def __init__(self, drone: Drone, world: World):
        self.drone = drone
        self.world = world

    def generate_intermediate_points(self, point_a, point_b, custom_points=None):
        if custom_points is not None:
            return [np.array(point_a, dtype=float)] + custom_points + [np.array(point_b, dtype=float)]
        else:
            num_points = 10
            points = [np.array(point_a, dtype=float)]
            for i in range(1, num_points + 1):
                inter = np.array(point_a, dtype=float) + (np.array(point_b, dtype=float) - np.array(point_a, dtype=float)) * (i / (num_points + 1))
                points.append(inter)
            points.append(np.array(point_b, dtype=float))
            return points

    def simulate_trajectory(self, point_a, point_b, dt=0.1, horizontal_threshold=5.0, vertical_threshold=2.0, custom_points=None):
        start_time = time.time()
        point_a = np.array(point_a, dtype=float)
        point_b = np.array(point_b, dtype=float)
        targets = self.generate_intermediate_points(point_a, point_b, custom_points=custom_points)
        trajectory = [self.drone.position.copy()]
        total_distance = 0.0  
        t_elapsed = 0
        log_data = []  # Log: time, pos, pitch, yaw, rpms, velocity

        while targets:
            target = targets[0]
            print(f"Moving to target: {target}")
            horizontal_err = np.linalg.norm(self.drone.position[:2] - target[:2])
            vertical_err = abs(self.drone.position[2] - target[2])
            while horizontal_err > horizontal_threshold or vertical_err > vertical_threshold:
                previous_position = self.drone.position.copy()
                pitch, yaw, rpms, pos, vel = self.drone.update_control(target, dt)
                step_distance = np.linalg.norm(self.drone.position - previous_position)
                total_distance += step_distance
                trajectory.append(self.drone.position.copy())
                t_elapsed += dt
                horizontal_err = np.linalg.norm(self.drone.position[:2] - target[:2])
                vertical_err = abs(self.drone.position[2] - target[2])
                log_data.append([round(t_elapsed, 2), pos[0], pos[1], pos[2],
                                 round(pitch, 2), round(yaw, 2),
                                 int(rpms[0]), int(rpms[1]), int(rpms[2]), int(rpms[3]),
                                 round(vel[0], 2), round(vel[1], 2), round(vel[2], 2)])
                print(log_data[-1])
            print(f"Target reached: {target}\n\n\n\n\n")
            targets.pop(0)
        
        elapsed = time.time() - start_time
        print(f"Simulation completed in {elapsed:.2f} seconds.")
        print(f"The drone reached all targets in {t_elapsed:.2f} seconds.")
        print(f"Total distance traveled: {total_distance:.2f} meters.")

        csv_filename = "drone_simulation_log.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time(s)", "X", "Y", "Z", "Pitch", "Yaw", "FL", "FR", "RL", "RR", "Vx", "Vy", "Vz"])
            writer.writerows(log_data)
        print(f"Log saved to {csv_filename}.")
        return trajectory, log_data, self.generate_intermediate_points(point_a, point_b, custom_points=custom_points)

# ----------------- Main Usage -----------------
if __name__ == "__main__":
    world = World(grid_size=10, max_world_size=1000)
    world.set_area_parameters(0, 20, 0, 20, {'temperature': 20, 'humidity': 50})

    A = [0, 0, 50]
    B = [1000, 1000, 50]

    np.random.seed(42)
    custom_points = []
    for _ in range(10):
        rand_x = np.random.uniform(A[0], B[0])
        rand_y = np.random.uniform(A[1], B[1])
        rand_z = np.random.uniform(A[2], B[2])
        custom_points.append(np.array([rand_x, rand_y, rand_z], dtype=float))

    drone = Drone(
        model_name="SimpleDrone",
        x=A[0], y=A[1], z=A[2],
    )

    sim = Simulation(drone, world)
    trajectory, log_data, all_targets = sim.simulate_trajectory(point_a=A, point_b=B, dt=0.1,
                                                                  horizontal_threshold=5.0, vertical_threshold=2.0,)
                                                                  # custom_points=custom_points)

    # # Plot 2D con traiettoria e waypoint etichettati
    # traj_arr = np.array(trajectory)
    # plt.figure(figsize=(8, 6))
    # plt.plot(traj_arr[:, 0], traj_arr[:, 1], 'k-', linewidth=1.5, label="Trajectory")
    # plt.scatter(A[0], A[1], color='green', s=50, label="A (Start)")
    # plt.scatter(B[0], B[1], color='red', s=50, label="B (Target)")
    # for i, pt in enumerate(all_targets[1:-1], 1):
    #     plt.scatter(pt[0], pt[1], color='blue', s=30)
    #     plt.text(pt[0]+5, pt[1]+5, f"{i}", color='blue', fontsize=8)
    # plt.xlabel("X (m)")
    # plt.ylabel("Y (m)")
    # plt.title("Drone 2D XY Trajectory with Waypoints")
    # plt.xlim(0, 1000)
    # plt.ylim(0, 1000)
    # plt.legend()
    # plt.grid(True)
    # plt.show()
        
    # Animazione 3D con etichette, scatter per i target e log in basso
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation

    traj_arr = np.array(trajectory)
    x_data = traj_arr[:, 0]
    y_data = traj_arr[:, 1]
    z_data = traj_arr[:, 2]

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    line, = ax.plot([], [], [], 'k--', lw=1.5)
    point, = ax.plot([], [], [], 'ro', markersize=8)

    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_zlim(0, max(z_data)*1.1)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Drone Trajectory Animation')

    # Scatter marker per i waypoint
    ax.scatter(A[0], A[1], A[2], color='green', s=50, label='A')
    for i, pt in enumerate(all_targets[1:-1], 1):
        ax.scatter(pt[0], pt[1], pt[2], color='blue', s=30)
        ax.text(pt[0], pt[1], pt[2], f"{i}", color='blue', fontsize=8)
    ax.scatter(B[0], B[1], B[2], color='red', s=50, label='B')

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
                       f"Velocity: ({ld[10]}, {ld[11]}, {ld[12]})\n"
                       f"Pitch: {ld[4]} rad, Yaw: {ld[5]} rad")
            log_text.set_text(log_str)
        return line, point, log_text

    ani = animation.FuncAnimation(fig, update, frames=len(x_data),
                                  init_func=init, interval=0.03, blit=True)
    plt.show()
