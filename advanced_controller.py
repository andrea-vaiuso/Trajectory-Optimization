import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# --- PID Controller and Helper Functions ---

def wrap_angle(angle: float) -> float:
    """
    Wrap an angle to the range [-pi, pi].

    Parameters:
        angle (float): Input angle in radians.

    Returns:
        float: Wrapped angle in radians.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def euler_to_rot(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Convert Euler angles (roll=phi, pitch=theta, yaw=psi) to a rotation matrix.
    Assumes a rotation order: Rz(psi) * Ry(theta) * Rx(phi).

    Parameters:
        phi (float): Roll angle in radians.
        theta (float): Pitch angle in radians.
        psi (float): Yaw angle in radians.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi),  np.cos(psi), 0],
                   [0, 0, 1]])
    Ry = np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi),  np.cos(phi)]])
    return Rz @ Ry @ Rx

class Controller:
    def __init__(self, kp: float, ki: float, kd: float):
        """
        Initialize PID controller parameters.

        Parameters:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, current_value: float, target_value: float, dt: float) -> float:
        """
        Compute the PID controller output.

        Parameters:
            current_value (float): The current value.
            target_value (float): The target value.
            dt (float): Time step in seconds.

        Returns:
            float: PID controller output.
        """
        error = target_value - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class QuadCopterController:
    def __init__(self, state: dict,
                 kp_pos: float, ki_pos: float, kd_pos: float,
                 kp_alt: float, ki_alt: float, kd_alt: float,
                 kp_att: float, ki_att: float, kd_att: float,
                 kp_yaw: float, ki_yaw: float, kd_yaw: float,
                 m: float, g: float, b: float,
                 u1_limit: float = 100.0, u2_limit: float = 10.0, 
                 u3_limit: float = 10.0, u4_limit: float = 10.0,
                 max_angle_deg: float = 30):
        """
        Initialize the quadcopter controller with PID controllers for position, altitude, attitude, and yaw.

        Parameters:
            state (dict): Initial state of the quadcopter.
            kp_pos, ki_pos, kd_pos (float): PID gains for x and y position.
            kp_alt, ki_alt, kd_alt (float): PID gains for altitude (z).
            kp_att, ki_att, kd_att (float): PID gains for roll & pitch.
            kp_yaw, ki_yaw, kd_yaw (float): PID gains for yaw.
            m (float): Mass of the drone.
            g (float): Gravitational acceleration.
            b (float): Motor thrust coefficient.
            u1_limit, u2_limit, u3_limit, u4_limit (float): Saturation limits for control commands.
            max_angle_deg (float): Maximum tilt angle in degrees.
        """
        self.u1_limit = u1_limit
        self.u2_limit = u2_limit
        self.u3_limit = u3_limit
        self.u4_limit = u4_limit

        # PID controllers for position (x, y, z)
        self.pid_x = Controller(kp_pos, ki_pos, kd_pos)
        self.pid_y = Controller(kp_pos, ki_pos, kd_pos)
        self.pid_z = Controller(kp_alt, ki_alt, kd_alt)
        
        # PID controllers for attitude (roll & pitch) and separate PID for yaw
        self.pid_roll  = Controller(kp_att, ki_att, kd_att)
        self.pid_pitch = Controller(kp_att, ki_att, kd_att)
        self.pid_yaw   = Controller(kp_yaw, ki_yaw, kd_yaw)
        
        self.state = state
        self.m = m
        self.g = g
        self.b = b
        self.max_angle = np.radians(max_angle_deg)

    def update(self, state: dict, target: dict, dt: float) -> tuple:
        """
        Compute control commands for the quadcopter.

        Parameters:
            state (dict): Current state of the drone with keys 'pos' and 'angles'.
            target (dict): Target position with keys 'x', 'y', 'z'.
            dt (float): Time step in seconds.

        Returns:
            tuple: (thrust_command, roll_command, pitch_command, yaw_command)
        """
        x, y, z = state['pos']
        roll, pitch, yaw = state['angles']
        x_t, y_t, z_t = target['x'], target['y'], target['z']

        # Outer loop: position control with hover feed-forward
        compensation = np.clip(1.0 / (np.cos(pitch) * np.cos(roll)), 1.0, 1.5)
        hover_thrust = self.m * self.g * compensation
        pid_z_output = self.pid_z.update(z, z_t, dt)
        thrust_command = hover_thrust + pid_z_output

        pitch_des = np.clip(self.pid_x.update(x, x_t, dt), -self.max_angle, self.max_angle)
        roll_des  = np.clip(-self.pid_y.update(y, y_t, dt), -self.max_angle, self.max_angle)
        
        # Compute desired yaw from target position
        dx = target['x'] - x
        dy = target['y'] - y
        yaw_des = np.arctan2(dy, dx)
        
        # Inner loop: attitude control
        roll_command = self.pid_roll.update(roll, roll_des, dt)
        pitch_command = self.pid_pitch.update(pitch, pitch_des, dt)
        # For yaw, you may choose to track the yaw_des value; here we use 0 as reference
        yaw_command = self.pid_yaw.update(yaw, 0, dt)

        # Saturate the commands
        thrust_command = np.clip(thrust_command, 0, self.u1_limit)
        roll_command = np.clip(roll_command, -self.u2_limit, self.u2_limit)
        pitch_command = np.clip(pitch_command, -self.u3_limit, self.u3_limit)
        yaw_command = np.clip(yaw_command, -self.u4_limit, self.u4_limit)
        
        return (thrust_command, roll_command, pitch_command, yaw_command)

class QuadcopterModel:
    def __init__(self, m: float, I: np.ndarray, b: float, d: float, l: float, 
                 Cd: np.ndarray, Ca: np.ndarray, Jr: float,
                 init_state: dict, controller: QuadCopterController,
                 g: float = 9.81, max_rpm: float = 5000.0):
        """
        Initialize the quadcopter model with physical parameters.

        Parameters:
            m (float): Mass of the drone.
            I (np.ndarray): Moment of inertia (array of three values).
            b (float): Motor thrust coefficient.
            d (float): Drag torque coefficient.
            l (float): Arm length of the drone.
            Cd (np.ndarray): Drag coefficients for x, y, z.
            Ca (np.ndarray): Rotational damping coefficients for roll, pitch, yaw.
            Jr (float): Rotor inertia.
            init_state (dict): Initial state of the drone.
            controller (QuadCopterController): Controller instance for the drone.
            g (float): Gravitational acceleration.
            max_rpm (float): Maximum motor RPM.
        """
        self.m = m
        self.I = I
        self.b = b
        self.d = d
        self.l = l
        self.Cd = Cd
        self.Ca = Ca
        self.Jr = Jr
        self.g = g
        self.state = init_state
        self.controller = controller
        self.max_rpm = max_rpm
        
        self.max_rpm_sq = (self.max_rpm * 2 * np.pi / 60)**2
        self._compute_hover_rpm()

    def _compute_hover_rpm(self) -> None:
        """
        Estimate the motor RPM required for hovering and print the result.
        """
        T_hover = self.m * self.g
        w_hover = np.sqrt(T_hover / (4 * self.b))
        rpm_hover = w_hover * 60.0 / (2.0 * np.pi)
        print(f"[INFO] Hover thrust needed = {T_hover:.2f} N, hover rpm per motor ~ {rpm_hover:.1f} rpm")

    def __str__(self) -> str:
        """
        Return a string representation of the quadcopter state.

        Returns:
            str: String describing the current state.
        """
        return f"Quadcopter Model: state = {self.state}"

    def _translational_dynamics(self, state: dict) -> np.ndarray:
        """
        Compute the translational accelerations.

        Parameters:
            state (dict): Current state of the drone.

        Returns:
            np.ndarray: Accelerations [x_ddot, y_ddot, z_ddot].
        """
        omega = self._rpm_to_omega(state['rpm'])
        x_dot, y_dot, z_dot = state['vel']
        roll, pitch, yaw = state['angles']
        thrust = self.b * np.sum(np.square(omega))
        
        v = np.linalg.norm(state['vel'])
        rho = 1.225  # Air density in kg/m³
        A = 0.1      # Reference area in m²
        C_d = 0.47   # Drag coefficient

        if v > 0:
            drag_magnitude = 0.5 * rho * A * C_d * v**2
            drag_vector = drag_magnitude * (state['vel'] / v)
        else:
            drag_vector = np.array([0.0, 0.0, 0.0])

        x_ddot = (thrust / self.m *
                  (np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll))
                  - drag_vector[0] / self.m)
        y_ddot = (thrust / self.m *
                  (np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll))
                  - drag_vector[1] / self.m)
        z_ddot = (thrust / self.m *
                  (np.cos(pitch) * np.cos(roll))
                  - drag_vector[2] / self.m - self.g)

        return np.array([x_ddot, y_ddot, z_ddot])

    def _rotational_dynamics(self, state: dict) -> np.ndarray:
        """
        Compute the rotational accelerations.

        Parameters:
            state (dict): Current state of the drone.

        Returns:
            np.ndarray: Angular accelerations [phi_ddot, theta_ddot, psi_ddot].
        """
        omega = self._rpm_to_omega(state['rpm'])
        phi_dot, theta_dot, psi_dot = state['ang_vel']
        
        roll_torque = self.l * self.b * (omega[3]**2 - omega[1]**2)
        pitch_torque = self.l * self.b * (omega[2]**2 - omega[0]**2)
        yaw_torque = self.d * (omega[0]**2 - omega[1]**2 + omega[2]**2 - omega[3]**2)
        Omega_r = self.Jr * (omega[0] - omega[1] + omega[2] - omega[3])

        phi_ddot = (roll_torque / self.I[0]
                    - self.Ca[0] * np.sign(phi_dot) * phi_dot**2 / self.I[0]
                    - Omega_r / self.I[0] * theta_dot
                    - (self.I[2] - self.I[1]) / self.I[0] * theta_dot * psi_dot)
        theta_ddot = (pitch_torque / self.I[1]
                      - self.Ca[1] * np.sign(theta_dot) * theta_dot**2 / self.I[1]
                      + Omega_r / self.I[1] * phi_dot
                      - (self.I[0] - self.I[2]) / self.I[1] * phi_dot * psi_dot)
        psi_ddot = (yaw_torque / self.I[2]
                    - self.Ca[2] * np.sign(psi_dot) * psi_dot**2 / self.I[2]
                    - (self.I[1] - self.I[0]) / self.I[2] * phi_dot * theta_dot)

        return np.array([phi_ddot, theta_ddot, psi_ddot])
    
    def _rpm_to_omega(self, rpm: np.ndarray) -> np.ndarray:
        """
        Convert motor RPM values to angular velocities (rad/s).

        Parameters:
            rpm (np.ndarray): Array of motor RPM values.

        Returns:
            np.ndarray: Array of angular velocities in rad/s.
        """
        return rpm * 2 * np.pi / 60
    
    def _mixer(self, u1: float, u2: float, u3: float, u4: float) -> tuple:
        """
        Compute individual motor RPM commands based on control inputs.

        Parameters:
            u1 (float): Thrust command.
            u2 (float): Roll command.
            u3 (float): Pitch command.
            u4 (float): Yaw command.

        Returns:
            tuple: (rpm1, rpm2, rpm3, rpm4) motor commands.
        """
        b = self.b
        d = self.d
        l = self.l
        
        w1_sq = (u1 / (4 * b)) - (u3 / (2 * b * l)) + (u4 / (4 * d))
        w2_sq = (u1 / (4 * b)) - (u2 / (2 * b * l)) - (u4 / (4 * d))
        w3_sq = (u1 / (4 * b)) + (u3 / (2 * b * l)) + (u4 / (4 * d))
        w4_sq = (u1 / (4 * b)) + (u2 / (2 * b * l)) - (u4 / (4 * d))
        
        w1_sq = np.clip(w1_sq, 0.0, self.max_rpm_sq)
        w2_sq = np.clip(w2_sq, 0.0, self.max_rpm_sq)
        w3_sq = np.clip(w3_sq, 0.0, self.max_rpm_sq)
        w4_sq = np.clip(w4_sq, 0.0, self.max_rpm_sq)
        
        w1 = np.sqrt(w1_sq)
        w2 = np.sqrt(w2_sq)
        w3 = np.sqrt(w3_sq)
        w4 = np.sqrt(w4_sq)
        
        rpm1 = w1 * 60.0 / (2.0 * np.pi)
        rpm2 = w2 * 60.0 / (2.0 * np.pi)
        rpm3 = w3 * 60.0 / (2.0 * np.pi)
        rpm4 = w4 * 60.0 / (2.0 * np.pi)

        return rpm1, rpm2, rpm3, rpm4

    def _rk4_step(self, state: dict, dt: float) -> dict:
        """
        Perform one Runge-Kutta 4th order integration step.

        Parameters:
            state (dict): Current state of the drone.
            dt (float): Time step in seconds.

        Returns:
            dict: New state after dt.
        """
        def f(s: dict) -> dict:
            return {
                'pos': s['vel'],
                'vel': self._translational_dynamics(s),
                'angles': s['ang_vel'],
                'ang_vel': self._rotational_dynamics(s)
            }
        
        k1 = f(state)
        state1 = {key: state[key] + k1[key] * (dt / 2) for key in ['pos', 'vel', 'angles', 'ang_vel']}
        state1['rpm'] = state['rpm']
        k2 = f(state1)
        
        state2 = {key: state[key] + k2[key] * (dt / 2) for key in ['pos', 'vel', 'angles', 'ang_vel']}
        state2['rpm'] = state['rpm']
        k3 = f(state2)
        
        state3 = {key: state[key] + k3[key] * dt for key in ['pos', 'vel', 'angles', 'ang_vel']}
        state3['rpm'] = state['rpm']
        k4 = f(state3)
        
        new_state = {}
        for key in ['pos', 'vel', 'angles', 'ang_vel']:
            new_state[key] = state[key] + (dt / 6) * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key])
        new_state['rpm'] = state['rpm']
        
        return new_state

    def update_state(self, state: dict, target: dict, dt: float) -> dict:
        """
        Update the drone state by computing control commands, mixing motor RPMs, and integrating the dynamics.

        Parameters:
            state (dict): Current state of the drone.
            target (dict): Current target position.
            dt (float): Time step in seconds.

        Returns:
            dict: Updated state.
        """
        u1, u2, u3, u4 = self.controller.update(state, target, dt)
        rpm1, rpm2, rpm3, rpm4 = self._mixer(u1, u2, u3, u4)
        state["rpm"] = np.array([rpm1, rpm2, rpm3, rpm4])
        state = self._rk4_step(state, dt)
        state['angles'] = np.array([wrap_angle(a) for a in state['angles']])
        return state

# --- Main Simulation and Plotting ---

def main():
    # Simulation parameters
    dt: float = 0.007
    simulation_time: float = 200.0
    num_steps: int = int(simulation_time / dt)
    frame_skip: int = 6

    # --- List of Flight Targets ---
    # Define multiple targets; for example, first target: (100, 100, 40), second: (50, 85, 60),
    # third: (80, 20, 10), and fourth: (0, 0, 0).
    targets = [
        {'x': 100.0, 'y': 100.0, 'z': 40.0},
        {'x': 50.0, 'y': 85.0, 'z': 60.0},
        {'x': 80.0, 'y': 20.0, 'z': 10.0},
        {'x': 0.0, 'y': 0.0, 'z': 0.0}
    ]
    current_target_idx: int = 0
    target = targets[current_target_idx]

    # PID gains for various controllers
    kp_pos, ki_pos, kd_pos = 0.1, 1e-6, 0.5
    kp_alt, ki_alt, kd_alt = 0.2, 1e-5, 1.3
    kp_att, ki_att, kd_att = 0.05, 1e-6, 0.05
    kp_yaw, ki_yaw, kd_yaw = 0.5, 1e-6, 0.1

    # Drone physical parameters
    params = {
        'm': 5.2,
        'g': 9.81,
        'I': np.array([3.8e-3, 3.8e-3, 7.1e-3]),
        'b': 3.13e-5,
        'd': 7.5e-7,
        'l': 0.32,
        'Cd': np.array([0.1, 0.1, 0.15]),
        'Ca': np.array([0.1, 0.1, 0.15]),
        'Jr': 6e-5
    }

    # Initial state of the drone
    state = {
        'pos': np.array([0.0, 0.0, 0.0]),
        'vel': np.array([0.0, 0.0, 0.0]),
        'angles': np.array([0.0, 0.0, 0.0]),
        'ang_vel': np.array([0.0, 0.0, 0.0]),
        'rpm': np.array([0.0, 0.0, 0.0, 0.0])
    }
    start_position = state['pos'].copy()

    # Initialize Controller and Model
    quad_controller = QuadCopterController(
        state, 
        kp_pos, ki_pos, kd_pos,     
        kp_alt, ki_alt, kd_alt,     
        kp_att, ki_att, kd_att,
        kp_yaw, ki_yaw, kd_yaw,   
        m=params['m'], g=params['g'], b=params['b'],
        u1_limit=100.0, u2_limit=10.0, u3_limit=5.0, u4_limit=10.0
    )

    drone = QuadcopterModel(
        m=params['m'],
        I=params['I'],
        b=params['b'],
        d=params['d'],
        l=params['l'],
        Cd=params['Cd'],
        Ca=params['Ca'],
        Jr=params['Jr'],
        init_state=state,
        controller=quad_controller,
        max_rpm=10000.0
    )

    # --- Simulation Loop ---
    positions = []
    angles_history = []
    rpms_history = []
    time_history = []
    horiz_speed_history = []
    vertical_speed_history = []
    target_reach_times = []  # Record simulation times when a target is reached

    for step in range(num_steps):
        state = drone.update_state(state, target, dt)
        current_time = step * dt

        # Check if the drone has reached the current target within a 3 m error threshold
        pos_error = np.linalg.norm(state['pos'] - np.array([target['x'], target['y'], target['z']]))
        if pos_error < 3.0:
            target_reach_times.append(current_time)
            if current_target_idx < len(targets) - 1:
                current_target_idx += 1
                target = targets[current_target_idx]
                print(f"Target reached at t={current_time:.2f}s. Switching to target: {target}")
            else:
                print(f"Final target reached at t={current_time:.2f}s within error threshold. Ending simulation early.")
                break

        if step % frame_skip == 0:
            positions.append(state['pos'].copy())
            angles_history.append(state['angles'].copy())
            rpms_history.append(state['rpm'].copy())
            time_history.append(current_time)
            horiz_speed_history.append(np.linalg.norm(state['vel'][:2]))
            vertical_speed_history.append(state['vel'][2])

    positions = np.array(positions)
    angles_history = np.array(angles_history)
    rpms_history = np.array(rpms_history)
    time_history = np.array(time_history)
    horiz_speed_history = np.array(horiz_speed_history)
    vertical_speed_history = np.array(vertical_speed_history)

    # --- Animation Plot (3D Drone Trajectory & Status) ---
    fig_anim = plt.figure(figsize=(10, 8))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    ax_anim.set_xlim(0, 100)
    ax_anim.set_ylim(0, 100)
    ax_anim.set_zlim(0, 100)
    ax_anim.set_xlabel('X')
    ax_anim.set_ylabel('Y')
    ax_anim.set_zlabel('Z')
    ax_anim.set_title('Quadcopter Animation')

    # Plot the trajectory line and current drone position
    trajectory_line, = ax_anim.plot([], [], [], 'b--', lw=2, label='Trajectory')
    drone_scatter = ax_anim.scatter([], [], [], color='red', s=50, label='Drone')
    time_text = ax_anim.text2D(0.05, 0.05, "", transform=ax_anim.transAxes, fontsize=10,
                               bbox=dict(facecolor='white', alpha=0.8))
    current_quivers = []

    # Mark starting point and target points
    ax_anim.scatter(start_position[0], start_position[1], start_position[2],
                    marker='o', color='green', s=100, label='Start')
    for i, tgt in enumerate(targets, start=1):
        ax_anim.scatter(tgt['x'], tgt['y'], tgt['z'], marker='X', color='purple', s=100,
                        label=f'Target {i}' if i == 1 else None)
        ax_anim.text(tgt['x'], tgt['y'], tgt['z'] + 2, f'{i}', color='black', fontsize=12, ha='center')

    def init_anim():
        trajectory_line.set_data([], [])
        trajectory_line.set_3d_properties([])
        drone_scatter._offsets3d = ([], [], [])
        time_text.set_text("")
        return trajectory_line, drone_scatter, time_text

    def update_anim(frame):
        nonlocal current_quivers
        xdata = positions[:frame, 0]
        ydata = positions[:frame, 1]
        zdata = positions[:frame, 2]
        trajectory_line.set_data(xdata, ydata)
        trajectory_line.set_3d_properties(zdata)

        pos = positions[frame]
        drone_scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

        # Remove previous quivers
        for q in current_quivers:
            q.remove()
        current_quivers.clear()

        phi, theta, psi = angles_history[frame]
        R = euler_to_rot(phi, theta, psi)
        arrow_len = 4
        x_body = R @ np.array([1, 0, 0])
        y_body = R @ np.array([0, 1, 0])
        z_body = R @ np.array([0, 0, 1])
        qx = ax_anim.quiver(pos[0], pos[1], pos[2],
                            arrow_len * x_body[0], arrow_len * x_body[1], arrow_len * x_body[2],
                            color='r')
        qy = ax_anim.quiver(pos[0], pos[1], pos[2],
                            arrow_len * y_body[0], arrow_len * y_body[1], arrow_len * y_body[2],
                            color='g')
        qz = ax_anim.quiver(pos[0], pos[1], pos[2],
                            arrow_len * z_body[0], arrow_len * z_body[1], arrow_len * z_body[2],
                            color='b')
        current_quivers.extend([qx, qy, qz])

        current_time = frame * dt * frame_skip
        current_rpm = rpms_history[frame]
        text_str = (f"Time: {current_time:.2f} s\n"
                    f"RPM: [{current_rpm[0]:.2f}, {current_rpm[1]:.2f}, "
                    f"{current_rpm[2]:.2f}, {current_rpm[3]:.2f}]\n"
                    f"Vertical Speed: {vertical_speed_history[frame]:.2f} m/s\n"
                    f"Horiz Speed: {horiz_speed_history[frame]:.2f} m/s\n"
                    f"Pitch: {angles_history[frame][0]:.4f} rad\n"
                    f"Roll: {angles_history[frame][1]:.4f} rad\n"
                    f"Yaw: {angles_history[frame][2]:.4f} rad")
        time_text.set_text(text_str)

        return trajectory_line, drone_scatter, time_text, *current_quivers

    ani = animation.FuncAnimation(fig_anim, update_anim, frames=len(positions),
                                  init_func=init_anim, interval=50, blit=False, repeat=True)
    plt.show()  # Close animation window to continue

    # --- Post-Simulation Plots (Positions, Attitude, RPMs, Speeds) ---

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    
    # Build time segments based on target reach times.
    # seg_times: list of times [0, t1, t2, ..., t_end]
    t_end = time_history[-1]
    seg_times = [0.0] + target_reach_times[:]
    if seg_times[-1] < t_end:
        seg_times.append(t_end)
    segments = []
    # Each segment is active with the target that was current during that segment.
    for i in range(len(seg_times) - 1):
        # Use target index i if available; if i exceeds available targets, use the last target.
        tgt_idx = i if i < len(targets) else len(targets) - 1
        segments.append((seg_times[i], seg_times[i+1], targets[tgt_idx]))
    
    # X Position Plot
    axs[0, 0].plot(time_history, positions[:, 0], label='X Position')
    for i, seg in enumerate(segments):
        axs[0, 0].hlines(y=seg[2]['x'], xmin=seg[0], xmax=seg[1], colors='r', 
                         linestyles='--', label='Target X' if i == 0 else None)
    for t in target_reach_times:
        axs[0, 0].axvline(x=t, color='k', linestyle='--')
    axs[0, 0].set_title('X Position')
    axs[0, 0].set_ylabel('X (m)')
    axs[0, 0].legend()
    
    # Y Position Plot
    axs[1, 0].plot(time_history, positions[:, 1], label='Y Position')
    for i, seg in enumerate(segments):
        axs[1, 0].hlines(y=seg[2]['y'], xmin=seg[0], xmax=seg[1], colors='r', 
                         linestyles='--', label='Target Y' if i == 0 else None)
    for t in target_reach_times:
        axs[1, 0].axvline(x=t, color='k', linestyle='--')
    axs[1, 0].set_title('Y Position')
    axs[1, 0].set_ylabel('Y (m)')
    axs[1, 0].legend()
    
    # Z Position Plot
    axs[2, 0].plot(time_history, positions[:, 2], label='Z Position')
    for i, seg in enumerate(segments):
        axs[2, 0].hlines(y=seg[2]['z'], xmin=seg[0], xmax=seg[1], colors='r', 
                         linestyles='--', label='Target Z' if i == 0 else None)
    for t in target_reach_times:
        axs[2, 0].axvline(x=t, color='k', linestyle='--')
    axs[2, 0].set_title('Z Position')
    axs[2, 0].set_ylabel('Z (m)')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].legend()
    
    # Attitude Plot (Pitch, Roll, Yaw)
    axs[0, 1].plot(time_history, angles_history[:, 0], label='Pitch')
    axs[0, 1].plot(time_history, angles_history[:, 1], label='Roll')
    axs[0, 1].plot(time_history, angles_history[:, 2], label='Yaw')
    axs[0, 1].set_title('Attitude (Pitch, Roll, Yaw)')
    axs[0, 1].set_ylabel('Angle (rad)')
    axs[0, 1].legend()
    
    # Motor RPMs Plot
    axs[1, 1].plot(time_history, rpms_history[:, 0], label='RPM1')
    axs[1, 1].plot(time_history, rpms_history[:, 1], label='RPM2')
    axs[1, 1].plot(time_history, rpms_history[:, 2], label='RPM3')
    axs[1, 1].plot(time_history, rpms_history[:, 3], label='RPM4')
    axs[1, 1].set_title('Motor RPMs')
    axs[1, 1].set_ylabel('RPM')
    axs[1, 1].legend()
    
    # Speeds Plot (Horizontal & Vertical)
    axs[2, 1].plot(time_history, horiz_speed_history, label='Horizontal Speed')
    axs[2, 1].plot(time_history, vertical_speed_history, label='Vertical Speed')
    axs[2, 1].set_title('Speeds')
    axs[2, 1].set_ylabel('Speed (m/s)')
    axs[2, 1].set_xlabel('Time (s)')
    axs[2, 1].legend()
    
    axs[0, 1].set_xlabel('Time (s)')
    axs[1, 1].set_xlabel('Time (s)')
    
    fig.suptitle('Drone Simulation Data vs Time', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()
