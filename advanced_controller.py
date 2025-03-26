import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# --- Utility Functions and PID Controller ---

def wrap_angle(angle: float) -> float:
    """
    Wrap an angle in the range [-pi, pi].

    Parameters:
        angle (float): Angle in radians.

    Returns:
        float: Wrapped angle in radians.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def euler_to_rot(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Convert Euler angles (roll=phi, pitch=theta, yaw=psi) into a rotation matrix.
    Assumes the rotation order Rz(psi) * Ry(theta) * Rx(phi).

    Parameters:
        phi (float): Roll angle in radians.
        theta (float): Pitch angle in radians.
        psi (float): Yaw angle in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix.
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
        Initialize the PID controller.

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
            current_value (float): The current measurement.
            target_value (float): The desired setpoint.
            dt (float): Time step.

        Returns:
            float: Control output.
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
            state (dict): Current state of the drone.
            kp_pos, ki_pos, kd_pos (float): PID gains for position.
            kp_alt, ki_alt, kd_alt (float): PID gains for altitude.
            kp_att, ki_att, kd_att (float): PID gains for attitude (roll & pitch).
            kp_yaw, ki_yaw, kd_yaw (float): PID gains for yaw.
            m (float): Mass of the drone.
            g (float): Gravitational acceleration.
            b (float): Thrust coefficient.
            u1_limit, u2_limit, u3_limit, u4_limit (float): Saturation limits for the control commands.
            max_angle_deg (float): Maximum tilt angle in degrees.
        """
        self.u1_limit = u1_limit
        self.u2_limit = u2_limit
        self.u3_limit = u3_limit
        self.u4_limit = u4_limit

        # PID for position (x, y, z)
        self.pid_x = Controller(kp_pos, ki_pos, kd_pos)
        self.pid_y = Controller(kp_pos, ki_pos, kd_pos)
        self.pid_z = Controller(kp_alt, ki_alt, kd_alt)
        
        # PID for attitude (roll and pitch) and a separate PID for yaw
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
        Compute the control commands for the quadcopter.

        Parameters:
            state (dict): Current state of the drone.
            target (dict): Target position with keys 'x', 'y', and 'z'.
            dt (float): Time step.

        Returns:
            tuple: (thrust_command, roll_command, pitch_command, yaw_command)
        """
        x, y, z = state['pos']
        roll, pitch, yaw = state['angles']
        x_t, y_t, z_t = target['x'], target['y'], target['z']

        # Outer loop: position control with feed-forward for hover
        compensation = np.clip(1.0 / (np.cos(pitch) * np.cos(roll)), 1.0, 1.5)
        hover_thrust = self.m * self.g * compensation
        pid_z_output = self.pid_z.update(z, z_t, dt)
        thrust_command = hover_thrust + pid_z_output

        pitch_des = np.clip(self.pid_x.update(x, x_t, dt), -self.max_angle, self.max_angle)
        roll_des  = np.clip(-self.pid_y.update(y, y_t, dt), -self.max_angle, self.max_angle)
        
        # Compute desired yaw based on target position
        dx = target['x'] - x
        dy = target['y'] - y
        yaw_des = np.arctan2(dy, dx)
        
        # Inner loop: attitude control
        roll_command = self.pid_roll.update(roll, roll_des, dt)
        pitch_command = self.pid_pitch.update(pitch, pitch_des, dt)
        yaw_command = self.pid_yaw.update(yaw, 0, dt)  # alternatively, use yaw_des

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
        Initialize the physical model of the quadcopter.

        Parameters:
            m (float): Mass.
            I (np.ndarray): Moment of inertia vector.
            b (float): Thrust coefficient.
            d (float): Drag coefficient.
            l (float): Arm length.
            Cd (np.ndarray): Translational drag coefficients.
            Ca (np.ndarray): Rotational damping coefficients.
            Jr (float): Rotor inertia.
            init_state (dict): Initial state.
            controller (QuadCopterController): Controller instance.
            g (float): Gravitational acceleration.
            max_rpm (float): Maximum RPM.
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
        Compute the RPM value needed for hovering flight.
        """
        T_hover = self.m * self.g
        w_hover = np.sqrt(T_hover / (4 * self.b))
        rpm_hover = w_hover * 60.0 / (2.0 * np.pi)
        # Uncomment the following line for debug information:
        # print(f"[INFO] Hover thrust needed = {T_hover:.2f} N, hover rpm per motor ~ {rpm_hover:.1f} rpm")

    def __str__(self) -> str:
        """
        Return a string representation of the quadcopter model.
        """
        return f"Quadcopter Model: state = {self.state}"

    def _translational_dynamics(self, state: dict) -> np.ndarray:
        """
        Compute the translational accelerations.

        Parameters:
            state (dict): Current state.

        Returns:
            np.ndarray: Acceleration vector [x_ddot, y_ddot, z_ddot].
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
            state (dict): Current state.

        Returns:
            np.ndarray: Angular acceleration vector [phi_ddot, theta_ddot, psi_ddot].
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
        Convert motor RPM to angular velocity (rad/s).

        Parameters:
            rpm (np.ndarray): Array of motor RPMs.

        Returns:
            np.ndarray: Angular velocities in rad/s.
        """
        return rpm * 2 * np.pi / 60
    
    def _mixer(self, u1: float, u2: float, u3: float, u4: float) -> tuple:
        """
        Compute the RPM for each motor based on the control inputs.

        Parameters:
            u1, u2, u3, u4 (float): Control inputs.

        Returns:
            tuple: RPM values for each motor.
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
            state (dict): Current state.
            dt (float): Time step.

        Returns:
            dict: New state after the integration step.
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
        Update the drone's state by computing control commands, mixing motor RPMs,
        and integrating the dynamics.

        Parameters:
            state (dict): Current state.
            target (dict): Target position with keys 'x', 'y', and 'z'.
            dt (float): Time step.

        Returns:
            dict: Updated state.
        """
        u1, u2, u3, u4 = self.controller.update(state, target, dt)
        rpm1, rpm2, rpm3, rpm4 = self._mixer(u1, u2, u3, u4)
        state["rpm"] = np.array([rpm1, rpm2, rpm3, rpm4])
        state = self._rk4_step(state, dt)
        state['angles'] = np.array([wrap_angle(a) for a in state['angles']])
        return state

# --- Look-Ahead Target Function ---

def compute_moving_target(drone_pos, seg_start, seg_end, v_des, k=1.0):
    """
    Compute the dynamic target point along a segment with a look-ahead distance of L = k*v_des.

    Parameters:
        drone_pos (np.array): Current drone position [x, y, z].
        seg_start (np.array): Start point of the segment.
        seg_end (np.array): End point of the segment.
        v_des (float): Desired speed for this segment.
        k (float): Scaling factor for the look-ahead distance.

    Returns:
        tuple: (target, progress) where target is the dynamic target point [x, y, z],
               and progress is the fraction of the segment covered.
    """
    seg_vector = seg_end - seg_start
    seg_length = np.linalg.norm(seg_vector)
    if seg_length == 0:
        return seg_end, 1.0
    seg_dir = seg_vector / seg_length

    # Project current position onto the segment
    proj_length = np.dot(drone_pos - seg_start, seg_dir)
    # Look-ahead distance
    L = k * v_des
    target_length = proj_length + L

    # Do not exceed the final waypoint
    if target_length > seg_length:
        target_length = seg_length
    target = seg_start + target_length * seg_dir
    progress = target_length / seg_length
    return target, progress

# --- Main: Simulation and Plotting ---

def main():
    """
    Run the drone simulation and plot the results.
    The simulation stops early if the drone reaches the final target (within a 2-meter threshold).
    """
    # Simulation parameters
    dt = 0.007
    simulation_time = 200.0
    num_steps = int(simulation_time / dt)
    frame_skip = 8
    threshold = 2.0  # Stop simulation if within 2 meters of final target
    dynamic_target_shift_threshold_prc = 0.7 # Shift to next segment if a certain percentage of current segment is covered

    # --- Define Waypoints (with desired speed) ---
    waypoints = [
        {'x': 10.0, 'y': 10.0, 'z': 70.0, 'v': 10},  # Start near origin at high altitude
        {'x': 90.0, 'y': 10.0, 'z': 70.0, 'v': 10},  # Far in x, near y, maintaining high altitude
        {'x': 90.0, 'y': 90.0, 'z': 90.0, 'v': 0.5},   # Far in both x and y with even higher altitude
        {'x': 10.0, 'y': 90.0, 'z': 20.0, 'v': 10},   # Sharp maneuver: near x, far y with dramatic altitude drop
        {'x': 50.0, 'y': 50.0, 'z': 40.0, 'v': 10},   # Central target with intermediate altitude
        {'x': 60.0, 'y': 60.0, 'z': 40.0, 'v': 10},   # Hovering target 1
        {'x': 70.0, 'y': 70.0, 'z': 40.0, 'v': 10},   # Hovering target 2
        {'x': 80.0, 'y': 80.0, 'z': 40.0, 'v': 10},   # Hovering target 3
        {'x': 10.0, 'y': 10.0, 'z': 10.0, 'v': 10}    # Final target: near origin at low altitude
    ]

    # Initial drone state
    state = {
        'pos': np.array([0.0, 0.0, 0.0]),
        'vel': np.array([0.0, 0.0, 0.0]),
        'angles': np.array([0.0, 0.0, 0.0]),
        'ang_vel': np.array([0.0, 0.0, 0.0]),
        'rpm': np.array([0.0, 0.0, 0.0, 0.0])
    }
    start_position = state['pos'].copy()

    # PID controller settings (yaw gains remain fixed)
    kp_yaw, ki_yaw, kd_yaw = 0.5, 1e-6, 0.1
    kp_pos, ki_pos, kd_pos = 0.080397, 6.6749e-07, 0.18084
    kp_alt, ki_alt, kd_alt = 6.4593, 0.00042035, 10.365
    kp_att, ki_att, kd_att = 2.7805, 0.00045168, 0.36006

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

    # Initialize the quadcopter controller and model
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

    # --- Initialize Dynamic Target Strategy ---
    current_seg_idx = 0
    seg_start = state['pos'].copy()
    seg_end = np.array([waypoints[current_seg_idx]['x'], 
                        waypoints[current_seg_idx]['y'], 
                        waypoints[current_seg_idx]['z']])
    v_des = waypoints[current_seg_idx]['v']
    k_lookahead = 1.0  # Scaling parameter for look-ahead distance

    # Lists for storing data for animation and plotting
    positions = []
    angles_history = []
    rpms_history = []
    time_history = []
    horiz_speed_history = []
    vertical_speed_history = []
    targets = []  # List to store the dynamic target

    # Simulation loop
    for step in range(num_steps):
        # Compute dynamic target along the current segment
        target_dynamic, progress = compute_moving_target(state['pos'], seg_start, seg_end, v_des, k=k_lookahead)
        
        # If progress on current segment is nearly complete, move to the next segment if available
        if progress >= dynamic_target_shift_threshold_prc:
            current_seg_idx += 1
            if current_seg_idx < len(waypoints):
                seg_start = seg_end
                seg_end = np.array([waypoints[current_seg_idx]['x'], 
                                    waypoints[current_seg_idx]['y'], 
                                    waypoints[current_seg_idx]['z']])
                v_des = waypoints[current_seg_idx]['v']
                target_dynamic, progress = compute_moving_target(state['pos'], seg_start, seg_end, v_des, k=k_lookahead)
            else:
                target_dynamic = seg_end  # Final waypoint: fixed target
        
        # Update the drone state using the dynamic target
        state = drone.update_state(state, {'x': target_dynamic[0], 'y': target_dynamic[1], 'z': target_dynamic[2]}, dt)
        current_time = step * dt

        # Save data every 'frame_skip' steps
        if step % frame_skip == 0:
            positions.append(state['pos'].copy())
            angles_history.append(state['angles'].copy())
            rpms_history.append(state['rpm'].copy())
            time_history.append(current_time)
            horiz_speed_history.append(np.linalg.norm(state['vel'][:2]))
            vertical_speed_history.append(state['vel'][2])
            targets.append(target_dynamic.copy())

        # Check if drone reached the final target (within threshold)

        final_target = np.array([waypoints[-1]['x'], waypoints[-1]['y'], waypoints[-1]['z']])
        if np.linalg.norm(state['pos'] - final_target) < threshold:
            print(f"Final target reached at time: {current_time:.2f} s")
            break

    positions = np.array(positions)
    angles_history = np.array(angles_history)
    rpms_history = np.array(rpms_history)
    time_history = np.array(time_history)
    horiz_speed_history = np.array(horiz_speed_history)
    vertical_speed_history = np.array(vertical_speed_history)
    targets = np.array(targets)

    # --- Animation: 3D Trajectory of the Drone ---
    fig_anim = plt.figure(figsize=(10, 8))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    ax_anim.set_xlim(0, 100)
    ax_anim.set_ylim(0, 100)
    ax_anim.set_zlim(0, 100)
    ax_anim.set_xlabel('X')
    ax_anim.set_ylabel('Y')
    ax_anim.set_zlabel('Z')
    ax_anim.set_title('Quadcopter Animation')

    trajectory_line, = ax_anim.plot([], [], [], 'b--', lw=2, label='Trajectory')
    drone_scatter = ax_anim.scatter([], [], [], color='red', s=50, label='Drone')
    # Add a scatter for the dynamic target
    target_scatter = ax_anim.scatter([], [], [], marker='*', color='magenta', s=100, label='Target')
    
    time_text = ax_anim.text2D(0.05, 0.05, "", transform=ax_anim.transAxes, fontsize=10,
                               bbox=dict(facecolor='white', alpha=0.8))
    
    # Display the starting point
    ax_anim.scatter(start_position[0], start_position[1], start_position[2],
                    marker='o', color='green', s=100, label='Start')
    # Display waypoints
    for i, wp in enumerate(waypoints, start=1):
        ax_anim.scatter(wp['x'], wp['y'], wp['z'], marker='X', color='purple', s=100,
                        label=f'Waypoint {i}' if i == 1 else None)
        ax_anim.text(wp['x'], wp['y'], wp['z'] + 2, f'{i}', color='black', fontsize=12, ha='center')

    def init_anim():
        trajectory_line.set_data([], [])
        trajectory_line.set_3d_properties([])
        drone_scatter._offsets3d = ([], [], [])
        target_scatter._offsets3d = ([], [], [])
        time_text.set_text("")
        return trajectory_line, drone_scatter, target_scatter, time_text

    def update_anim(frame):
        nonlocal targets
        xdata = positions[:frame, 0]
        ydata = positions[:frame, 1]
        zdata = positions[:frame, 2]
        trajectory_line.set_data(xdata, ydata)
        trajectory_line.set_3d_properties(zdata)

        pos = positions[frame]
        drone_scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
        
        # Update the dynamic target marker
        targ = targets[frame]
        target_scatter._offsets3d = ([targ[0]], [targ[1]], [targ[2]])

        # Update the arrows indicating the drone's attitude
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
                    f"RPM: [{current_rpm[0]:.2f}, {current_rpm[1]:.2f}, {current_rpm[2]:.2f}, {current_rpm[3]:.2f}]\n"
                    f"Vertical Speed: {vertical_speed_history[frame]:.2f} m/s\n"
                    f"Horizontal Speed: {horiz_speed_history[frame]:.2f} m/s\n"
                    f"Pitch: {angles_history[frame][0]:.4f} rad\n"
                    f"Roll: {angles_history[frame][1]:.4f} rad\n"
                    f"Yaw: {angles_history[frame][2]:.4f} rad")
        time_text.set_text(text_str)

        return trajectory_line, drone_scatter, target_scatter, time_text, *current_quivers

    # List to manage attitude arrow objects
    current_quivers = []

    ani = animation.FuncAnimation(fig_anim, update_anim, frames=len(positions),
                                  init_func=init_anim, interval=50, blit=False, repeat=True)
    plt.show()

    # --- Post-Simulation Plots (Positions, Attitude, RPM, Speeds) ---
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    
    # X Position
    axs[0, 0].plot(time_history, positions[:, 0], label='X Position')
    for wp in waypoints:
        axs[0, 0].axhline(y=wp['x'], linestyle='--', color='r', label='Waypoint X' if wp == waypoints[0] else None)
    axs[0, 0].set_title('X Position')
    axs[0, 0].set_ylabel('X (m)')
    axs[0, 0].legend()
    
    # Y Position
    axs[1, 0].plot(time_history, positions[:, 1], label='Y Position')
    for wp in waypoints:
        axs[1, 0].axhline(y=wp['y'], linestyle='--', color='r', label='Waypoint Y' if wp == waypoints[0] else None)
    axs[1, 0].set_title('Y Position')
    axs[1, 0].set_ylabel('Y (m)')
    axs[1, 0].legend()
    
    # Z Position
    axs[2, 0].plot(time_history, positions[:, 2], label='Z Position')
    for wp in waypoints:
        axs[2, 0].axhline(y=wp['z'], linestyle='--', color='r', label='Waypoint Z' if wp == waypoints[0] else None)
    axs[2, 0].set_title('Z Position')
    axs[2, 0].set_ylabel('Z (m)')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].legend()
    
    # Attitude: Pitch, Roll, Yaw
    axs[0, 1].plot(time_history, angles_history[:, 0], label='Pitch')
    axs[0, 1].plot(time_history, angles_history[:, 1], label='Roll')
    axs[0, 1].plot(time_history, angles_history[:, 2], label='Yaw')
    axs[0, 1].set_title('Attitude (Pitch, Roll, Yaw)')
    axs[0, 1].set_ylabel('Angle (rad)')
    axs[0, 1].legend()
    
    # Motor RPMs
    axs[1, 1].plot(time_history, rpms_history[:, 0], label='RPM1')
    axs[1, 1].plot(time_history, rpms_history[:, 1], label='RPM2')
    axs[1, 1].plot(time_history, rpms_history[:, 2], label='RPM3')
    axs[1, 1].plot(time_history, rpms_history[:, 3], label='RPM4')
    axs[1, 1].set_title('Motor RPMs')
    axs[1, 1].set_ylabel('RPM')
    axs[1, 1].legend()
    
    # Speeds: Horizontal and Vertical
    axs[2, 1].plot(time_history, horiz_speed_history, label='Horizontal Speed', color='r')
    axs[2, 1].plot(time_history, vertical_speed_history, label='Vertical Speed', color='g')
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
