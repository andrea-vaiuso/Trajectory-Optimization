import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.pyplot as plt


# controllore PID

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def euler_to_rot(phi, theta, psi):
    """
    Convert Euler angles (roll=phi, pitch=theta, yaw=psi) to a rotation matrix.
    Assumes a rotation order: Rz(psi) * Ry(theta) * Rx(phi)
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
    def __init__(self, kp, ki, kd):
        """Initialize PID parameters."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, current_value, target_value, dt):
        """Compute the PID output."""
        error = target_value - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative



class QuadCopterController:
    def __init__(self, state, 
                 kp_pos, ki_pos, kd_pos,
                 kp_alt, ki_alt, kd_alt,  # PID per posizione (x, y, z)
                 kp_att, ki_att, kd_att,  # PID per assetto (roll, pitch, yaw)
                 m, g, b,
                 u1_limit=100.0, u2_limit=10.0, u3_limit=10.0, u4_limit=10.0,
                 max_angle_deg=30):
        
        self.u1_limit = u1_limit
        self.u2_limit = u2_limit
        self.u3_limit = u3_limit
        self.u4_limit = u4_limit

        # PID per la posizione (x, y, z)
        self.pid_x = Controller(kp_pos, ki_pos, kd_pos)
        self.pid_y = Controller(kp_pos, ki_pos, kd_pos)
        self.pid_z = Controller(kp_alt, ki_alt, kd_alt)
        
        # PID per l’assetto (roll=phi, pitch=theta, yaw=psi)
        self.pid_roll   = Controller(kp_att, ki_att, kd_att)
        self.pid_pitch = Controller(kp_att, ki_att, kd_att)
        self.pid_yaw   = Controller(kp_att, ki_att, kd_att)
        
        # Info di stato e tempo
        self.state = state

        # Per il feed-forward su z
        self.m = m
        self.g = g
        self.b = b
        self.max_angle = np.radians(max_angle_deg)

    def update(self, state, target, dt):
        """
        Calcola i comandi di controllo (u1, u2, u3, u4):
          - u1 ~ thrust
          - u2 ~ controllo roll
          - u3 ~ controllo pitch
          - u4 ~ controllo yaw
        """
        x,  y,  z  = state['pos']
        roll, pitch, yaw = state['angles']
        x_t, y_t, z_t = target['x'], target['y'], target['z']

        # Outer loop: posizione
        # Feed-forward di hover: m*g
        # + PID (che "raffina" intorno a mg in base all'errore su z)
        compensation = np.clip(1.0 / (np.cos(pitch) * np.cos(roll)), 1.0, 1.5)
        hover_thrust = self.m * self.g * compensation
        pid_z_output = self.pid_z.update(z, z_t, dt)
        thrust_command_u1 = hover_thrust + pid_z_output  # TOT thrust

        pitch_des = np.clip(self.pid_x.update(x, x_t, dt), -self.max_angle, self.max_angle)
        roll_des   = np.clip(-self.pid_y.update(y, y_t, dt), -self.max_angle, self.max_angle)
        
        # Calcolo yaw desiderato
        dx = (target['x'] - x)
        dy = (target['y'] - y)
        yaw_des = np.arctan2(dy, dx)
        
        # Inner loop: assetto
        roll_command_u2 = self.pid_roll.update(roll, roll_des, dt) # roll
        pitch_command_u3 = self.pid_pitch.update(pitch, pitch_des, dt) # pitch
        yaw_command_u4 = self.pid_yaw.update(yaw, yaw_des, dt) # yaw

        # Saturazione dei comandi
        thrust_command_u1 = np.clip(thrust_command_u1, 0, self.u1_limit)
        roll_command_u2 = np.clip(roll_command_u2, -self.u2_limit, self.u2_limit)
        pitch_command_u3 = np.clip(pitch_command_u3, -self.u3_limit, self.u3_limit)
        yaw_command_u4 = np.clip(yaw_command_u4, -self.u4_limit, self.u4_limit)
        
        return (thrust_command_u1, roll_command_u2, pitch_command_u3, yaw_command_u4)

# quadcopter model definition
class QuadcopterModel:
    def __init__(self,
                m,I,b,d,l,Cd,Ca,Jr,
                init_state,
                controller,
                g=9.81,
                max_rpm=5000.0,
                ):
        """Inizializza i parametri del quadricottero"""
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
        
        self.max_rpm_sq = (self.max_rpm * 2*np.pi/60)**2
        self._compute_hover_rpm()

    def _compute_hover_rpm(self):
        """Stima del regime di rotazione (uguale per tutti i motori) necessario per l'hover"""
        T_hover = self.m * self.g
        # Se i motori girano tutti alla stessa velocità w:
        # T_hover = 4 * b * w^2  =>  w = sqrt(T_hover / (4*b))
        w_hover = np.sqrt(T_hover / (4*self.b))
        rpm_hover = w_hover * 60.0 / (2.0*np.pi)
        print(f"[INFO] Hover thrust needed={T_hover:.2f} N, hover rpm each motor ~ {rpm_hover:.1f} rpm")

    def __str__(self):
        return f"Quadcopter Model: state={self.state}"

    def _translational_dynamics(self, state):
        # Convert RPM to angular velocity if needed
        omega = self._rpm_to_omega(state['rpm'])
        x_dot, y_dot, z_dot = state['vel']
        roll, pitch, yaw = state['angles']
        thrust = self.b * np.sum(np.square(omega))
        
        # Compute gravitational force
        g = self.g

        # --- Quadratic Drag Calculation ---
        # Compute the speed (magnitude of the velocity vector)
        v = np.linalg.norm(state['vel'])
        # Define air properties and reference area (adjust as needed)
        rho = 1.225    # air density in kg/m^3
        A = 0.1        # reference area in m^2 (estimate for your drone)
        C_d = 0.47     # drag coefficient (modify based on drone geometry)

        if v > 0:
            # Calculate the magnitude of the drag force
            drag_magnitude = 0.5 * rho * A * C_d * v**2
            # Drag force vector, opposing the velocity direction
            drag_vector = drag_magnitude * (state['vel'] / v)
        else:
            drag_vector = np.array([0.0, 0.0, 0.0])
        # --- End Quadratic Drag Calculation ---

        # Compute accelerations including the thrust contribution and subtracting drag
        x_ddot = (thrust / self.m *
                (np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll))
                - drag_vector[0] / self.m)
        y_ddot = (thrust / self.m *
                (np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll))
                - drag_vector[1] / self.m)
        z_ddot = (thrust / self.m *
                (np.cos(pitch)*np.cos(roll))
                - drag_vector[2] / self.m - g)

        return np.array([x_ddot, y_ddot, z_ddot])

    def _rotational_dynamics(self, state):
        """Calcola la dinamica rotazionale"""
        omega = self._rpm_to_omega(state['rpm'])
        phi_dot, theta_dot, psi_dot = state['ang_vel']
        
        # roll, pitch, yaw
        roll_torque = self.l * self.b * (omega[3]**2 - omega[1]**2)
        pitch_torque = self.l * self.b * (omega[2]**2 - omega[0]**2)
        yaw_torque = self.d * (omega[0]**2 - omega[1]**2 + omega[2]**2 - omega[3]**2)
        Omega_r = self.Jr * (omega[0] - omega[1] + omega[2] - omega[3])

        # Drag rotazionale + effetti giroscopici
        phi_ddot = (roll_torque / self.I[0]
                    - self.Ca[0] * np.sign(phi_dot) * phi_dot**2 / self.I[0]
                    - Omega_r/self.I[0]*theta_dot
                    - (self.I[2] - self.I[1])/self.I[0]*theta_dot*psi_dot)

        theta_ddot = (pitch_torque / self.I[1]
                      - self.Ca[1] * np.sign(theta_dot) * theta_dot**2 / self.I[1]
                      + Omega_r/self.I[1]*phi_dot
                      - (self.I[0] - self.I[2])/self.I[1]*phi_dot*psi_dot)

        psi_ddot = (yaw_torque / self.I[2]
                    - self.Ca[2] * np.sign(psi_dot) * psi_dot**2 / self.I[2]
                    - (self.I[1] - self.I[0])/self.I[2]*phi_dot*theta_dot)

        return np.array([phi_ddot, theta_ddot, psi_ddot])
    
    def _rpm_to_omega(self, rpm: np.ndarray):
        """Converte i giri al minuto in velocità angolare rad/s"""
        return rpm * 2 * np.pi / 60
    
    def _mixer(self, u1, u2, u3, u4):
        # 4) Calcolo motori: saturazione a 0 e a max_rpm
        b = self.b
        d = self.d
        l = self.l
        
        w1_sq = (u1/(4*b)) - (u3/(2*b*l)) + (u4/(4*d))
        w2_sq = (u1/(4*b)) - (u2/(2*b*l)) - (u4/(4*d))
        w3_sq = (u1/(4*b)) + (u3/(2*b*l)) + (u4/(4*d))
        w4_sq = (u1/(4*b)) + (u2/(2*b*l)) - (u4/(4*d))
        
        # Clamp a [0, max_rpm^2]
        w1_sq = np.clip(w1_sq, 0.0, self.max_rpm_sq)
        w2_sq = np.clip(w2_sq, 0.0, self.max_rpm_sq)
        w3_sq = np.clip(w3_sq, 0.0, self.max_rpm_sq)
        w4_sq = np.clip(w4_sq, 0.0, self.max_rpm_sq)
        
        w1 = np.sqrt(w1_sq)
        w2 = np.sqrt(w2_sq)
        w3 = np.sqrt(w3_sq)
        w4 = np.sqrt(w4_sq)
        
        # Converti in RPM
        rpm1 = w1 * 60.0 / (2.0 * np.pi)
        rpm2 = w2 * 60.0 / (2.0 * np.pi)
        rpm3 = w3 * 60.0 / (2.0 * np.pi)
        rpm4 = w4 * 60.0 / (2.0 * np.pi)

        return rpm1, rpm2, rpm3, rpm4

    def _rk4_step(self, state, dt):
        # Funzione ausiliaria che ritorna le derivate (d/dt) dello stato
        def f(s):
            return {
                'pos': s['vel'],
                'vel': self._translational_dynamics(s),
                'angles': s['ang_vel'],
                'ang_vel': self._rotational_dynamics(s)
            }
        
        # Calcolo k1
        k1 = f(state)
        
        # Calcolo k2: stato a metà passo
        state1 = {key: state[key] + k1[key]*(dt/2) for key in ['pos', 'vel', 'angles', 'ang_vel']}
        state1['rpm'] = state['rpm']  # rpm rimangono costanti durante l'integrazione
        k2 = f(state1)
        
        # Calcolo k3: altro stato a metà passo usando k2
        state2 = {key: state[key] + k2[key]*(dt/2) for key in ['pos', 'vel', 'angles', 'ang_vel']}
        state2['rpm'] = state['rpm']
        k3 = f(state2)
        
        # Calcolo k4: stato al passo completo usando k3
        state3 = {key: state[key] + k3[key]*dt for key in ['pos', 'vel', 'angles', 'ang_vel']}
        state3['rpm'] = state['rpm']
        k4 = f(state3)
        
        # Combina i k per ottenere il nuovo stato
        new_state = {}
        for key in ['pos', 'vel', 'angles', 'ang_vel']:
            new_state[key] = state[key] + (dt/6) * (k1[key] + 2*k2[key] + 2*k3[key] + k4[key])
        new_state['rpm'] = state['rpm']
        
        return new_state

    def update_state(self, state, target, dt):
        # 1) Ottieni i comandi di controllo dal controllore
        u1, u2, u3, u4 = self.controller.update(state, target, dt)
        rpm1, rpm2, rpm3, rpm4 = self._mixer(u1, u2, u3, u4)
        state["rpm"] = np.array([rpm1, rpm2, rpm3, rpm4])
        
        # 2) Integra lo stato usando il metodo RK4
        state = self._rk4_step(state, dt)
        
        # 3) Normalizza gli angoli nell'intervallo [-pi, pi]
        state['angles'] = np.array([wrap_angle(a) for a in state['angles']])
        
        return state


def main():
    # --- Simulation Parameters ---
    dt = 0.007
    simulation_time = 200.0
    num_steps = int(simulation_time / dt)
    frame_skip = 6

    # Flight target
    target = {'x': 100.0, 'y': 100.0, 'z': 100.0}

    # PID gains (position, altitude, attitude)
    kp_pos, ki_pos, kd_pos = 0.1, 1e-6, 0.5
    kp_alt, ki_alt, kd_alt = 0.05, 1e-6, 0.7
    kp_att, ki_att, kd_att = 0.01, 1e-6, 0.05

    # Drone physical parameters
    params = {
        'm': 5.2,              # kg
        'g': 9.81,             # m/s²
        'I': np.array([3.8e-3, 3.8e-3, 7.1e-3]),  # kg·m²
        'b': 3.13e-5,          # N·s²
        'd': 7.5e-7,           # N·m·s²
        'l': 0.32,             # m
        'Cd': np.array([0.1, 0.1, 0.15]),
        'Ca': np.array([0.1, 0.1, 0.15]),
        'Jr': 6e-5
    }

    # Initial state of the drone
    state = {
        'pos': np.array([0.0, 0.0, 0.0]),
        'vel': np.array([0.0, 0.0, 0.0]),
        'angles': np.array([0.0, 0.0, 0.0]),   # [Pitch, Roll, Yaw]
        'ang_vel': np.array([0.0, 0.0, 0.0]),
        'rpm': np.array([0.0, 0.0, 0.0, 0.0])
    }

    # --- Initialize Controller and Model ---
    quad_controller = QuadCopterController(
        state, 
        kp_pos, ki_pos, kd_pos,     # Position PID
        kp_alt, ki_alt, kd_alt,     # Altitude PID
        kp_att, ki_att, kd_att,     # Attitude PID
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

    for step in range(num_steps):
        state = drone.update_state(state, target, dt)
        if step % frame_skip == 0:
            positions.append(state['pos'].copy())
            angles_history.append(state['angles'].copy())
            rpms_history.append(state['rpm'].copy())
            time_history.append(step * dt)
            horiz_speed_history.append(np.linalg.norm(state['vel'][:2]))
            vertical_speed_history.append(state['vel'][2])

    # Convert lists to numpy arrays for plotting
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

    # Pre-create trajectory line, drone scatter, and status text
    trajectory_line, = ax_anim.plot([], [], [], 'b--', lw=2, label='Trajectory')
    drone_scatter = ax_anim.scatter([], [], [], color='red', s=50, label='Drone')
    time_text = ax_anim.text2D(0.05, 0.05, "", transform=ax_anim.transAxes, fontsize=10,
                               bbox=dict(facecolor='white', alpha=0.8))
    current_quivers = []

    # Helper: Convert Euler angles to rotation matrix
    def euler_to_rot(phi, theta, psi):
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

    def init_anim():
        trajectory_line.set_data([], [])
        trajectory_line.set_3d_properties([])
        drone_scatter._offsets3d = ([], [], [])
        time_text.set_text("")
        return trajectory_line, drone_scatter, time_text

    def update_anim(frame):
        nonlocal current_quivers
        # Update trajectory
        xdata = positions[:frame, 0]
        ydata = positions[:frame, 1]
        zdata = positions[:frame, 2]
        trajectory_line.set_data(xdata, ydata)
        trajectory_line.set_3d_properties(zdata)

        # Update drone position
        pos = positions[frame]
        drone_scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

        # Remove previous quivers if any
        for q in current_quivers:
            q.remove()
        current_quivers = []

        # Add quiver arrows for body axes
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

        # Update status text
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

    # --- Post-Simulation Plots (6 Subplots in 2 Columns x 3 Rows) ---
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    
    # First Column: X, Y, Z Positions
    # X Position
    axs[0, 0].plot(time_history, positions[:, 0], label='X Position')
    axs[0, 0].axhline(target['x'], color='r', linestyle='--', label='Target X')
    axs[0, 0].set_title('X Position')
    axs[0, 0].set_ylabel('X (m)')
    axs[0, 0].legend()
    
    # Y Position
    axs[1, 0].plot(time_history, positions[:, 1], label='Y Position')
    axs[1, 0].axhline(target['y'], color='r', linestyle='--', label='Target Y')
    axs[1, 0].set_title('Y Position')
    axs[1, 0].set_ylabel('Y (m)')
    axs[1, 0].legend()
    
    # Z Position
    axs[2, 0].plot(time_history, positions[:, 2], label='Z Position')
    axs[2, 0].axhline(target['z'], color='r', linestyle='--', label='Target Z')
    axs[2, 0].set_title('Z Position')
    axs[2, 0].set_ylabel('Z (m)')
    axs[2, 0].legend()
    axs[2, 0].set_xlabel('Time (s)')
    
    # Second Column: Attitude, Motor RPMs, Speeds
    # Attitude (Pitch, Roll, Yaw)
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
    
    # Horizontal & Vertical Speeds
    axs[2, 1].plot(time_history, horiz_speed_history, label='Horizontal Speed')
    axs[2, 1].plot(time_history, vertical_speed_history, label='Vertical Speed')
    axs[2, 1].set_title('Speeds')
    axs[2, 1].set_ylabel('Speed (m/s)')
    axs[2, 1].legend()
    axs[2, 1].set_xlabel('Time (s)')
    
    # Optionally, set the x-label for the top two plots in the second column if desired:
    axs[0, 1].set_xlabel('Time (s)')
    axs[1, 1].set_xlabel('Time (s)')
    
    fig.suptitle('Drone Simulation Data vs Time', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
