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
        """Compute the PID output using RK4 to integrate the error."""
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
                 u1_limit=100.0, u2_limit=10.0, u3_limit=10.0, u4_limit=10.0):
        
        self.u1_limit = u1_limit
        self.u2_limit = u2_limit
        self.u3_limit = u3_limit
        self.u4_limit = u4_limit

        # PID per la posizione (x, y, z)
        self.pid_x = Controller(kp_pos, ki_pos, kd_pos)
        self.pid_y = Controller(kp_pos, ki_pos, kd_pos)
        self.pid_z = Controller(kp_alt, ki_alt, kd_alt)
        
        # PID per l’assetto (roll=phi, pitch=theta, yaw=psi)
        self.pid_phi   = Controller(kp_att, ki_att, kd_att)  # roll desiderato
        self.pid_theta = Controller(kp_att, ki_att, kd_att)  # pitch desiderato
        self.pid_psi   = Controller(kp_att, ki_att, kd_att)  # yaw desiderato (verrà impostato dinamicamente)
        
        # Info di stato e tempo
        self.state = state

        # Per il feed-forward su z
        self.m = m
        self.g = g
        self.b = b

    def update(self, state, target, dt):
        """
        Calcola i comandi di controllo (u1, u2, u3, u4):
          - u1 ~ thrust
          - u2 ~ controllo roll
          - u3 ~ controllo pitch
          - u4 ~ controllo yaw
        """
        x,  y,  z  = state['pos']
        phi, theta, psi = state['angles']
        x_t, y_t, z_t = target['x'], target['y'], target['z']
        

        # Outer loop: posizione
        # Feed-forward di hover: m*g
        # + PID (che "raffina" intorno a mg in base all'errore su z)
        compensation = np.clip(1.0 / (np.cos(theta) * np.cos(phi)), 1.0, 1.5)
        hover_thrust = self.m * self.g * compensation
        pid_z_output = self.pid_z.update(z, z_t, dt)
        u1 = hover_thrust + pid_z_output  # TOT thrust

        max_angle = np.radians(20)  # 20 gradi
        theta_des = np.clip(self.pid_x.update(x, x_t, dt), -max_angle, max_angle)
        phi_des   = np.clip(-self.pid_y.update(y, y_t, dt), -max_angle, max_angle)
        
        # Calcolo yaw desiderato
        dx = (target['x'] - x)
        dy = (target['y'] - y)
        psi_des = np.arctan2(dy, dx)
        # print(f"dx: {dx:.2f}, dy: {dy:.2f}, psi: {psi:.2f}, psi_des: {psi_des:.2f}, phi: {phi:.2f}, phi_des:{phi_des:.2f}, theta: {theta:.2f}, theta_des: {theta_des:.2f}")
        # Inner loop: assetto
        u2 = self.pid_phi.update(phi, phi_des, dt)
        u3 = self.pid_theta.update(theta, theta_des, dt)
        u4 = self.pid_psi.update(psi, psi_des, dt)

        # Saturazione dei comandi
        u1 = np.clip(u1, 0, self.u1_limit)
        u2 = np.clip(u2, -self.u2_limit, self.u2_limit)
        u3 = np.clip(u3, -self.u3_limit, self.u3_limit)
        u4 = np.clip(u4, -self.u4_limit, self.u4_limit)
        
        return (u1, u2, u3, u4)

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
        """Calcola la dinamica traslazionale"""
        omega = self._rpm_to_omega(state['rpm'])
        x_dot, y_dot, z_dot = state['vel']
        phi, theta, psi = state['angles']
        thrust = self.b * np.sum(np.square(omega))
        
        # Con drag semplice proporzionale a x_dot
        x_ddot = (thrust / self.m *
                  (np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi))
                  - self.Cd[0]*x_dot/self.m)
        y_ddot = (thrust / self.m *
                  (np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi))
                  - self.Cd[1]*y_dot/self.m)
        z_ddot = (thrust / self.m *
                  (np.cos(theta)*np.cos(phi)) - self.Cd[2]*z_dot/self.m - self.g)

        return np.array([x_ddot, y_ddot, z_ddot])

    def _rotational_dynamics(self, state):
        """Calcola la dinamica rotazionale"""
        omega = self._rpm_to_omega(state['rpm'])
        phi_dot, theta_dot, psi_dot = state['ang_vel']
        
        # roll, pitch, yaw
        u2 = self.l * self.b * (omega[3]**2 - omega[1]**2)
        u3 = self.l * self.b * (omega[2]**2 - omega[0]**2)
        u4 = self.d * (omega[0]**2 - omega[1]**2 + omega[2]**2 - omega[3]**2)
        Omega_r = self.Jr * (omega[0] - omega[1] + omega[2] - omega[3])

        # Drag rotazionale + effetti giroscopici
        phi_ddot = (u2 / self.I[0]
                    - self.Ca[0] * np.sign(phi_dot) * phi_dot**2 / self.I[0]
                    - Omega_r/self.I[0]*theta_dot
                    - (self.I[2] - self.I[1])/self.I[0]*theta_dot*psi_dot)

        theta_ddot = (u3 / self.I[1]
                      - self.Ca[1] * np.sign(theta_dot) * theta_dot**2 / self.I[1]
                      + Omega_r/self.I[1]*phi_dot
                      - (self.I[0] - self.I[2])/self.I[1]*phi_dot*psi_dot)

        psi_ddot = (u4 / self.I[2]
                    - self.Ca[2] * np.sign(psi_dot) * psi_dot**2 / self.I[2]
                    - (self.I[1] - self.I[0])/self.I[2]*phi_dot*theta_dot)

        return np.array([phi_ddot, theta_ddot, psi_ddot])
    
    def _rpm_to_omega(self, rpm: np.ndarray):
        """Converte i giri al minuto in velocità angolare rad/s"""
        return rpm * 2 * np.pi / 60
    
    def _command_to_rpm(self, u1, u2, u3, u4):
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
        rpm1, rpm2, rpm3, rpm4 = self._command_to_rpm(u1, u2, u3, u4)
        state["rpm"] = np.array([rpm1, rpm2, rpm3, rpm4])
        
        # 2) Integra lo stato usando il metodo RK4
        state = self._rk4_step(state, dt)
        
        # 3) Normalizza gli angoli nell'intervallo [-pi, pi]
        state['angles'] = np.array([wrap_angle(a) for a in state['angles']])
        
        return state



if __name__ == "__main__":
    dt = 0.01
    simulation_time = 200.0

    # Flight target
    target = {
        'x': 10.0,
        'y': 10.0,
        'z': 100.0
    }

    #kp_pos, ki_pos, kd_pos = 0.1, 0.0, 0.5
    #kp_alt, ki_alt, kd_alt = 0.05, 0.0, 0.7
    #kp_att, ki_att, kd_att = 0.01, 0.0, 0.05

    # reactivity, sability, overshoot
    kp_pos, ki_pos, kd_pos = 0.1018, 1e-6, 1.2
    kp_alt, ki_alt, kd_alt = 0.0587, 1e-6, 0.7341
    kp_att, ki_att, kd_att = 0.0500, 1e-6, 0.0797

    params = {
        'm': 5.2,  # kg
        'g': 9.81,  # m/s^2
        'I': np.array([3.8e-3, 3.8e-3, 7.1e-3]),  # kg·m²
        'b': 3.13e-5,  # N·s²
        'd': 7.5e-7,   # N·m·s²
        'l': 0.32,     # m
        'Cd': np.array([0.1, 0.1, 0.15]),
        'Ca': np.array([0.1, 0.1, 0.15]),
        'Jr': 6e-5
    }

    # Initial state
    state = {
        'pos': np.array([0.0, 0.0, 0.0]),
        'vel': np.array([0.0, 0.0, 0.0]),
        'angles': np.array([0.0, 0.0, 0.0]),  # [phi, theta, psi]
        'ang_vel': np.array([0.0, 0.0, 0.0]),
        'rpm': np.array([0.0, 0.0, 0.0, 0.0])
    }

    

    # Initialize controller and model
    quad_controller = QuadCopterController(
        state, 
        kp_pos, ki_pos, kd_pos,    # Position PID
        kp_alt, ki_alt, kd_alt,    # Altitude PID
        kp_att, ki_att, kd_att,    # Attitude PID
        m=params['m'], g=params['g'], b=params['b'],
        u1_limit=100.0, u2_limit=10.0, u3_limit=5.0, u4_limit=10.0
    )
    """
        - u1 ~ thrust
        - u2 ~ controllo roll
        - u3 ~ controllo pitch
        - u4 ~ controllo yaw"
    """
    
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


    
    num_steps = int(simulation_time / dt)

    positions = []
    angles_history = []
    rpms_history = []
    time_history = []
    horiz_speed_history = []
    vertical_speed_history = []

    frame_skip = 6
    for step in range(num_steps):
        state = drone.update_state(state, target, dt)
        if step % frame_skip == 0:
            positions.append(state['pos'].copy())
            angles_history.append(state['angles'].copy())
            rpms_history.append(state['rpm'].copy())
            time_history.append(step * dt)
            horiz_speed_history.append(np.linalg.norm(state['vel'][:2]))
            vertical_speed_history.append(state['vel'][2])
            # print(f"Time: {step*dt:.2f}s, Pos: {state['pos']}, Ang: {state['angles']}")
            # You can comment out time.sleep(dt) to run simulation faster
            # time.sleep(dt)

    positions = np.array(positions)
    angles_history = np.array(angles_history)
    rpms_history = np.array(rpms_history)
    time_history = np.array(time_history)
    horiz_speed_history = np.array(horiz_speed_history)
    vertical_speed_history = np.array(vertical_speed_history)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set fixed plot limits (0 to 50 for x, y, z)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Quadcopter Animation')

    # Pre-create the trajectory line and drone scatter
    trajectory_line, = ax.plot([], [], [], 'b--', lw=2, label='Trajectory')
    drone_scatter = ax.scatter([], [], [], color='red', s=50, label='Drone')
    # Pre-create text annotation for time and RPM
    time_text = ax.text2D(0.05, 0.05, "", transform=ax.transAxes, fontsize=10,
                            bbox=dict(facecolor='white', alpha=0.8))

    # (Optional) For the body axes, you can choose to not display quivers to speed up rendering.
    # If you do want quivers, a strategy is to remove and add them each frame.
    current_quivers = []

    # Helper function: Euler angles to rotation matrix (as defined in your code)
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

    def init():
        trajectory_line.set_data([], [])
        trajectory_line.set_3d_properties([])
        drone_scatter._offsets3d = ([], [], [])
        time_text.set_text("")
        return trajectory_line, drone_scatter, time_text

    def update_frame(frame):
        global current_quivers
        # Update trajectory line without clearing axes
        xdata = positions[:frame, 0]
        ydata = positions[:frame, 1]
        zdata = positions[:frame, 2]
        trajectory_line.set_data(xdata, ydata)
        trajectory_line.set_3d_properties(zdata)
        
        # Update the drone scatter position
        pos = positions[frame]
        drone_scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
        
        # Remove previous quivers for body axes if any
        for q in current_quivers:
            q.remove()
        current_quivers = []
        
        # Optionally, add quiver arrows for body axes
        # Compute rotation matrix from Euler angles
        phi, theta, psi = angles_history[frame]
        R = euler_to_rot(phi, theta, psi)
        arrow_len = 4
        x_body = R @ np.array([1, 0, 0])
        y_body = R @ np.array([0, 1, 0])
        z_body = R @ np.array([0, 0, 1])
        
        qx = ax.quiver(pos[0], pos[1], pos[2], arrow_len*x_body[0], arrow_len*x_body[1], arrow_len*x_body[2], color='r')
        qy = ax.quiver(pos[0], pos[1], pos[2], arrow_len*y_body[0], arrow_len*y_body[1], arrow_len*y_body[2], color='g')
        qz = ax.quiver(pos[0], pos[1], pos[2], arrow_len*z_body[0], arrow_len*z_body[1], arrow_len*z_body[2], color='b')
        current_quivers.extend([qx, qy, qz])
        
        # Update text annotation with current time and RPM
        current_time = frame * dt * frame_skip
        current_rpm = rpms_history[frame]
        text_str = f"Time: {current_time:.2f} s\n " +\
            f"RPM: [{current_rpm[0]:.2f}, {current_rpm[1]:.2f}, {current_rpm[2]:.2f}, {current_rpm[3]:.2f}] "+\
            f"\nVertical Speed: {vertical_speed_history[frame]:.2f} m/s\nHoriz Speed: {horiz_speed_history[frame]:.2f} m/s"+\
            f"\nPitch: {angles_history[frame][0]:.2f} rad\nRoll: {angles_history[frame][1]:.2f} rad\nYaw: {angles_history[frame][2]:.2f} rad"
        time_text.set_text(text_str)
        
        return trajectory_line, drone_scatter, time_text, *current_quivers

    # Create animation. Using a short interval; note that full-screen mode and 3D rendering
    # in Matplotlib are inherently heavy, so performance is still limited.
    ani = animation.FuncAnimation(fig, update_frame, frames=len(positions),
                                init_func=init, interval=50, blit=False, repeat=True)
    plt.show()

        # ----- Post-Simulation: Plot x, y, z vs Time with Target Lines -----
    fig2, axs = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
    axs[0].plot(time_history, positions[:, 0], label='x position')
    axs[0].axhline(target['x'], color='r', linestyle='--', label='Target x')
    axs[0].set_ylabel('X (m)')
    axs[0].legend()
    axs[1].plot(time_history, positions[:, 1], label='y position')
    axs[1].axhline(target['y'], color='r', linestyle='--', label='Target y')
    axs[1].set_ylabel('Y (m)')
    axs[1].legend()
    axs[2].plot(time_history, positions[:, 2], label='z position')
    axs[2].axhline(target['z'], color='r', linestyle='--', label='Target z')
    axs[2].set_ylabel('Z (m)')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend()
    axs[3].plot(time_history, horiz_speed_history, label='speed')
    axs[3].set_ylabel('Horiz Speed (m/s)')
    axs[3].set_xlabel('Time (s)')
    axs[3].legend()
    axs[4].plot(time_history, vertical_speed_history, label='speed')
    axs[4].set_ylabel('Vertical Speed (m/s)')
    axs[4].set_xlabel('Time (s)')
    axs[4].legend()
    fig2.suptitle('Drone Position vs Time')
    plt.show()