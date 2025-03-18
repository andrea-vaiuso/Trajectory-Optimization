import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# ----- Helper functions -----
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

# ----- Your Classes -----
class Controller:
    def __init__(self, kp, ki, kd):
        """Initialize the PID parameters."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0
    
    def update(self, current_value, target_value, dt):
        """Compute the PID output based on the current error."""
        error = target_value - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class QuadCopterController:
    def __init__(self, state, 
                 kp_pos, ki_pos, kd_pos,  # PID for position (x, y, z)
                 kp_att, ki_att, kd_att,  # PID for attitude (roll, pitch, yaw)
                 m, g, b, dt=0.01):
        # Position PID controllers:
        self.pid_x = Controller(kp_pos, ki_pos, kd_pos)
        self.pid_y = Controller(kp_pos, ki_pos, kd_pos)
        self.pid_z = Controller(kp_pos, ki_pos, kd_pos)
        
        # Attitude PID controllers (for roll, pitch, yaw):
        self.pid_phi   = Controller(kp_att, ki_att, kd_att)   # roll desired
        self.pid_theta = Controller(kp_att, ki_att, kd_att)   # pitch desired
        self.pid_psi   = Controller(kp_att, ki_att, kd_att)   # yaw desired (computed dynamically)
        
        self.dt = dt
        self.state = state

        # For z feed-forward:
        self.m = m
        self.g = g
        self.b = b

    def update(self, state, target, dt):
        """
        Compute control commands (u1, u2, u3, u4):
          - u1: thrust
          - u2: roll control
          - u3: pitch control
          - u4: yaw control
        """
        x, y, z = state['pos']
        phi, theta, psi = state['angles']
        x_t, y_t, z_t = target['x'], target['y'], target['z']

        # Outer loop: position control (using feed-forward for hover)
        hover_thrust = self.m * self.g
        pid_z_output = self.pid_z.update(z, z_t, dt)
        u1 = hover_thrust + pid_z_output  # Total thrust

        max_angle = np.radians(20)  # maximum desired angle in radians
        # Use PID outputs for x and y but invert for roll
        theta_des = np.clip(self.pid_x.update(x, x_t, dt), -max_angle, max_angle)
        phi_des   = np.clip(-self.pid_y.update(y, y_t, dt), -max_angle, max_angle)
        
        # Compute desired yaw: point toward the target
        dx = target['x'] - x
        dy = target['y'] - y
        psi_des = np.arctan2(dy, dx)
        # Uncomment the following line to see the logs:
        # print(f"dx: {dx:.2f}, dy: {dy:.2f}, psi: {psi:.2f}, psi_des: {psi_des:.2f}, phi: {phi:.2f}, phi_des: {phi_des:.2f}, theta: {theta:.2f}, theta_des: {theta_des:.2f}")

        # Inner loop: attitude control
        u2 = self.pid_phi.update(phi, phi_des, dt)
        u3 = self.pid_theta.update(theta, theta_des, dt)
        u4 = self.pid_psi.update(psi, psi_des, dt)
        
        return (u1, u2, u3, u4)

class QuadcopterModel:
    def __init__(self,
                 m, I, b, d, l, Cd, Ca, Jr,
                 init_state,
                 controller,
                 g=9.81,
                 max_rpm=5000.0):
        """Initialize the quadcopter parameters."""
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
        """Compute the motor RPM required for hover."""
        T_hover = self.m * self.g
        w_hover = np.sqrt(T_hover / (4*self.b))
        rpm_hover = w_hover * 60.0 / (2.0*np.pi)
        print(f"[INFO] Hover thrust needed = {T_hover:.2f} N, hover rpm (each motor) ~ {rpm_hover:.1f} rpm")

    def __str__(self):
        return f"Quadcopter Model: state = {self.state}"

    def _translational_dynamics(self, state):
        """Compute translational dynamics."""
        omega = self._rpm_to_omega(state['rpm'])
        x_dot, y_dot, z_dot = state['vel']
        phi, theta, psi = state['angles']
        thrust = self.b * np.sum(np.square(omega))
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
        """Compute rotational dynamics."""
        omega = self._rpm_to_omega(state['rpm'])
        phi_dot, theta_dot, psi_dot = state['ang_vel']
        u2 = self.l * self.b * (omega[3]**2 - omega[1]**2)
        u3 = self.l * self.b * (omega[2]**2 - omega[0]**2)
        u4 = self.d * (omega[0]**2 - omega[1]**2 + omega[2]**2 - omega[3]**2)
        Omega_r = self.Jr * (omega[0] - omega[1] + omega[2] - omega[3])
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
        """Convert RPM to rad/s."""
        return rpm * 2 * np.pi / 60
    
    def _command_to_rpm(self, u1, u2, u3, u4):
        b = self.b
        d = self.d
        l = self.l
        w1_sq = (u1/(4*b)) - (u3/(2*b*l)) + (u4/(4*d))
        w2_sq = (u1/(4*b)) - (u2/(2*b*l)) - (u4/(4*d))
        w3_sq = (u1/(4*b)) + (u3/(2*b*l)) + (u4/(4*d))
        w4_sq = (u1/(4*b)) + (u2/(2*b*l)) - (u4/(4*d))
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

    def update_state(self, state, target, dt):
        u1, u2, u3, u4 = self.controller.update(state, target, dt)
        rpm1, rpm2, rpm3, rpm4 = self._command_to_rpm(u1, u2, u3, u4)
        state["rpm"] = np.array([rpm1, rpm2, rpm3, rpm4])
        acc = self._translational_dynamics(state)
        ang_acc = self._rotational_dynamics(state)
        state['pos']     += state['vel'] * dt
        state['vel']     += acc * dt
        state['angles']  += state['ang_vel'] * dt
        state['ang_vel'] += ang_acc * dt
        state['angles'] = np.array([wrap_angle(a) for a in state['angles']])
        return state

# ----- Simulation and Data Storage -----
dt = 0.01

# PID Gains (as provided)
kp_pos = 0.10
ki_pos = 0.0
kd_pos = 0.5

kp_att = 0.010
ki_att = 0.0
kd_att = 0.05

params = {
    'm': 5.2,       # kg
    'g': 9.81,      # m/s²
    'I': np.array([3.8e-3, 3.8e-3, 7.1e-3]),  # kg·m²
    'b': 3.13e-5,   # N·s²
    'd': 7.5e-7,    # N·m·s²
    'l': 0.32,      # m
    'Cd': np.array([0.1, 0.1, 0.15]),
    'Ca': np.array([0.1, 0.1, 0.15]),
    'Jr': 6e-5
}

# Initial state
state = {
    'pos': np.array([0.0, 0.0, 0.0]),
    'vel': np.array([0.0, 0.0, 0.0]),
    'angles': np.array([0.0, 0.0, 0.0]),
    'ang_vel': np.array([0.0, 0.0, 0.0]),
    'rpm': np.array([0.0, 0.0, 0.0, 0.0])
}

# Flight target
target = {
    'x': 20.0,
    'y': 20.0,
    'z': 20.0
}

quad_controller = QuadCopterController(
    state, 
    kp_pos, ki_pos, kd_pos,   # Position PID
    kp_att, ki_att, kd_att,   # Attitude PID
    m=params['m'], g=params['g'], b=params['b'], 
    dt=dt
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

simulation_time = 20.0
num_steps = int(simulation_time / dt)

positions = []
angles_history = []
rpms_history = []
time_history = []

for step in range(num_steps):
    state = drone.update_state(state, target, dt)
    positions.append(state['pos'].copy())
    angles_history.append(state['angles'].copy())
    rpms_history.append(state['rpm'].copy())
    time_history.append(step * dt)
    # Uncomment the next line to slow down simulation if needed:
    # time.sleep(dt)

positions = np.array(positions)
angles_history = np.array(angles_history)
rpms_history = np.array(rpms_history)
time_history = np.array(time_history)

# ----- Animation -----
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 50)
ax.set_ylim(0, 50)
ax.set_zlim(0, 50)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Quadcopter Animation')

def update_frame(frame):
    ax.cla()
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_zlim(0, 50)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Quadcopter Animation')
    
    # Plot trajectory up to the current frame
    ax.plot(positions[:frame, 0], positions[:frame, 1], positions[:frame, 2], 'b--')
    
    pos = positions[frame]
    ax.scatter(pos[0], pos[1], pos[2], color='red', s=50)
    
    # Compute body axes from Euler angles
    phi, theta, psi = angles_history[frame]
    R = euler_to_rot(phi, theta, psi)
    x_body = R @ np.array([1, 0, 0])
    y_body = R @ np.array([0, 1, 0])
    z_body = R @ np.array([0, 0, 1])
    arrow_len = 4
    ax.quiver(pos[0], pos[1], pos[2], arrow_len*x_body[0], arrow_len*x_body[1], arrow_len*x_body[2], color='r')
    ax.quiver(pos[0], pos[1], pos[2], arrow_len*y_body[0], arrow_len*y_body[1], arrow_len*y_body[2], color='g')
    ax.quiver(pos[0], pos[1], pos[2], arrow_len*z_body[0], arrow_len*z_body[1], arrow_len*z_body[2], color='b')
    
    # Display simulation time and current RPM in the bottom left corner
    current_time = frame * dt
    current_rpm = rpms_history[frame]
    text_str = f"Time: {current_time:.2f} s\nRPM: {current_rpm}"
    ax.text2D(0.05, 0.05, text_str, transform=ax.transAxes, fontsize=10,
              bbox=dict(facecolor='white', alpha=0.8))

# Set interval (in ms). Note: even reducing below 1 may not speed up if rendering is limited.
ani = animation.FuncAnimation(fig, update_frame, frames=len(positions), interval=1, repeat=True)
plt.show()

# ----- Post-Simulation: Plot x, y, z vs Time with Target Lines -----
fig2, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
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
fig2.suptitle('Drone Position vs Time')
plt.show()
