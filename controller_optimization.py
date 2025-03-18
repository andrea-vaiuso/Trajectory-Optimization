import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence

# --- Helper Functions ---
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def euler_to_rot(phi, theta, psi):
    """Convert Euler angles (roll=phi, pitch=theta, yaw=psi) to a rotation matrix."""
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

# --- Classes (same as before) ---
class Controller:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0
    
    def update(self, current_value, target_value, dt):
        error = target_value - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class QuadCopterController:
    def __init__(self, state, 
                 kp_pos, ki_pos, kd_pos,   # lateral (x,y) position PID
                 kp_alt, ki_alt, kd_alt,   # altitude (z) PID
                 kp_att, ki_att, kd_att,   # attitude PID
                 m, g, b, dt=0.01):
        self.pid_x = Controller(kp_pos, ki_pos, kd_pos)
        self.pid_y = Controller(kp_pos, ki_pos, kd_pos)
        self.pid_z = Controller(kp_alt, ki_alt, kd_alt)
        self.pid_phi   = Controller(kp_att, ki_att, kd_att)
        self.pid_theta = Controller(kp_att, ki_att, kd_att)
        self.pid_psi   = Controller(kp_att, ki_att, kd_att)
        self.dt = dt
        self.state = state
        self.m = m
        self.g = g
        self.b = b

    def update(self, state, target, dt):
        x, y, z = state['pos']
        phi, theta, psi = state['angles']
        x_t, y_t, z_t = target['x'], target['y'], target['z']
        hover_thrust = self.m * self.g
        pid_z_output = self.pid_z.update(z, z_t, dt)
        u1 = hover_thrust + pid_z_output

        max_angle = np.radians(20)
        theta_des = np.clip(self.pid_x.update(x, x_t, dt), -max_angle, max_angle)
        phi_des   = np.clip(-self.pid_y.update(y, y_t, dt), -max_angle, max_angle)

        dx = target['x'] - x
        dy = target['y'] - y
        psi_des = np.arctan2(dy, dx)
        u2 = self.pid_phi.update(phi, phi_des, dt)
        u3 = self.pid_theta.update(theta, theta_des, dt)
        u4 = self.pid_psi.update(psi, psi_des, dt)
        return (u1, u2, u3, u4)

class QuadcopterModel:
    def __init__(self, m, I, b, d, l, Cd, Ca, Jr,
                 init_state, controller, g=9.81, max_rpm=5000.0):
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
        T_hover = self.m * self.g
        w_hover = np.sqrt(T_hover / (4*self.b))
        rpm_hover = w_hover * 60.0 / (2.0*np.pi)
        # print(f"[INFO] Hover thrust needed = {self.m*self.g:.2f} N, hover rpm ~ {rpm_hover:.1f}")

    def _translational_dynamics(self, state):
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
        return rpm * 2 * np.pi / 60
    
    def _command_to_rpm(self, u1, u2, u3, u4):
        b, d, l = self.b, self.d, self.l
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

def simulate_drone(gains):
    """
    Run the simulation with the specified gains and return a cost.
    The cost is computed as the sum of the settling time (time required for the drone
    to remain continuously within a threshold of the target for T_final seconds)
    and a weighted steady-state error (average error over the final T_final seconds).
    If the drone never settles, the settling time is set to the full simulation time.
    
    gains: [kp_pos, ki_pos, kd_pos, kp_alt, ki_alt, kd_alt, kp_att, ki_att, kd_att]
    """
    # Unpack gains
    kp_pos, ki_pos, kd_pos, kp_alt, ki_alt, kd_alt, kp_att, ki_att, kd_att = gains

    dt = 0.01
    simulation_time = 100.0
    num_steps = int(simulation_time / dt)
    
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
    
    state = {
        'pos': np.array([0.0, 0.0, 0.0]),
        'vel': np.array([0.0, 0.0, 0.0]),
        'angles': np.array([0.0, 0.0, 0.0]),
        'ang_vel': np.array([0.0, 0.0, 0.0]),
        'rpm': np.array([0.0, 0.0, 0.0, 0.0])
    }
    target = {'x': 20.0, 'y': 20.0, 'z': 20.0}
    
    # Initialize the controller with separate altitude gains.
    quad_controller = QuadCopterController(
        state,
        kp_pos, ki_pos, kd_pos,    # lateral position PID gains
        kp_alt, ki_alt, kd_alt,    # altitude PID gains
        kp_att, ki_att, kd_att,    # attitude PID gains
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
    
    positions = []
    time_history = []
    for step in range(num_steps):
        state = drone.update_state(state, target, dt)
        positions.append(state['pos'].copy())
        time_history.append(step * dt)
    positions = np.array(positions)
    time_history = np.array(time_history)
    
    # Compute error (Euclidean distance) at each time step.
    target_vec = np.array([target['x'], target['y'], target['z']])
    errors = np.linalg.norm(positions - target_vec, axis=1)
    
    # Define the settling criteria.
    threshold = 0.5   # meters: maximum error allowed to be considered "settled"
    T_final = 5.0     # seconds: the drone must remain within the threshold for this duration
    final_steps = int(T_final / dt)
    
    # Find the earliest time index such that for the next T_final seconds the error remains below threshold.
    settling_time = simulation_time  # default if never settled
    for i in range(len(errors) - final_steps):
        if np.all(errors[i:i+final_steps] < threshold):
            settling_time = time_history[i]
            break

    # Steady-state error: average error over the last T_final seconds.
    steady_state_error = np.mean(errors[-final_steps:])
    
    # Combine the metrics.
    lambda_factor = 50  # weight for the steady-state error term (adjust as needed)
    cost = settling_time + lambda_factor * steady_state_error
    
    if np.isnan(cost) or np.isinf(cost):
        cost = 1e6
    return cost


# --- Global iteration counter and callback ---
iteration = 0
n_calls = 200
def print_callback(result):
    global iteration
    iteration += 1
    print(f"Iteration {iteration}\{n_calls}: cost = {result.fun:.4f}")

# --- Define search space for 9 parameters ---
space = [
    Real(0.05, 0.2, name='kp_pos'),
    Real(0.0, 0.05, name='ki_pos'),
    Real(0.2, 1.0, name='kd_pos'),
    Real(0.01, 0.1, name='kp_alt'),
    Real(0.0, 0.05, name='ki_alt'),
    Real(0.2, 1.0, name='kd_alt'),
    Real(0.005, 0.05, name='kp_att'),
    Real(0.0, 0.01, name='ki_att'),
    Real(0.01, 0.1, name='kd_att')
]

# --- Run Bayesian Optimization using gp_minimize ---
result = gp_minimize(
    func=simulate_drone, 
    dimensions=space, 
    n_calls=n_calls, 
    random_state=0, 
    callback=[print_callback]
)

print("Best gain values found:")
print(f"kp_pos, ki_pos, kd_pos = {result.x[0]:.4f}, {result.x[1]:.4f}, {result.x[2]:.4f}")
print(f"kp_alt, ki_alt, kd_alt = {result.x[3]:.4f}, {result.x[4]:.4f}, {result.x[5]:.4f}")
print(f"kp_att, ki_att, kd_att = {result.x[6]:.4f}, {result.x[7]:.4f}, {result.x[8]:.4f}")
print(f"Minimum error = {result.fun:.4f}")

plot_convergence(result)
plt.show()
