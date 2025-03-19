import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence
from advanced_controller import QuadCopterController, QuadcopterModel

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
    target = {'x': 40.0, 'y': 40.0, 'z': 0.0}
    
    # Initialize the controller with separate altitude gains.
    quad_controller = QuadCopterController(
        state, 
        kp_pos, ki_pos, kd_pos,    # Position PID
        kp_alt, ki_alt, kd_alt,    # Altitude PID
        kp_att, ki_att, kd_att,    # Attitude PID
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
n_calls = 100
def print_callback(result):
    global iteration
    iteration += 1
    print(f"Iteration {iteration}\{n_calls}: cost = {result.fun:.4f}")

# --- Define search space for 9 parameters ---
space = [
    Real(0.15, 0.25, name='kp_pos'),
    Real(0.0, 0.01, name='ki_pos'),
    Real(0.75, 1.25, name='kd_pos'),
    Real(0.0075, 0.0125, name='kp_alt'),
    Real(0.0, 0.01, name='ki_alt'),
    Real(0.15, 0.25, name='kd_alt'),
    Real(0.0375, 0.0625, name='kp_att'),
    Real(0.0, 0.01, name='ki_att'),
    Real(0.075, 0.125, name='kd_att')
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
