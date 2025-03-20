"""
bayes_optimizer.py

This file implements a Bayesian optimizer for tuning the controller gains defined in advanced_controller.py.
The objective is to reduce the time to reach the final target while also minimizing the final distance error
and reducing oscillations in the pitch and roll values.
"""

import numpy as np
from skopt import gp_minimize
from advanced_controller import QuadCopterController, QuadcopterModel
import matplotlib.pyplot as plt

iteration = 0
n_calls = 500
costs = []

def simulate_gains(gains: list) -> float:
    """
    Run the quadcopter simulation with the specified gains and return a cost.
    The cost is defined as:
         cost = (time to reach final target)
              + penalty_weight_distance * (final distance error)
              + osc_weight * (pitch oscillation + roll oscillation)
    
    Parameters:
        gains (list of float): List of 12 gain parameters in the following order:
            [kp_pos, ki_pos, kd_pos, kp_alt, ki_alt, kd_alt,
             kp_att, ki_att, kd_att, kp_yaw, ki_yaw, kd_yaw].
    
    Returns:
        float: The overall cost.
               If the final target is not reached within the maximum simulation time,
               the cost reflects both the maximum time and the remaining distance error.
    """
    global iteration, costs
    iteration += 1

    # Simulation parameters (same as advanced_controller)
    dt = 0.007
    simulation_time = 300.0
    num_steps = int(simulation_time / dt)

    # Define the same targets as in advanced_controller
    targets = [
        {'x': 10.0, 'y': 10.0, 'z': 70.0},  # Start near origin but at high altitude (rapid ascent required)
        {'x': 90.0, 'y': 10.0, 'z': 70.0},  # Far x, near y, maintaining high altitude (long horizontal flight)
        {'x': 90.0, 'y': 90.0, 'z': 90.0},  # Far in both x and y with an even higher altitude (climb and diagonal flight)
        {'x': 10.0, 'y': 90.0, 'z': 20.0},  # Sharp maneuver: near x, far y, with a dramatic altitude drop
        {'x': 50.0, 'y': 50.0, 'z': 40.0},  # Central target with intermediate altitude (transition maneuver)
        {'x': 60.0, 'y': 60.0, 'z': 40.0},  # Hovering target 1
        {'x': 70.0, 'y': 70.0, 'z': 40.0},  # Gradual move to hovering target 2
        {'x': 80.0, 'y': 80.0, 'z': 40.0},  # Gradual move to hovering target 3
        {'x': 10.0, 'y': 10.0, 'z': 10.0}   # Final target: near origin at low altitude (drone must come to a near stop)
    ]
    current_target_idx = 0
    target = targets[current_target_idx]

    # Drone physical parameters (same as advanced_controller)
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

    # Initialize lists for recording pitch and roll values over time.
    angles_history = []

    # Unpack the gains (12 parameters)
    kp_pos, ki_pos, kd_pos, kp_alt, ki_alt, kd_alt, \
    kp_att, ki_att, kd_att = gains

    # Initialize the controller and drone model with the given gains
    quad_controller = QuadCopterController(
        state, 
        kp_pos, ki_pos, kd_pos,     
        kp_alt, ki_alt, kd_alt,     
        kp_att, ki_att, kd_att,
        kp_att, ki_att, kd_att,   
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

    final_time = simulation_time
    # Run simulation loop
    for step in range(num_steps):
        state = drone.update_state(state, target, dt)
        angles_history.append(state['angles'].copy())
        current_time = step * dt
        pos_error = np.linalg.norm(state['pos'] - np.array([target['x'], target['y'], target['z']]))
        if pos_error < 3.0:
            current_target_idx += 1
            if current_target_idx >= len(targets):
                final_time = current_time
                break
            else:
                target = targets[current_target_idx]

    # Compute final distance error from the final target
    final_target = targets[-1]
    final_distance = np.linalg.norm(state['pos'] - np.array([final_target['x'], final_target['y'], final_target['z']]))

    # Compute oscillation penalty for pitch and roll.
    # Convert angles_history to numpy array for easier diff computation.
    angles_history = np.array(angles_history)
    pitch_osc = np.sum(np.abs(np.diff(angles_history[:, 0])))
    roll_osc  = np.sum(np.abs(np.diff(angles_history[:, 1])))
    
    # Define penalty weights for distance error and oscillations
    osc_weight = 3.0

    cost = final_time + (final_distance ** 0.9) + osc_weight * (pitch_osc + roll_osc)
    costs.append(cost)
    print(f"{iteration}/{n_calls} - SimTime: {final_time:.2f} s, "
          f"DistFinalTarget: {final_distance:.2f} m, "
          f"Osc: {pitch_osc+pitch_osc:.2f}, Cost: {cost:.2f}, Best: {min(costs):.2f} it({int(costs.index(min(costs)))+1})")
    return cost

def objective(gains: list) -> float:
    """
    Objective function for Bayesian optimization. Runs the simulation with the provided gains
    and returns the overall cost that combines time to reach the final target, the final distance error,
    and oscillations in pitch and roll (the lower, the better).

    Parameters:
        gains (list of float): List of 12 controller gain parameters.

    Returns:
        float: Overall cost.
    """
    return simulate_gains(gains)

if __name__ == "__main__":
    # Define search space bounds for the 12 gain parameters.
    # Adjust these bounds based on your system's requirements.
    space = [
        (0.00001, 20),   # kp_pos
        (1e-10, 1e-1),  # ki_pos
        (0.00001, 20),    # kd_pos
        (0.00001, 20),    # kp_alt
        (1e-10, 1e-1),  # ki_alt
        (0.00001, 20),   # kd_alt
        (0.00001, 20),   # kp_att
        (1e-10, 1e-1),  # ki_att
        (0.00001, 20),   # kd_att
    ]

    # Run Bayesian optimization with 100 evaluations (n_calls)
    res = gp_minimize(objective, space, n_calls=n_calls, random_state=42)

    print("Best gain parameters found:")
    print(res.x)
    print("Minimum cost:")
    print(res.fun)

    # Plot best costs over iterations
    plt.plot(costs)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost over Iterations")
    plt.show()


