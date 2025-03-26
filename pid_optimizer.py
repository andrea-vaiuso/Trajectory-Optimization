import numpy as np
from bayes_opt import BayesianOptimization

iteration = 0
n_iter = 500
costs = []

# Import the necessary classes and functions from advanced_controller.py
from advanced_controller import (
    QuadCopterController,
    QuadcopterModel,
    compute_moving_target,
    wrap_angle
)

def simulate_pid(kp_pos, ki_pos, kd_pos,
                 kp_alt, ki_alt, kd_alt,
                 kp_att, ki_att, kd_att):
    """
    Runs the simulation with the provided PID gains and returns the cost.
    The cost is computed as:
       cost = final_time + (final_distance ** 0.9) + osc_weight*(pitch_osc + roll_osc)
    where:
       - final_time is the simulation time when the drone reached the final target (or the max simulation time)
       - final_distance is the remaining distance to the final target
       - pitch_osc and roll_osc are the sums of absolute differences of pitch and roll over time.
    """
    # Simulation parameters
    dt = 0.007
    simulation_time = 200.0
    num_steps = int(simulation_time / dt)
    
    # Define waypoints (with desired speeds)
    waypoints = [
        {'x': 10.0, 'y': 10.0, 'z': 70.0, 'v':10},  # Start near origin but at high altitude (rapid ascent required)
        {'x': 90.0, 'y': 10.0, 'z': 70.0, 'v':10},  # Far x, near y, maintaining high altitude (long horizontal flight)
        {'x': 90.0, 'y': 90.0, 'z': 90.0, 'v':10},   # Far in both x and y with an even higher altitude (climb and diagonal flight)
        {'x': 10.0, 'y': 90.0, 'z': 20.0, 'v':10},   # Sharp maneuver: near x, far y, with a dramatic altitude drop
        {'x': 50.0, 'y': 50.0, 'z': 40.0, 'v':10},  # Central target with intermediate altitude (transition maneuver)
        {'x': 60.0, 'y': 60.0, 'z': 40.0, 'v':10},  # Hovering target 1
        {'x': 70.0, 'y': 70.0, 'z': 40.0, 'v':10},  # Gradual move to hovering target 2
        {'x': 80.0, 'y': 80.0, 'z': 40.0, 'v':10},  # Gradual move to hovering target 3
        {'x': 10.0, 'y': 10.0, 'z': 10.0, 'v':10}   # Final target: near origin at low altitude (drone must come to a near stop)
    ]
    
    # Initial drone state
    state = {
        'pos': np.array([0.0, 0.0, 0.0]),
        'vel': np.array([0.0, 0.0, 0.0]),
        'angles': np.array([0.0, 0.0, 0.0]),
        'ang_vel': np.array([0.0, 0.0, 0.0]),
        'rpm': np.array([0.0, 0.0, 0.0, 0.0])
    }
    
    # PID gains for yaw remain fixed
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
    
    # Create the QuadCopterController with the optimization parameters
    quad_controller = QuadCopterController(
        state, 
        kp_pos, ki_pos, kd_pos,     # PID for position
        kp_alt, ki_alt, kd_alt,     # PID for altitude
        kp_att, ki_att, kd_att,     # PID for attitude (roll and pitch)
        kp_yaw, ki_yaw, kd_yaw,     # PID for yaw (fixed)
        m=params['m'], g=params['g'], b=params['b'],
        u1_limit=100.0, u2_limit=10.0, u3_limit=5.0, u4_limit=10.0
    )
    
    # Create the drone model
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
    
    # Dynamic target initialization
    current_seg_idx = 0
    seg_start = state['pos'].copy()
    seg_end = np.array([waypoints[current_seg_idx]['x'],
                        waypoints[current_seg_idx]['y'],
                        waypoints[current_seg_idx]['z']])
    v_des = waypoints[current_seg_idx]['v']
    k_lookahead = 1.0
    
    # Data logs for computing cost
    positions = []
    angles_history = []
    time_history = []
    
    final_time = simulation_time  # default final time if never reached final target
    threshold = 2.0  # distance threshold to consider final target reached
    
    for step in range(num_steps):
        # Compute dynamic target along current segment
        target_dynamic, progress = compute_moving_target(state['pos'], seg_start, seg_end, v_des, k=k_lookahead)
        # If progress on the segment exceeds a threshold, move to the next segment
        if progress >= 0.8:
            current_seg_idx += 1
            if current_seg_idx < len(waypoints):
                seg_start = seg_end
                seg_end = np.array([waypoints[current_seg_idx]['x'],
                                    waypoints[current_seg_idx]['y'],
                                    waypoints[current_seg_idx]['z']])
                v_des = waypoints[current_seg_idx]['v']
                target_dynamic, progress = compute_moving_target(state['pos'], seg_start, seg_end, v_des, k=k_lookahead)
            else:
                target_dynamic = seg_end  # Final target
        
        # Update the drone state based on the dynamic target
        state = drone.update_state(state, 
                                   {'x': target_dynamic[0],
                                    'y': target_dynamic[1],
                                    'z': target_dynamic[2]},
                                   dt)
        current_time = step * dt
        
        positions.append(state['pos'].copy())
        angles_history.append(state['angles'].copy())
        time_history.append(current_time)

        final_target = np.array([waypoints[-1]['x'], waypoints[-1]['y'], waypoints[-1]['z']])
        if np.linalg.norm(state['pos'] - final_target) < threshold:
            final_time = current_time
            break
        

    positions = np.array(positions)
    angles_history = np.array(angles_history)
    
    # Compute cost
    final_target = {'x': waypoints[-1]['x'], 'y': waypoints[-1]['y'], 'z': waypoints[-1]['z']}
    final_distance = np.linalg.norm(state['pos'] - np.array([final_target['x'],
                                                              final_target['y'],
                                                              final_target['z']]))
    # Compute oscillation penalty for pitch and roll
    pitch_osc = np.sum(np.abs(np.diff(angles_history[:, 0])))
    roll_osc  = np.sum(np.abs(np.diff(angles_history[:, 1])))
    osc_weight = 3.0

    cost = final_time + (final_distance ** 0.9) + osc_weight * (pitch_osc + roll_osc)
    
    return cost, final_time

def objective(kp_pos, ki_pos, kd_pos,
              kp_alt, ki_alt, kd_alt,
              kp_att, ki_att, kd_att):
    """
    The objective function for Bayesian optimization.
    Since we want to minimize the cost, we return its negative (BayesianOptimization maximizes).
    """
    global iteration
    iteration += 1
    cost, final_time = simulate_pid(kp_pos, ki_pos, kd_pos,
                        kp_alt, ki_alt, kd_alt,
                        kp_att, ki_att, kd_att)
    costs.append(-cost)
    print(f"{iteration}/{n_iter}: cost={-cost}, best: {-max(costs)}, final_time={final_time}")
    return -cost

def main():
    # Define the bounds for the optimization variables
    pbounds = {
        'kp_pos': (0.005, 0.5),   # around 0.05
        'ki_pos': (1e-8, 1e-6),   # around 1e-7
        'kd_pos': (0.01, 1.0),    # around 0.1
        'kp_alt': (1.0, 10.0),    # around 5.0
        'ki_alt': (1e-4, 1e-2),   # around 0.00061276
        'kd_alt': (1.0, 20.0),    # around 8.59
        'kp_att': (1.0, 3.0),     # around 1.95
        'ki_att': (1e-4, 1e-3),   # around 0.000464
        'kd_att': (0.1, 1.0)      # around 0.42447
    }
    
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
    )
    
    # Run the optimization: first with some random initial points then with iterations.
    optimizer.maximize(
        init_points=5,
        n_iter=n_iter,
    )
    
    print("Best parameters found:")
    best = optimizer.max['params']

    print("kp_yaw, ki_yaw, kd_yaw = 0.5, 1e-6, 0.1")
    print("kp_pos, ki_pos, kd_pos = {:.5g}, {:.5g}, {:.5g}".format(
        best['kp_pos'], best['ki_pos'], best['kd_pos']))
    print("kp_alt, ki_alt, kd_alt = {:.5g}, {:.5g}, {:.5g}".format(
        best['kp_alt'], best['ki_alt'], best['kd_alt']))
    print("kp_att, ki_att, kd_att = {:.5g}, {:.5g}, {:.5g}".format(
        best['kp_att'], best['ki_att'], best['kd_att']))

if __name__ == "__main__":
    main()
