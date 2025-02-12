from SimComponent import Simulation
from World import World
from Drone import Drone
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real  # Per definire dimensioni discrete/continue
from utility import showPlot, plotCosts
import time
import datetime
import json

iterations = 0
costs = [1e9]

def get_cost_gains(A: dict, B: dict, drone: Drone):
    """
    Calcola i cost gains basati sui punti A e B e sulle specifiche del drone.
    """
    distAB = np.sqrt((B["x"] - A["x"])**2 + (B["y"] - A["y"])**2 + (B["z"] - A["z"])**2) * 1.1
    maxvel = np.sqrt(drone.max_horizontal_speed**2 + drone.max_vertical_speed**2)
    noise_rule_cost_gain = 1
    altitude_rule_cost_gain = 1
    time_cost_gain = maxvel / distAB
    distance_cost_gain = 1 / distAB
    power_cost_gain = time_cost_gain / drone.hover_rpm
    return noise_rule_cost_gain, altitude_rule_cost_gain, time_cost_gain, distance_cost_gain, power_cost_gain

def main():
    # Optimization parameters
    grid_size = 10 # Grid size (in meters)
    max_world_size = 1000 # Maximum world size (in meters)
    num_points = 7 # Number of intermediate points
    n_iterations = 500 # Number of optimization iterations
    perturbation_factor = 0.35 # Perturbation factor for the maximum offset starting from the distance between A and B
    grid_step = 2

    print("Loading world...")
    world = World.load_world("world.pkl")

    # Definizione dei punti A (inizio) e B (fine)
    A = {"x": 0, "y": 0, "z": 100, "h_speed": 20, "v_speed": 8}
    B = {"x": 1000, "y": 1000, "z": 100, "h_speed": 20, "v_speed": 8}

    drone = Drone(
        model_name="DJI Matrice 300 RTK",
        x=A["x"],
        y=A["y"],
        z=A["z"],
        min_RPM=2100,
        max_RPM=5000,
        hover_RPM=2700,
        max_horizontal_speed=20.0,
        max_vertical_speed=8.0
    )
    print("Loading noise model...")
    angle_noise_model = np.load("dnn_sound_model/angles_swl.npy")
    print("Initializing simulation...")
    sim = Simulation(drone, world, angle_noise_model)

    # Calcola i cost gains
    noise_gain, altitude_gain, time_gain, distance_gain, power_gain = get_cost_gains(A, B, drone)
    
    # Calcola la distanza tra A e B e definisci la perturbazione massima
    distAB = np.sqrt((B["x"] - A["x"])**2 + (B["y"] - A["y"])**2 + (B["z"] - A["z"])**2)
    perturbation_factor = 0.25  # ad esempio il 25% della distanza totale
    max_offset = perturbation_factor * distAB

    def cost_function(params):
        """
        Calcola il costo totale della traiettoria generata con num_points intermedi.
        
        I parametri ottimizzati sono organizzati per ciascun punto come:
        [offset_x, offset_y, offset_z, h_speed, v_speed]
        Qui applichiamo la discretizzazione interamente nella funzione: 
        gli offset vengono arrotondati all'intero più vicino e moltiplicati per grid_step.
        """
        global iterations, costs
        iterations += 1
        custom_points = []
        for i in range(num_points):
            t = (i + 1) / (num_points + 1)
            base_point = {
                "x": A["x"] + t * (B["x"] - A["x"]),
                "y": A["y"] + t * (B["y"] - A["y"]),
                "z": A["z"] + t * (B["z"] - A["z"])
            }
            # Discretizziamo gli offset: li arrotondiamo all'intero più vicino,
            # poi moltiplichiamo per grid_step per ottenere il valore in metri.
            offset_x = int(round(params[i*5 + 0])) * grid_step
            offset_y = int(round(params[i*5 + 1])) * grid_step
            offset_z = int(round(params[i*5 + 2])) * grid_step
            
            # Assicuriamoci che la somma base_point + offset sia nel range
            final_x = np.clip(base_point["x"] + offset_x, 0, max_world_size)
            final_y = np.clip(base_point["y"] + offset_y, 0, max_world_size)
            final_z = np.clip(base_point["z"] + offset_z, 0, max_world_size)
            
            point = {
                "x": final_x,
                "y": final_y,
                "z": final_z,
                "h_speed": params[i*5 + 3],
                "v_speed": params[i*5 + 4]
            }
            custom_points.append(point)
        print(f"Iteration ({iterations}/{n_iterations}) | Best: {min(costs):.2f} | ", end="")
        _, total_cost, _, _ = sim.simulate_trajectory(
            point_a=A, point_b=B, dt=1,
            horizontal_threshold=5.0, vertical_threshold=2.0,
            custom_points=custom_points,
            print_log=False,
            noise_rule_cost_gain=noise_gain,
            altitude_rule_cost_gain=altitude_gain,
            time_cost_gain=time_gain,
            distance_cost_gain=distance_gain,
            power_cost_gain=power_gain,
            print_info=False,
            save_log=False
        )
        costs.append(total_cost)
        return total_cost

    # Costruiamo i bounds per ciascun punto intermedio.
    # Per ogni punto, calcoliamo il base_point (interpolazione lineare tra A e B)
    # e impostiamo i bounds per gli offset in modo che base_point + offset sia in [0, max_world_size].
    # I bounds per gli offset sono definiti in "unità discrete" (multipli di grid_step).
    dimensions = []
    x0 = []  # Punto iniziale per l'ottimizzazione

    for i in range(num_points):
        t = (i + 1) / (num_points + 1)
        base_point = {
            "x": A["x"] + t * (B["x"] - A["x"]),
            "y": A["y"] + t * (B["y"] - A["y"]),
            "z": A["z"] + t * (B["z"] - A["z"])
        }
        # Calcola i limiti per ciascun offset
        lower_x = max(-max_offset, -base_point["x"])
        upper_x = min(max_offset, max_world_size - base_point["x"])
        lower_y = max(-max_offset, -base_point["y"])
        upper_y = min(max_offset, max_world_size - base_point["y"])
        lower_z = max(-max_offset, -base_point["z"])
        upper_z = min(max_offset, max_world_size - base_point["z"])

        # Convertiamo i bounds in unità discrete (multipli di grid_step)
        lx_disc = int(np.ceil(lower_x / grid_step))
        ux_disc = int(np.floor(upper_x / grid_step))
        ly_disc = int(np.ceil(lower_y / grid_step))
        uy_disc = int(np.floor(upper_y / grid_step))
        lz_disc = int(np.ceil(lower_z / grid_step))
        uz_disc = int(np.floor(upper_z / grid_step))

        dimensions.extend([
            Integer(lx_disc, ux_disc),  # offset_x (in unità di grid_step)
            Integer(ly_disc, uy_disc),  # offset_y
            Integer(lz_disc, uz_disc)   # offset_z
        ])

        dimensions.extend([
            Real(5, 20),   # h_speed
            Real(3, 8)     # v_speed
        ])

        # Punto iniziale: per offset usiamo 0 (cioè 0 * grid_step = 0)
        # per le velocità usiamo i valori di A.
        x0.extend([0, 0, 0, A["h_speed"], A["v_speed"]])

    print("Starting optimization...")
    start_time = time.time()

    result = gp_minimize(cost_function, dimensions, x0=x0, n_calls=n_iterations, random_state=0)

    end_time = time.time()
    print("Optimal cost:", result.fun)
    print("Total optimization time: {:.2f} seconds".format(end_time - start_time))

    # Estrae i migliori parametri e li converte nei punti custom,
    # applicando anch'essi la discretizzazione.
    best_params = result.x

    print("Simulating best trajectory...")
    custom_points_best = []
    for i in range(num_points):
        t = (i + 1) / (num_points + 1)
        base_point = {
            "x": A["x"] + t * (B["x"] - A["x"]),
            "y": A["y"] + t * (B["y"] - A["y"]),
            "z": A["z"] + t * (B["z"] - A["z"])
        }
        final_x = np.clip(base_point["x"] + int(round(best_params[i*5 + 0])) * grid_step, 0, max_world_size)
        final_y = np.clip(base_point["y"] + int(round(best_params[i*5 + 1])) * grid_step, 0, max_world_size)
        final_z = np.clip(base_point["z"] + int(round(best_params[i*5 + 2])) * grid_step, 0, max_world_size)
        
        point = {
            "x": final_x,
            "y": final_y,
            "z": final_z,
            "h_speed": best_params[i*5 + 3],
            "v_speed": best_params[i*5 + 4]
        }
        custom_points_best.append(point)

    print("Best points:", custom_points_best)
    # Save the best custom points into a file
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    np.save(f"OptimizedTrajectory/{time_str}_bestpoints.npy", custom_points_best)
    plotCosts(costs[1:], save=True, datetime=time_str)
    # Create a dictionary to store optimization information
    optimization_info = {
        "n_iterations": int(n_iterations),
        "best_cost": float(result.fun),
        "optimization_time_seconds": float(end_time - start_time),
        "n_points": int(num_points),
        "coustom_points": custom_points_best,
        "A": {k: float(v) for k, v in A.items()},
        "B": {k: float(v) for k, v in B.items()},
        "grid_size": int(grid_size),
        "max_world_size": int(max_world_size),
        "perturbation_factor": float(perturbation_factor),
        "grid_step": int(grid_step),
        "noise_rule_cost_gain": float(noise_gain),
        "altitude_rule_cost_gain": float(altitude_gain),
        "time_cost_gain": float(time_gain),
        "distance_cost_gain": float(distance_gain),
        "power_cost_gain": float(power_gain),
        "drone": drone.to_dict(),
        "angle_noise_model": "dnn_sound_model/angles_swl.npy"
    }

    # Save the dictionary to a JSON file
    with open(f"OptimizedTrajectory/{time_str}_optimization_info.json", "w") as json_file:
        json.dump(optimization_info, json_file, indent=4)

    sim = Simulation(drone, world, angle_noise_model)
    
    print("Executing simulation...")
    trajectory, total_cost, log_data, all_targets = sim.simulate_trajectory(
        point_a=A, point_b=B, dt=0.1,
        horizontal_threshold=5.0, vertical_threshold=2.0,
        custom_points=custom_points_best,
        print_log=False,
        noise_rule_cost_gain=noise_gain,
        altitude_rule_cost_gain=altitude_gain,
        time_cost_gain=time_gain,
        distance_cost_gain=distance_gain,
        power_cost_gain=power_gain
    )

    showPlot(trajectory, A, B, all_targets, world, grid_size, max_world_size, log_data)

if __name__ == "__main__":
    main()
