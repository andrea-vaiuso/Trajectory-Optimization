from SimComponent import Simulation
from World import World
from Drone import Drone
import numpy as np
from skopt import gp_minimize
from utility import showPlot
import time

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
    grid_size = 10
    max_world_size = 1000
    num_points = 6  # Numero di punti intermedi
    n_iterations = 10

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
        dove gli offset vengono aggiunti al punto base ottenuto per interpolazione lineare.
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
            # Gli offset sono vincolati dai bounds definiti
            offset_x = params[i*5 + 0]
            offset_y = params[i*5 + 1]
            offset_z = params[i*5 + 2]
            
            # La somma base_point + offset sarà sicuramente nel range,
            # visto che i bounds sono stati costruiti appositamente,
            # ma per sicurezza si può applicare comunque il clipping.
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
        print(f"Iteration ({iterations}/{n_iterations} | Best: {min(costs):.2f}): ", end="")
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
    # Per ogni punto, calcoliamo il punto base (interpolazione lineare tra A e B)
    # e impostiamo i bounds per gli offset in modo che base_point + offset rimanga in [0, max_world_size]
    bounds = []
    for i in range(num_points):
        t = (i + 1) / (num_points + 1)
        base_point = {
            "x": A["x"] + t * (B["x"] - A["x"]),
            "y": A["y"] + t * (B["y"] - A["y"]),
            "z": A["z"] + t * (B["z"] - A["z"])
        }
        # Calcola i limiti per ciascun offset, in base al punto base
        lower_x = max(-max_offset, -base_point["x"])
        upper_x = min(max_offset, max_world_size - base_point["x"])
        lower_y = max(-max_offset, -base_point["y"])
        upper_y = min(max_offset, max_world_size - base_point["y"])
        lower_z = max(-max_offset, -base_point["z"])
        upper_z = min(max_offset, max_world_size - base_point["z"])
        
        bounds.extend([
            (lower_x, upper_x),   # offset_x
            (lower_y, upper_y),   # offset_y
            (lower_z, upper_z),   # offset_z
            (5, 20),              # h_speed
            (3, 8)                # v_speed
        ])
    print("Starting optimization...")
    start_time = time.time()
    result = gp_minimize(cost_function, bounds, n_calls=n_iterations, random_state=0)
    end_time = time.time()
    print("Optimal cost:", result.fun)
    print("Total optimization time: {:.2f} seconds".format(end_time - start_time))

    # Estrae i migliori parametri e li converte nei punti custom
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
        # Anche qui, applico il clipping per sicurezza (sebbene non strettamente necessario
        # dato che i bounds dovrebbero garantirlo)
        final_x = np.clip(base_point["x"] + best_params[i*5 + 0], 0, max_world_size)
        final_y = np.clip(base_point["y"] + best_params[i*5 + 1], 0, max_world_size)
        final_z = np.clip(base_point["z"] + best_params[i*5 + 2], 0, max_world_size)
        
        point = {
            "x": final_x,
            "y": final_y,
            "z": final_z,
            "h_speed": best_params[i*5 + 3],
            "v_speed": best_params[i*5 + 4]
        }
        custom_points_best.append(point)

    print("Best points:", custom_points_best)

    sim = Simulation(drone, world, angle_noise_model)
    
    print("Executing simulation...")
    # Esegui la simulazione con i punti intermedi ottimali
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
