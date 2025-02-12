import numpy as np
from Entity.Simulation import Simulation
from Entity.World import World
from Entity.Drone import Drone
from utility import showPlot, show2DWorld
from Old_scripts.optimization import get_cost_gains
import json
from optimization import execute_simulation

def create_AtoB_custom_points(A, B, num_points):
    custom_points = []
    for i in range(1, num_points + 1):
        new_point = {
            "x": A["x"] + (B["x"] - A["x"]) * i / (num_points + 1),
            "y": A["y"] + (B["y"] - A["y"]) * i / (num_points + 1),
            "z": A["z"] + (B["z"] - A["z"]) * i / (num_points + 1),
            "h_speed": 20,
            "v_speed": 8,
        }
        custom_points.append(new_point)
    return custom_points

def create_random_custom_points(num_points, max_world_size, min_v_speed = 15, min_h_speed = 12):
    custom_points = []
    for i in range(1, num_points + 1):
        new_point = {
            "x": np.random.uniform(0, max_world_size),
            "y": np.random.uniform(0, max_world_size),
            "z": np.random.uniform(0, max_world_size),
            "h_speed": np.random.uniform(min_v_speed, 20),
            "v_speed": np.random.uniform(min_h_speed, 15),
        }
        custom_points.append(new_point)
    return custom_points


def create_custom_points_towards_B(A, B, num_points):
    custom_points = []
    for i in range(1, num_points + 1):
        t = i / (num_points + 1)
        new_point = {
            "x": A["x"] + t * (B["x"] - A["x"]) + np.random.uniform(-150, 150),
            "y": A["y"] + t * (B["y"] - A["y"]) + np.random.uniform(-150, 150),
            "z": A["z"] + t * (B["z"] - A["z"]) + np.random.uniform(-150, 150),
            "h_speed": np.random.uniform(15, 20),
            "v_speed": np.random.uniform(12, 15),
        }
        custom_points.append(new_point)
    return custom_points

def load_custom_points(file):
    with open(file, "r") as f:
        sim_file = json.load(f)
        return sim_file["custom_points"]

# ----------------- Main Usage -----------------
if __name__ == "__main__":

    angle_noise_model = np.load("dnn_sound_model/angles_swl.npy")

    max_world_size = 1000
    grid_size = 10
    num_points = 7 
    print("Loading world...")
    world = World.load_world("world_winterthur.pkl")

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

    print("Creating custom points...")
    custom_points = create_random_custom_points(num_points, max_world_size)
    #custom_points = load_custom_points("OptimizedTrajectory/2025-02-11_20-55-31_optimization_info.json")

    # Start simulation
    print("Simulating trajectory...")
    execute_simulation(drone, world, angle_noise_model, A, B, custom_points, get_cost_gains(A, B, drone))

