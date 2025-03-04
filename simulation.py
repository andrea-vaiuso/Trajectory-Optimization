import numpy as np
from Entity.Simulation import Simulation
from Entity.World import World
from Entity.Drone import Drone
from utility import showPlot, show2DWorld
from Old_scripts.optimization import get_cost_gains
import json
from optimization import execute_simulation
import yaml

def create_AtoB_custom_points(A, B, num_points):
    custom_points = []
    for _ in range(1, num_points + 1):
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

def load_custom_points_npy(file):
    return np.load(file, allow_pickle=True)

# ----------------- Main Usage -----------------
if __name__ == "__main__":
    with open("optimization_params.yaml", "r") as file:
        params = yaml.safe_load(file)

    grid_size = params["grid_size"]
    max_world_size = params["max_world_size"]
    num_points = params["num_points"]
    n_iterations = params["n_iterations"]
    perturbation_factor = params["perturbation_factor"]
    grid_step = params["grid_step"]
    world_file_name = params["world_file_name"]
    A = params["A"]
    B = params["B"]

    print("Loading world...")
    world = World.load_world(world_file_name)


    print("Creating drone...")
    drone = Drone(
        model_name=params["drone_model_name"],
        x=A["x"],
        y=A["y"],
        z=A["z"],
        min_RPM=params["min_RPM"],
        max_RPM=params["max_RPM"],
        hover_RPM=params["hover_RPM"],
        max_horizontal_speed=params["max_horizontal_speed"],
        max_vertical_speed=params["max_vertical_speed"]
    )

    print("Loading noise model...")
    angle_noise_model = np.load(params["noise_model"])
    print("Creating custom points...")
    #custom_points = load_custom_points("OptimizedTrajectory/2025-02-14_08-58-00/optimization_info.json")
    #custom_points = load_custom_points("OptimizedTrajectory/2025-02-11_20-55-31_optimization_info.json")
    custom_points = load_custom_points_npy("OptimizedTrajectory/2025-02-28_01-26-52/bestpoints.npy")

    # Start simulation
    print("Simulating trajectory...")
    execute_simulation(drone, world, angle_noise_model, A, B, custom_points, get_cost_gains(A, B, drone), log_folder="OptimizedTrajectory/2025-02-28_01-26-52")

