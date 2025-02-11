import numpy as np
from SimComponent import Simulation
from World import World
from Drone import Drone
from utility import showPlot
from optimization import get_cost_gains

housing_estate = {
    "id": 1,
    "name": "Housing Estate",
    "min_altitude": 150, 
    "max_altitude": 1000, 
    "noise_penalty": 2,
    "color": "blue"
}

industrial_area = {
    "id": 2,
    "name": "Industrial Area",
    "min_altitude": 70, 
    "max_altitude": 1000, 
    "noise_penalty": 1,
    "color": "yellow"
}

open_field = {
    "id": 3,
    "name": "Open Field",
    "min_altitude": 0, 
    "max_altitude": 1000, 
    "noise_penalty": 0,
    "color": "green"
}

def create_custom_points(A, B, num_points):
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

# ----------------- Main Usage -----------------
if __name__ == "__main__":

    angle_noise_model = np.load("dnn_sound_model/angles_swl.npy")

    max_world_size = 1000
    grid_size = 10
    print("Creating world...")
    world = World(grid_size=grid_size, max_world_size=max_world_size)

    # Initialize the area as open field
    world.set_area_parameters(0, max_world_size, 0, max_world_size, open_field)
    # Set the areas
    world.set_area_parameters(320, 640, 40, 750, housing_estate)
    world.set_area_parameters(360, 790, 780, 985, industrial_area)

    A = {
        "x": 0,
        "y": 0,
        "z": 500,
        "h_speed": 20,
        "v_speed": 8,
    }

    B = {
        "x": 1000,
        "y": 1000,
        "z": 500,
        "h_speed": 20,
        "v_speed": 8,
    }

    print("Initializing drone...")
    drone = Drone(
        model_name="DJI Matrice 300 RTK",
        x=A["x"], y=A["y"], z=A["z"], 
        min_RPM=2100,
        max_RPM=5000,
        hover_RPM=2700,
        max_horizontal_speed=20.0,  
        max_vertical_speed=8.0
    )

    print("Creating custom points...")
    custom_points = create_custom_points(A, B, 6)
    
    # Initialize simulation
    sim = Simulation(drone, world, angle_noise_model)

    # Set the cost gains
    noise_rule_cost_gain, altitude_rule_cost_gain, time_cost_gain, distance_cost_gain, power_cost_gain = get_cost_gains(A, B, drone)

    # Start simulation
    print("Simulating trajectory...")
    trajectory, total_cost, log_data, all_targets = sim.simulate_trajectory(point_a = A, point_b = B, dt = 0.1,
                                                                  horizontal_threshold = 5.0, vertical_threshold=2.0,
                                                                  custom_points = custom_points,
                                                                  print_log = False,
                                                                  noise_rule_cost_gain = noise_rule_cost_gain,
                                                                  altitude_rule_cost_gain = altitude_rule_cost_gain,
                                                                  time_cost_gain = time_cost_gain,
                                                                  distance_cost_gain = distance_cost_gain,
                                                                  power_cost_gain = power_cost_gain
                                                                )

    showPlot(trajectory, A, B, all_targets, world, grid_size, max_world_size, log_data)

