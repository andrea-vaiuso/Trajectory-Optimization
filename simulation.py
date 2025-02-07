import numpy as np
from SimComponent import Drone, World, Simulation
from utility import showPlot

housing_estate = {
    "id": 1,
    "name": "Housing Estate",
    "min_altitude": 70, 
    "max_altitude": 1000, 
    "noise_penalty": 1,
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
    "min_altitude": 20, 
    "max_altitude": 1000, 
    "noise_penalty": 0,
    "color": "green"
}


# ----------------- Main Usage -----------------
if __name__ == "__main__":

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
        "z": 20,
        "h_speed": 20,
        "v_speed": 8,
    }

    B = {
        "x": 1000,
        "y": 1000,
        "z": 20,
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
    custom_points = []
    num_points = 10
    for i in range(1, num_points + 1):
        t = i / (num_points + 1)
        custom_points.append({
            "x": A["x"] + t * (B["x"] - A["x"]),
            "y": A["y"] + t * (B["y"] - A["y"]),
            "z": A["z"] + t * (B["z"] - A["z"]), # + (-1)**i * 10,
            "h_speed": 20,
            "v_speed": 8,
        })

    # Load angle noise model from npy file
    angle_noise_model = np.load("dnn_sound_model/angles_swl.npy")
    # Initialize simulation
    sim = Simulation(drone, world, angle_noise_model)

    # Set the cost gains
    distAB = np.sqrt((B["x"] - A["x"])**2 + (B["y"] - A["y"])**2 + (B["z"] - A["z"])**2) * 1.1
    maxvel = np.sqrt(drone.max_horizontal_speed**2 + drone.max_vertical_speed**2)
    noise_rule_cost_gain = 1
    altitude_rule_cost_gain = 1e-4
    time_cost_gain = maxvel / distAB
    distance_cost_gain = 1 / distAB
    power_cost_gain = time_cost_gain / drone.hover_rpm

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

