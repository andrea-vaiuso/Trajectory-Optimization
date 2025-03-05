import yaml
from optimization import optimize
from simulation import execute_simulation
from Entity.World import World
from Entity.Drone import Drone

def main():
    with open("optimization_params.yaml", "r") as file:
        params = yaml.safe_load(file)

    print("Loading world...")
    world = World.load_world(params["world_file_name"])

    print("Creating drones...")
    drone_1 = Drone(
        model_name=params["drone_model_name"],
        x=params["A"][0]["x"],
        y=params["A"][0]["y"],
        z=params["A"][0]["z"],
        min_RPM=params["min_RPM"],
        max_RPM=params["max_RPM"],
        hover_RPM=params["hover_RPM"],
        max_horizontal_speed=params["max_horizontal_speed"],
        max_vertical_speed=params["max_vertical_speed"]
    )
    drone_2 = Drone(
        model_name=params["drone_model_name"],
        x=params["A"][1]["x"],
        y=params["A"][1]["y"],
        z=params["A"][1]["z"],
        min_RPM=params["min_RPM"],
        max_RPM=params["max_RPM"],
        hover_RPM=params["hover_RPM"],
        max_horizontal_speed=params["max_horizontal_speed"],
        max_vertical_speed=params["max_vertical_speed"]
    )

    drones = [drone_1, drone_2]

    sim, world, A, B, custom_points_best, cost_gains, save_folder = optimize(params, world, drones)

    print("Executing simulation...")
    execute_simulation(
        sim, world, A, B, custom_points_best, cost_gains,
        showplots=True, interval=30, log_folder=save_folder,
        dt=0.1, print_info=True, save_log=True, print_log=False
    )

if __name__ == "__main__":
    main()