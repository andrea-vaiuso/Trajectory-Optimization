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

    print("Creating drone...")
    drone = Drone(
        model_name=params["drone_model_name"],
        x=params["A"]["x"],
        y=params["A"]["y"],
        z=params["A"]["z"],
        min_RPM=params["min_RPM"],
        max_RPM=params["max_RPM"],
        hover_RPM=params["hover_RPM"],
        max_horizontal_speed=params["max_horizontal_speed"],
        max_vertical_speed=params["max_vertical_speed"]
    )

    sim, world, A, B, custom_points_best, cost_gains, save_folder = optimize(params, world, drone)

    print("Executing simulation...")
    execute_simulation(
        sim, world, A, B, custom_points_best, cost_gains,
        showplots=True, interval=30, log_folder=save_folder,
        dt=0.1, print_info=True, save_log=True, print_log=False
    )

if __name__ == "__main__":
    main()