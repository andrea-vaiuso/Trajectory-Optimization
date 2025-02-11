from World import World


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

max_world_size = 1000
grid_size = 10
print("Creating world...")
world = World(grid_size=grid_size, max_world_size=max_world_size)

# Initialize the area as open field
world.set_area_parameters(0, max_world_size, 0, max_world_size, open_field)
# Set the areas
world.set_area_parameters(320, 640, 40, 750, housing_estate)
world.set_area_parameters(360, 790, 780, 985, industrial_area)
print("Saving world...")
world.save_world("world.pkl")
print("World created and saved as world.pkl")

print("Loading world...")
world = World.load_world("world.pkl")
print("World checked and loaded from world.pkl")