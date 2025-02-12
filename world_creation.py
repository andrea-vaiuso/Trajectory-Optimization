from Entity.World import World
from utility import show2DWorld

max_world_size = 1000
grid_size = 10
print("Creating world...")
world = World(grid_size=grid_size, max_world_size=max_world_size, world_name="Simple Testing World")

housing_estate = world.AREA_PARAMS[1]
industrial_area = world.AREA_PARAMS[2]
open_field = world.AREA_PARAMS[3]

# Initialize the area as open field
world.set_area_parameters(0, max_world_size, 0, max_world_size, open_field)
# Set the areas
world.set_area_parameters(320, 640, 40, 750, housing_estate)
world.set_area_parameters(360, 790, 780, 985, industrial_area)
print("Saving world...")
world.save_world("world_simple.pkl")
print("World created and saved as world.pkl")

print("Loading world...")
world = World.load_world("world_simple.pkl")
print("World checked and loaded from world.pkl")
show2DWorld(world, grid_size)