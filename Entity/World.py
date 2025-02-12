import json
import pickle
import numpy as np
from PIL import Image

# ---------------- World Class ----------------
class World:
    # Dizionario statico per mappare l'ID ai parametri dell'area
    AREA_PARAMS = {
        1: {
            "id": 1,
            "name": "Housing Estate",
            "min_altitude": 150, 
            "max_altitude": 1000, 
            "noise_penalty": 2,
            "color": "blue",
            "alpha": 0.2
        },
        2: {
            "id": 2,
            "name": "Industrial Area",
            "min_altitude": 70, 
            "max_altitude": 1000, 
            "noise_penalty": 1,
            "color": "yellow",
            "alpha": 0.2
        },
        3: {
            "id": 3,
            "name": "Open Field",
            "min_altitude": 0, 
            "max_altitude": 1000, 
            "noise_penalty": 0,
            "color": "green",
            "alpha": 0.1
        }
    }
    
    # Imposta l'ID di default (ad esempio, Open Field)
    DEFAULT_AREA_ID = 3

    def __init__(self, grid_size, max_world_size, world_name="World", background_image_path=None):
        self.grid_size = grid_size
        self.max_world_size = max_world_size
        self.grid = {}  # mapping: area coordinate tuple -> area_id
        self.world_name = world_name
        self.background_image = None
        if background_image_path:
            # Check if the background is squared and save it
            self.background_image = np.array(Image.open(background_image_path).convert('RGB'))
            if self.background_image.shape[0] != self.background_image.shape[1]:
                print("Warning: the background image is not squared. Cropping it...")
                min_dim = min(self.background_image.shape[0], self.background_image.shape[1])
                self.background_image = self.background_image[:min_dim, :min_dim]

    def get_area(self, x, y, z):
        return (x // self.grid_size, y // self.grid_size, z // self.grid_size)

    def set_area_parameters(self, x_1, x_2, y_1, y_2, parameters):
        """
        Imposta l'ID dell'area nelle coordinate specificate.
        I parametri vengono presi dal dizionario 'parameters', che deve contenere almeno la chiave "id".
        """
        area_id = parameters["id"]
        for x in range(x_1, x_2 + 1, self.grid_size):
            for y in range(y_1, y_2 + 1, self.grid_size):
                for z in range(0, self.max_world_size * self.grid_size, self.grid_size):
                    area = self.get_area(x, y, z)
                    self.grid[area] = area_id

    def get_area_parameters(self, x, y, z):
        area = self.get_area(x, y, z)
        area_id = self.grid.get(area, World.DEFAULT_AREA_ID)
        return World.AREA_PARAMS.get(area_id, {})

    def get_area_center_point(self, x, y, z):
        area = self.get_area(x, y, z)
        return ((area[0] + 0.5) * self.grid_size, 
                (area[1] + 0.5) * self.grid_size, 
                (area[2] + 0.5) * self.grid_size)

    def get_areas_in_circle(self, x, y, height, radius):
        areas_in_circle = []
        parameters_in_circle = []
        radius_squared = radius ** 2
        
        # Limiti del grid
        min_x = max(0, x - radius)
        max_x = min(self.max_world_size * self.grid_size, x + radius)
        min_y = max(0, y - radius)
        max_y = min(self.max_world_size * self.grid_size, y + radius)

        x_range = np.arange(min_x, max_x + 1, self.grid_size)
        y_range = np.arange(min_y, max_y + 1, self.grid_size)
        z_range = np.arange(0, height * self.grid_size, self.grid_size)

        x_mesh, y_mesh = np.meshgrid(x_range, y_range, indexing='ij')
        x_flat, y_flat = x_mesh.ravel(), y_mesh.ravel()

        distances_sq = (x_flat - x) ** 2 + (y_flat - y) ** 2
        valid_indices = np.where(distances_sq <= radius_squared)[0]

        for idx in valid_indices:
            i, j = x_flat[idx], y_flat[idx]
            for z in z_range:
                area_center = self.get_area_center_point(i, j, z)
                area_params = self.get_area_parameters(i, j, z)
                areas_in_circle.append(area_center)
                parameters_in_circle.append(area_params)

        return areas_in_circle, parameters_in_circle

    def save_world(self, filename):
        data = {
            'grid_size': self.grid_size,
            'max_world_size': self.max_world_size,
            'grid': self.grid,
            'world_name': self.world_name,
            'background_image': self.background_image
        }
        with open(filename, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_world(cls, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        world = cls(data['grid_size'], data['max_world_size'], data['world_name'])
        world.grid = data['grid']
        if data['background_image'] is not None:
            world.background_image = data['background_image']
        return world