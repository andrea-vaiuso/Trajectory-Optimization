import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pickle
from Entity.World import World

class WorldEditor:
    def __init__(self, root, world):
        self.root = root
        self.world = world
        self.selected_area = None
        self.rectangles = []
        self.start_x = None
        self.start_y = None
        self.undo_stack = []

        self.canvas = tk.Canvas(root, width=self.world.max_world_size, height=self.world.max_world_size)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.load_background()
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("<Control-z>", self.undo_last_action)
        
        self.area_var = tk.StringVar(root)
        area_names = {v['name']: k for k, v in world.AREA_PARAMS.items()}
        self.area_var.set(next(iter(area_names)))
        
        button_frame = tk.Frame(root)
        button_frame.pack()

        self.area_menu = tk.OptionMenu(button_frame, self.area_var, *area_names.keys())
        self.area_menu.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.set_full_area_button = tk.Button(button_frame, text="Set Full Area", command=self.set_full_area)
        self.set_full_area_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.undo_button = tk.Button(button_frame, text="Undo", command=self.undo_last_action)
        self.undo_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.save_button = tk.Button(button_frame, text="Save World", command=self.save_world)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)
    
    def load_background(self):
        if self.world.background_image is not None:
            image = Image.fromarray(self.world.background_image)
            image = image.resize((self.world.max_world_size, self.world.max_world_size))
            self.bg_image = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.bg_image)
    
    def on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        area_id = self.get_selected_area_id()
        color = self.world.AREA_PARAMS[area_id]["color"]
        self.rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline=color, fill=color, stipple="gray50")
    
    def on_drag(self, event):
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)
    
    def on_release(self, event):
        x1, y1, x2, y2 = self.start_x, self.start_y, event.x, event.y
        area_id = self.get_selected_area_id()
        self.rectangles.append((x1, y1, x2, y2, area_id, self.rect_id))
        self.undo_stack.append(self.rect_id)
    
    def get_selected_area_id(self):
        area_name = self.area_var.get()
        area_id = next((k for k, v in self.world.AREA_PARAMS.items() if v['name'] == area_name), None)
        return area_id
    
    def set_full_area(self):
        area_id = self.get_selected_area_id()
        self.world.set_area_parameters(0, self.world.max_world_size, 0, self.world.max_world_size, self.world.AREA_PARAMS[area_id])
        rect_id = self.canvas.create_rectangle(0, 0, self.world.max_world_size, self.world.max_world_size, outline=self.world.AREA_PARAMS[area_id]["color"], fill=self.world.AREA_PARAMS[area_id]["color"], stipple="gray50")
        self.undo_stack.append(rect_id)
    
    def undo_last_action(self, event=None):
        if self.undo_stack:
            rect_id = self.undo_stack.pop()
            self.canvas.delete(rect_id)
            self.rectangles = [r for r in self.rectangles if r[-1] != rect_id]
    
    def save_world(self):
        world_name = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle Files", "*.pkl")], title="Save World As")
        if not world_name:
            return
        for rect in self.rectangles:
            x1, y1, x2, y2, area_id, _ = rect
            self.world.set_area_parameters(x1, x2, y1, y2, self.world.AREA_PARAMS[area_id])
        self.world.save_world(world_name)
        messagebox.showinfo("Saved", "World saved successfully!")

if __name__ == "__main__":
    max_world_size = 1000
    grid_size = 10
    root = tk.Tk()
    root.title("World Editor")
    
    image_path = filedialog.askopenfilename(title="Select Background Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not image_path:
        messagebox.showerror("Error", "No image selected. Exiting...")
        root.destroy()
    else:
        world = World(grid_size=grid_size, max_world_size=max_world_size, world_name="Winterthur Area", background_image_path=image_path)
        app = WorldEditor(root, world)
        root.mainloop()
