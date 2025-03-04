import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess

def compress_gif(input_path, output_path, optimize_level, lossy_value):
    """
    Compress the GIF using gifsicle.
    
    Parameters:
        input_path (str): Path to the input GIF file.
        output_path (str): Path to save the compressed GIF.
        optimize_level (int): Optimization level (1, 2, or 3).
        lossy_value (int): Lossy compression parameter (0 for no lossy, higher for more lossy).
    """
    try:
        # Build the gifsicle command.
        # The --optimize option performs various optimizations.
        # If lossy_value > 0, then --lossy applies a lossy compression algorithm.
        cmd = ["gifsicle", f"--optimize={optimize_level}"]
        if lossy_value > 0:
            cmd.append(f"--lossy={lossy_value}")
        cmd.extend([input_path, "-o", output_path])
        
        # Run the command.
        subprocess.run(cmd, check=True)
        messagebox.showinfo("Success", "GIF compressed and saved successfully.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Compression failed: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def open_file():
    """Open a file dialog to select a GIF file."""
    file_path = filedialog.askopenfilename(
        title="Select a GIF file",
        filetypes=[("GIF files", "*.gif")]
    )
    if file_path:
        file_label.config(text=file_path)
    return file_path

def compress_action():
    """Retrieve parameters from the GUI, then compress the selected GIF."""
    input_path = file_label.cget("text")
    if not input_path or not input_path.lower().endswith(".gif"):
        messagebox.showerror("Error", "Please select a valid GIF file.")
        return

    try:
        optimize_level = int(optimize_slider.get())
        lossy_value = int(lossy_slider.get())
    except ValueError:
        messagebox.showerror("Error", "Invalid compression values.")
        return
    
    output_path = filedialog.asksaveasfilename(
        defaultextension=".gif",
        filetypes=[("GIF files", "*.gif")],
        title="Save compressed GIF as"
    )
    if not output_path:
        return

    compress_gif(input_path, output_path, optimize_level, lossy_value)

# Build the GUI.
root = tk.Tk()
root.title("GIF Compressor")

# Row 0: Button to open a GIF file and a label to display its path.
open_button = tk.Button(root, text="Open GIF", command=open_file)
open_button.grid(row=0, column=0, padx=10, pady=10)

file_label = tk.Label(root, text="No file selected", wraplength=300)
file_label.grid(row=0, column=1, padx=10, pady=10)

# Row 1: Slider for selecting the optimization level.
optimize_label = tk.Label(root, text="Optimization Level (1 = Low, 3 = High)")
optimize_label.grid(row=1, column=0, padx=10, pady=10)
optimize_slider = tk.Scale(root, from_=1, to=3, orient=tk.HORIZONTAL)
optimize_slider.set(3)  # Default to highest optimization.
optimize_slider.grid(row=1, column=1, padx=10, pady=10)

# Row 2: Slider for selecting the lossy compression value.
lossy_label = tk.Label(root, text="Lossy Compression (0 = none, up to 100)")
lossy_label.grid(row=2, column=0, padx=10, pady=10)
lossy_slider = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL)
lossy_slider.set(0)  # Default to no lossy compression.
lossy_slider.grid(row=2, column=1, padx=10, pady=10)

# Row 3: Button to perform the compression.
compress_button = tk.Button(root, text="Compress GIF", command=compress_action)
compress_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()
