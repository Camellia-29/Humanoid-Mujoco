"""
02_visualization.py — Open the interactive 3D viewer to see the humanoid.

Run:
    D:/humanoid-sim/.venv/Scripts/python.exe scripts/02_visualization.py

Viewer controls:
    Left-drag  = rotate camera    Scroll = zoom    Right-drag = pan
    Space      = pause/resume physics
    Ctrl + left-drag = apply a force to a body (try pulling its foot!)
"""
import os
from pathlib import Path

import mujoco
import mujoco.viewer

# Load the model from models/humanoid.xml using a path relative to THIS file,
# so it works no matter where the project lives on disk.
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "humanoid.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

print("Model info:")
print(f"  generalized coordinates nq = {model.nq}   # degrees of freedom")
print(f"  actuators nu              = {model.nu}  # controllable joints")
print(f"  physics timestep dt       = {model.opt.timestep} s")
print()
print(f"Loading model from: {MODEL_PATH}")
print("Opening 3D viewer... (close the window to exit)")

# launch() blocks until the user closes the viewer window.
mujoco.viewer.launch(model, data)