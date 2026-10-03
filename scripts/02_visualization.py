"""
02_visualization.py — Open the interactive 3D viewer to see the humanoid.

Run:
    D:/humanoid-sim/.venv/Scripts/python.exe scripts/02_visualization.py

Viewer controls:
    Left-drag  = rotate camera    Scroll = zoom    Right-drag = pan
    Space      = pause/resume physics
    Backspace  = reset simulation
    Ctrl + left-drag = apply a force to a body (try pulling its foot!)
    [ / ]      = toggle geom groups (useful to see collision geoms)
"""
from pathlib import Path

import mujoco
import mujoco.viewer

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

print("Model info:")
print(f"  nq (qpos dim)         = {model.nq}  "
      f"(= 7 from free joint [3 pos + 4 quat] + 17 hinges)")
print(f"  nv (qvel dim)         = {model.nv}  "
      f"(= 6 from free joint [3 lin + 3 ang] + 17 hinges)")
print(f"  nu (actuators)        = {model.nu}  (17 motors, one per hinge here)")
print(f"  physics timestep dt   = {model.opt.timestep} s")
print()
print(f"Loading model from: {MODEL_PATH}")
print("NOTE: humanoid.xml has no <keyframe>, so Backspace resets to qpos0")
print("      (torso at 1.4 m, all joints 0).")
print("Opening 3D viewer... (close the window to exit)")

try:
    mujoco.viewer.launch(model, data)
except Exception as e:
    print(f"Failed to open viewer: {e}")
    print("On headless machines, use launch_passive + offscreen rendering.")