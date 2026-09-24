"""
03_model.py — Dissect humanoid.xml and see how the humanoid is built.

Run:
    D:/humanoid-sim/.venv/Scripts/python.exe scripts/03_model.py

If you want to "build your own simulation environment", the first step is
understanding the official model's structure. Open models/humanoid.xml in any
text editor and read along with this script's output.
"""
from pathlib import Path

import mujoco

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

JOINT_TYPE_NAMES = ["free(6DOF)", "ball(3DOF)", "slide(1DOF)", "hinge(1DOF)"]

print("=" * 60)
print("1. Body tree (bodies)")
print("=" * 60)
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    parent = model.body_parentid[i]
    parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent)
    mass = model.body_mass[i]
    print(f"  [{i:2d}] {name:20s} parent: {str(parent_name):12s} mass: {mass:.1f} kg")

print()
print("=" * 60)
print("2. Joints — where the robot can move")
print("=" * 60)
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jtype = int(model.jnt_type[i])  # 0=free, 1=ball, 2=slide, 3=hinge
    jtype_name = JOINT_TYPE_NAMES[jtype]
    print(f"  [{i:2d}] {name:25s} type: {jtype_name}")

print()
print("=" * 60)
print("3. Actuators — motors, what you can control")
print("=" * 60)
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    lo, hi = model.actuator_ctrlrange[i]
    print(f"  [{i:2d}] {name:25s} ctrl range: [{lo:.1f}, {hi:.1f}]")

print()
print("=" * 60)
print("4. Whole-robot parameters")
print("=" * 60)
total_mass = sum(model.body_mass)
print(f"  total mass:           {total_mass:.1f} kg")
print(f"  DoF nq:               {model.nq}  (= 7 torso[3 pos + 4 quat] + 17 joints)")
print(f"  control dimensions nu:{model.nu}")
print(f"  physics timestep:     {model.opt.timestep * 1000:.1f} ms "
      f"(= {int(1 / model.opt.timestep)} physics steps per second)")

print()
print("Next step: open models/humanoid.xml side-by-side with this output and")
print("trace the <body> <joint> <actuator> three-layer relationship. Try editing")
print("    a parameter (e.g. reduce torso mass) and re-run this script to see the change.")