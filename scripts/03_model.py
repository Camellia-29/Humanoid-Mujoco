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

# Map enum -> human-readable name.  Use symbolic constants instead of raw ints,
# so a future MuJoCo version that reorders mjtJoint does not silently break us.
JOINT_TYPE_NAMES = {
    mujoco.mjtJoint.mjJNT_FREE:  "free(6DOF)",
    mujoco.mjtJoint.mjJNT_BALL:  "ball(3DOF)",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide(1DOF)",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge(1DOF)",
}


def name_of(obj_type, idx):
    return mujoco.mj_id2name(model, obj_type, idx) or "<unnamed>"


print("=" * 60)
print("1. Body tree (bodies)")
print("=" * 60)
for i in range(model.nbody):
    name = name_of(mujoco.mjtObj.mjOBJ_BODY, i)
    parent = model.body_parentid[i]
    parent_name = "—" if parent == i else name_of(mujoco.mjtObj.mjOBJ_BODY, parent)
    mass = model.body_mass[i]
    print(f"  [{i:2d}] {name:20s} parent: {parent_name:12s} mass: {mass:.3f} kg")

print()
print("=" * 60)
print("2. Joints — where the robot can move")
print("=" * 60)
for i in range(model.njnt):
    name = name_of(mujoco.mjtObj.mjOBJ_JOINT, i)
    jtype_name = JOINT_TYPE_NAMES.get(model.jnt_type[i], "unknown")
    print(f"  [{i:2d}] {name:25s} type: {jtype_name}")

print()
print("=" * 60)
print("3. Actuators — motors, what you can control")
print("=" * 60)
print(f"  {'idx':>3}  {'actuator':20s}  {'joint':20s}  {'gear':>6}  "
      f"{'ctrl range':>14}  {'torque range':>18}")
for i in range(model.nu):
    name = name_of(mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    jid = int(model.actuator_trnid[i, 0])
    jname = name_of(mujoco.mjtObj.mjOBJ_JOINT, jid)
    gear = model.actuator_gear[i, 0]
    lo, hi = model.actuator_ctrlrange[i]
    # torque = gear * ctrl, so the *physical* torque range is much bigger
    # than the ctrl range -- this is the key fact behind script 11's /gear.
    print(f"  {i:3d}  {name:20s}  {jname:20s}  {gear:6.0f}  "
          f"[{lo:+.2f}, {hi:+.2f}]  "
          f"[{lo*gear:+.1f}, {hi*gear:+.1f}] N·m")

print()
print("=" * 60)
print("4. Whole-robot parameters")
print("=" * 60)
total_mass = model.body_mass[1:].sum()   # skip world body
print(f"  total mass:            {total_mass:.2f} kg")
print(f"  nq (qpos dim):         {model.nq}  "
      f"(= 7 from free joint [3 pos + 4 quat] + {model.nq - 7} hinge)")
print(f"  nv (qvel dim):         {model.nv}  "
      f"(= 6 from free joint [3 lin + 3 ang] + {model.nv - 6} hinge)")
print(f"  nu (actuators):        {model.nu}")
print(f"  physics timestep:      {model.opt.timestep * 1000:.1f} ms  "
      f"({int(round(1 / model.opt.timestep))} steps/s)")

print()
print("Next step: open models/humanoid.xml side-by-side with this output and")
print("trace the <body> <joint> <actuator> three-layer relationship.")
print("NOTE: changing mass or inertia will invalidate PD gains in scripts 08-11.")