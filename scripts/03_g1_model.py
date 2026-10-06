import mujoco
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "g1.xml"
print(f"Loading G1 model from: {MODEL_PATH}")

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

print("\n=== Model Info ===")
print(f"nq (positions): {model.nq}")
print(f"nv (velocities): {model.nv}")
print(f"nu (actuators): {model.nu}")
print(f"nbody: {model.nbody}")
print(f"njnt: {model.njnt}")
print(f"nkey: {model.nkey}")

# Body hierarchy
print("\n=== Body Hierarchy ===")
for i in range(model.nbody):
    name = model.body(i).name
    if isinstance(name, bytes):
        name = name.decode()
    parent = model.body(i).parentid
    print(f"  [{i}] {name} (parent={parent})")

# Joints
print("\n=== Joints ===")
for i in range(model.njnt):
    jnt = model.joint(i)
    name = jnt.name
    if isinstance(name, bytes):
        name = name.decode()

    jnt_type = ["free", "ball", "slide", "hinge"][int(model.jnt_type[i])]
    qpos_adr = model.jnt_qposadr[i]

    if model.jnt_type[i] == 0:
        nq = 7
    elif model.jnt_type[i] == 1:
        nq = 4
    else:
        nq = 1

    if nq > 0:
        range_str = str(model.jnt_range[i].tolist())
    else:
        range_str = "N/A"

    print(f"  [{i}] {name}: type={jnt_type}, qposadr={qpos_adr}, nq={nq}, range={range_str}")

# Actuators
print("\n=== Actuators ===")
for i in range(model.nu):
    act = model.actuator(i)
    name = act.name
    if isinstance(name, bytes):
        name = name.decode()

    act_type = ["motor", "position", "velocity"][int(model.actuator_ctrllimited[i])]
    gear = model.actuator_gear[i, 0]
    ctrlrange = model.actuator_ctrlrange[i]
    gainprm = model.actuator_gainprm[i, :3]

    print(f"  [{i}] {name}: type={act_type}, gear={gear:.1f}, ctrlrange={ctrlrange.tolist()}, gainprm={gainprm.tolist()}")

# Keyframes
print("\n=== Keyframes ===")
for i in range(model.nkey):
    kname = model.key(i).name
    if isinstance(kname, bytes):
        kname = kname.decode()
    print(f"  [{i}] {kname}")
    try:
        qpos = model.key(i).qpos
        print(f"      qpos[:10] = {qpos[:10].tolist()}")
    except Exception:
        try:
            qpos = model.keyframe_qpos[i]
            print(f"      qpos[:10] = {qpos[:10].tolist()}")
        except Exception:
            print(f"      (could not read qpos)")

# Stand pose
print("\n=== Stand Pose (qpos) ===")
try:
    kf_idx = -1
    for i in range(model.nkey):
        kn = model.key(i).name
        if isinstance(kn, bytes):
            kn = kn.decode()
        if "stand" in kn.lower():
            kf_idx = i
            break
    if kf_idx >= 0:
        stand_qpos = model.key(kf_idx).qpos
    else:
        stand_qpos = model.key(0).qpos
    print(f"qpos = {stand_qpos.tolist()}")
    print(f"len(qpos) = {len(stand_qpos)}")
except Exception as e:
    print(f"Error reading keyframe: {e}")

print("\nDone.")