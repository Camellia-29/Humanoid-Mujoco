import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "g1.xml"

# === Gait parameters (tune these) ===
GAIT_FREQ = 1.5          # Hz, steps per second
AMP_HIP_P = 0.35         # hip pitch amplitude (rad)
AMP_HIP_Y = 0.15         # hip yaw amplitude (rad)
AMP_KNEE = 0.40          # knee amplitude (rad)
AMP_ANKLE_P = 0.15       # ankle pitch amplitude (rad)
AMP_ANKLE_R = 0.05       # ankle roll amplitude (rad)
AMP_WAIST_Y = 0.10       # waist yaw amplitude (rad)
AMP_ARM = 0.30           # arm swing amplitude (rad)
PHASE_OFFSET = np.pi     # left/right phase offset

# === Build actuator name -> index map ===
def build_act_map(model):
    act_map = {}
    for i in range(model.nu):
        name = model.actuator(i).name
        if isinstance(name, bytes):
            name = name.decode()
        act_map[name] = i
    return act_map

# === Get stand qpos from keyframe ===
def get_stand_qpos(model):
    for i in range(model.nkey):
        kn = model.key(i).name
        if isinstance(kn, bytes):
            kn = kn.decode()
        if "stand" in kn.lower():
            return model.key(i).qpos.copy()
    return model.key(0).qpos.copy()

# === Main ===
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
act_map = build_act_map(model)
stand_qpos = get_stand_qpos(model)
pelvis_id = model.body("pelvis").id

print("Actuator map keys:", sorted(act_map.keys()))
print(f"Stand qpos len={len(stand_qpos)}, nu={model.nu}")

# Verify critical actuators exist
for key in ["left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint", "right_knee_joint",
            "left_ankle_pitch_joint", "right_ankle_pitch_joint", "waist_yaw_joint",
            "left_shoulder_pitch_joint", "right_shoulder_pitch_joint"]:
    assert key in act_map, f"Missing actuator: {key}"

def compute_ctrl(t):
    phase = 2 * np.pi * GAIT_FREQ * t
    ctrl = np.zeros(model.nu)

    # Legs
    for side, sign in [("left", 1), ("right", -1)]:
        p = phase + (PHASE_OFFSET if sign < 0 else 0)
        hip_p = sign * AMP_HIP_P * np.sin(p)
        hip_y = AMP_HIP_Y * np.sin(p)
        knee = AMP_KNEE * (1 - np.cos(p)) * 0.5
        ankle_p = sign * AMP_ANKLE_P * np.sin(p)
        ankle_r = AMP_ANKLE_R * np.cos(p * 2)

        ctrl[act_map[f"{side}_hip_pitch_joint"]] = hip_p
        ctrl[act_map[f"{side}_hip_yaw_joint"]] = hip_y
        ctrl[act_map[f"{side}_knee_joint"]] = knee
        ctrl[act_map[f"{side}_ankle_pitch_joint"]] = ankle_p
        ctrl[act_map[f"{side}_ankle_roll_joint"]] = ankle_r

    # Waist
    ctrl[act_map["waist_yaw_joint"]] = AMP_WAIST_Y * np.sin(phase)

    # Arms (opposite to legs)
    for side, sign in [("left", 1), ("right", -1)]:
        p = phase + (PHASE_OFFSET if sign < 0 else 0) + np.pi
        shoulder_p = sign * AMP_ARM * np.sin(p)
        ctrl[act_map[f"{side}_shoulder_pitch_joint"]] = shoulder_p

    return ctrl

# Reset to stand
data.qpos[:] = stand_qpos
mujoco.mj_forward(model, data)

fallen = False
def check_fallen():
    global fallen
    up = np.array([0, 0, 1])
    z_axis = data.xmat[pelvis_id][6:9]
    if np.dot(z_axis, up) < 0.3:
        fallen = True

def reset():
    global fallen
    data.qpos[:] = stand_qpos
    data.qvel[:] = 0
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)
    fallen = False

t0 = time.time()
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 3.0
    while viewer.is_running():
        t = time.time() - t0
        if fallen:
            reset()
            t0 = time.time()
            t = 0

        data.ctrl[:] = compute_ctrl(t)
        mujoco.mj_step(model, data)
        check_fallen()

        with viewer.lock():
            viewer.cam.lookat[:] = data.xpos[pelvis_id]
            viewer.sync()

        time.sleep(model.opt.timestep)