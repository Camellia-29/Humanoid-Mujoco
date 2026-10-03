"""
10_pd_controller.py — Classical PD controller and its LIMIT on a real humanoid.

Run:
    D:/humanoid-sim/.venv/Scripts/python.exe scripts/10_pd_controller.py

What this teaches:
  - PD control law:   tau = Kp * (q_target - q)  -  Kd * dq
  - PD clearly improves over zero-action (zero falls in <0.5 s, PD holds >1 s).
  - BUT simple joint-level PD cannot indefinitely stabilize this 3D biped,
    because it regulates JOINT ANGLES, not CENTER OF MASS.

This is exactly the gap that RL must fill: it learns a STATE-DEPENDENT
controller that actively regulates CoM, anticipates falls, and recovers.
"""
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# Actuator name -> index (look up by name; do not hard-code).
ACT_NAME = [
    "abdomen_y", "abdomen_z", "abdomen_x",
    "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
    "left_hip_x", "left_hip_z", "left_hip_y", "left_knee",
    "right_shoulder1", "right_shoulder2", "right_elbow",
    "left_shoulder1", "left_shoulder2", "left_elbow",
]
ACT = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
       for n in ACT_NAME}

# Per-actuator: (qpos address, qvel address, gear).
ACT_QPOS = np.zeros(model.nu, dtype=int)
ACT_DOF  = np.zeros(model.nu, dtype=int)
GEAR     = np.zeros(model.nu)
for i in range(model.nu):
    jid = int(model.actuator_trnid[i, 0])
    ACT_QPOS[i] = int(model.jnt_qposadr[jid])
    ACT_DOF[i]  = int(model.jnt_dofadr[jid])
    GEAR[i]     = model.actuator_gear[i, 0]

# Per-joint PD gains.  Rule of thumb: KP * err_max / gear < ctrl_max so we
# don't sit in saturation for a reasonable error (~0.5 rad).
#   knee     gear=200 -> KP=120
#   hip_y    gear=300 -> KP=200
#   hip_x/z  gear=100 -> KP= 60
#   abdomen  gear=100 -> KP= 60
#   arm      gear= 25 -> KP= 15
KP_DEFAULT = {
    "abdomen_y": 60.0, "abdomen_z": 60.0, "abdomen_x": 60.0,
    "right_hip_x": 60.0, "right_hip_z": 60.0, "right_hip_y": 200.0, "right_knee": 120.0,
    "left_hip_x":  60.0, "left_hip_z":  60.0, "left_hip_y":  200.0, "left_knee":  120.0,
    "right_shoulder1": 15.0, "right_shoulder2": 15.0, "right_elbow": 15.0,
    "left_shoulder1":  15.0, "left_shoulder2":  15.0, "left_elbow":  15.0,
}
KP = np.array([KP_DEFAULT[n] for n in ACT_NAME])
KD = 0.1 * KP

# Target pose: all joints neutral, with a small knee crouch.
TARGET_QPOS_RAW = {n: 0.0 for n in ACT_NAME}
TARGET_QPOS_RAW["right_knee"] = -0.05
TARGET_QPOS_RAW["left_knee"]  = -0.05
TARGET_QPOS = np.array([TARGET_QPOS_RAW[n] for n in ACT_NAME])

DT = model.opt.timestep
CTRL_LO = model.actuator_ctrlrange[:, 0]
CTRL_HI = model.actuator_ctrlrange[:, 1]

print("Script 10 — Classical PD controller (and its honest limit)")
print("=" * 60)
print(f"dt = {DT}s, per-joint KP (knee/hip_y/hip_xz/arm): "
      f"{KP[ACT['right_knee']]:.0f}/"
      f"{KP[ACT['right_hip_y']]:.0f}/"
      f"{KP[ACT['right_hip_x']]:.0f}/"
      f"{KP[ACT['right_shoulder1']]:.0f}")
print("Target: slight knee crouch (no arm targets).")
print("EXPECTATION: PD delays the fall but the robot WILL tip over after ~1.5 s,")
print("because joint-level PD does not regulate the center of mass.")
print("Compare with script 09 (zero action falls in <0.5 s).")
print("Close the viewer to stop.\n")


def pd_ctrl():
    """Compute ctrl for every actuator: ctrl = tau / gear, then clamp."""
    q  = data.qpos[ACT_QPOS]
    dq = data.qvel[ACT_DOF]
    tau = KP * (TARGET_QPOS - q) - KD * dq
    ctrl = tau / GEAR
    return np.clip(ctrl, CTRL_LO, CTRL_HI)


last_print = -1.0
upright_time_total = 0.0
fall_count = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[:] = pd_ctrl()
        mujoco.mj_step(model, data)

        if data.qpos[2] > 1.0:
            upright_time_total += DT

        if data.qpos[2] < 0.5:
            fall_count += 1
            print(f"  FALL #{fall_count} at t={data.time:.2f}s "
                  f"(upright {upright_time_total:.2f}s)")
            mujoco.mj_resetData(model, data)
            data.ctrl[:] = 0.0
            upright_time_total = 0.0
            last_print = data.time

        if data.time - last_print >= 0.5:
            print(f"  t={data.time:5.1f}s  z={data.qpos[2]:.3f}  "
                  f"|ctrl|={np.linalg.norm(data.ctrl):5.2f}  "
                  f"|qvel|={np.linalg.norm(data.qvel):5.2f}  falls={fall_count}")
            last_print = data.time

        viewer.sync()

print()
print("Reading guide:")
print("  Result vs script 09:")
print("    09 (zero action):         fell in <0.5 s")
print("    10 (per-joint PD):        held ~1.5-2 s per attempt, then tipped")
print()
print("  Why PD isn't enough:")
print("    Joint-level PD regulates INDIVIDUAL JOINTS, not the CENTER OF MASS.")
print("    Small perturbations grow, and the robot slowly tips. Whole-body")
print("    control would need to track CoM, apply ground reaction forces via")
print("    foot placement, and anticipate fall direction.")
print()
print("  RL learns exactly this kind of CoM-aware, state-dependent control.")
print("  The policy is the SAME interface (17 ctrl outputs), but it sees the")
print("  FULL 348-dim state including CoM-related features.")