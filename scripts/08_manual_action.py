"""
08_manual_action.py — Hold a static pose with PD position control.

Current behavior: the robot simply STANDS STILL (all joint targets = 0).

Why the base is pinned (discovered by experiment):
  humanoid.xml has NO ankle joints, so the robot physically cannot balance —
  under pure PD it tips over in ~1.5 s no matter what pose is commanded.
  Freezing the floating base (an invisible suspension rig) lets us study
  controlled joint motion. Real balance is what scripts 09/10 / RL are for.

Control law:
    q_des  = TARGET_POSE (held constant, no cycling)
    torque = KP * (q_des - q_actual) - KD * q_velocity
    ctrl   = torque / gear            (because torque = gear * ctrl)

Run:
    D:/humanoid-sim/.venv/Scripts/python.exe scripts/08_manual_action.py

TO PLAY WITH (top of file):
    TARGET_POSE — which joints to bend, e.g. {"right_knee": -0.5, "left_knee": -0.5}
    RAMP_SEC    — seconds to ease from neutral into TARGET_POSE at startup
    PIN_BASE    — False to see the robot collapse without the rig
"""
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# ---- Tunable knobs ----------------------------------------------------------
TARGET_POSE = {}        # e.g. {"right_knee": -0.5, "left_knee": -0.5}
RAMP_SEC    = 1.5       # seconds to ease into TARGET_POSE (0 = instant).
KP          = 120.0     # PD proportional gain.
KD          = 12.0      # PD derivative gain.
PIN_BASE    = True      # freeze the floating base (suspension rig).
# -----------------------------------------------------------------------------

JOINTS = [
    "abdomen_y", "abdomen_z", "abdomen_x",
    "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
    "left_hip_x", "left_hip_z", "left_hip_y", "left_knee",
    "right_shoulder1", "right_shoulder2", "right_elbow",
    "left_shoulder1", "left_shoulder2", "left_elbow",
]

# Resolve joints and their qpos/qvel addresses by NAME.
jid  = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                 for n in JOINTS])
qadr = np.array([model.jnt_qposadr[i] for i in jid])
vadr = np.array([model.jnt_dofadr[i] for i in jid])

# gear lookup by NAME too -- do NOT rely on the list order matching the XML
# <actuator> order.  Build a joint_id -> gear map from actuator_trnid.
joint2gear = {}
for a in range(model.nu):
    j = int(model.actuator_trnid[a, 0])
    joint2gear[j] = model.actuator_gear[a, 0]
gear = np.array([joint2gear[i] for i in jid])

BASE_POS  = data.qpos[0:3].copy()
BASE_QUAT = data.qpos[3:7].copy()

q_target = np.zeros(len(JOINTS))
for name, val in TARGET_POSE.items():
    q_target[JOINTS.index(name)] = val

label = ", ".join(f"{k}={v}" for k, v in TARGET_POSE.items()) or "stand still (all zeros)"
print("Script 08 — static pose via PD position control")
print("=" * 60)
print(f"TARGET_POSE: {label}")
print(f"RAMP_SEC = {RAMP_SEC}   KP = {KP}   KD = {KD}   PIN_BASE = {PIN_BASE}")
if PIN_BASE:
    print("Base is pinned (numerical suspension rig, not a physical constraint):")
    print("without it the robot cannot balance (no ankle joints) and falls in ~1.5 s.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Ease from neutral into the target pose once, then hold it.
        alpha = min(data.time / max(RAMP_SEC, 1e-9), 1.0)
        q_des = alpha * q_target

        # PD controller: angle error -> torque -> ctrl.
        q_err  = q_des - data.qpos[qadr]
        torque = KP * q_err - KD * data.qvel[vadr]
        data.ctrl[:] = torque / gear

        # Suspension rig: zero base velocity BEFORE stepping so the physics
        # integrates from a still base each frame.
        if PIN_BASE:
            data.qvel[0:6] = 0.0

        mujoco.mj_step(model, data)

        # Then snap base pose back (kills any residual drift).
        if PIN_BASE:
            data.qpos[0:3] = BASE_POS
            data.qpos[3:7] = BASE_QUAT
            data.qvel[0:6] = 0.0

        viewer.sync()

print("Viewer closed.")
print()
print("Reading guide:")
print("  1. Standing still: PD holds every joint at its target angle.")
print("  2. Try TARGET_POSE = {'right_knee': -0.5, 'left_knee': -0.5} for a slow squat.")
print("  3. Set PIN_BASE = False to watch it collapse — why balance control (09/10)")
print("     and RL training are needed for real standing.")
print()
print("  NOTE: this 'rig' is a numerical hack, not a physical constraint.")
print("  A proper suspension would use <equality><weld> in humanoid.xml.")