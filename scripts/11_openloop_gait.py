"""
11_openloop_gait.py — Open-loop periodic gait (CPG-style) that tries to walk.

Run:
    conda activate humanoid-sim
    python scripts/11_openloop_gait.py

What this teaches:
  - Walking can be approximated by sinusoidal joint trajectories
    (Central Pattern Generator idea from biology).
  - Phase offset pi between left/right legs creates alternating steps.
  - Arms swing counter-phase to the legs to cancel angular momentum.
  - PD on top of those trajectories keeps the joints tracking, with
    torque = gear * ctrl (so we divide the PD output by gear).

Model facts that constrain the gait (from models/humanoid.xml):
  - knee  range is [-160, -2] deg  -> "straight" is -2 deg, NOT 0.
  - hip_y range is [-110, 20] deg  -> +20 deg is the forward hard stop.
  - ctrlrange is [-0.4, 0.4] for ALL motors; gear varies 25..300.
  - So KP must be scaled per joint so a ~0.5 rad error does NOT saturate.

Expected behavior:
  With correct gains and ranges the gait walks a bit farther than the
  original broken version, but still falls after ~1.5-3 s: the model has
  NO ankle joints, so nothing can re-center the support point once drift
  accumulates. Open-loop = no feedback = guaranteed fall.
  RL adds the missing feedback loop and does 10x+ this distance.
"""
from pathlib import Path

import time

import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# ---------------------------------------------------------------------------
# Actuator bookkeeping: actuator index -> qpos address / qvel dof / gear.
# Use MuJoCo's own jnt_qposadr / jnt_dofadr instead of hand-rolled arithmetic.
# ---------------------------------------------------------------------------
ACT = {
    "abdomen_y": 0, "abdomen_z": 1, "abdomen_x": 2,
    "right_hip_x": 3, "right_hip_z": 4, "right_hip_y": 5, "right_knee": 6,
    "left_hip_x":  7, "left_hip_z":  8, "left_hip_y":  9, "left_knee":  10,
    "right_shoulder1": 11, "right_shoulder2": 12, "right_elbow": 13,
    "left_shoulder1":  14, "left_shoulder2":  15, "left_elbow":  16,
}

ACT_QPOS = np.zeros(model.nu, dtype=int)
ACT_DOF  = np.zeros(model.nu, dtype=int)
for i in range(model.nu):
    jid = int(model.actuator_trnid[i, 0])
    ACT_QPOS[i] = int(model.jnt_qposadr[jid])
    ACT_DOF[i]  = int(model.jnt_dofadr[jid])

GEAR     = np.array([model.actuator_gear[i, 0] for i in range(model.nu)])
CTRL_LO  = model.actuator_ctrlrange[:, 0].copy()
CTRL_HI  = model.actuator_ctrlrange[:, 1].copy()

# ---------------------------------------------------------------------------
# Gait parameters (ranges checked against humanoid.xml)
# ---------------------------------------------------------------------------
GAIT_FREQ   = 0.8          # Hz (one full gait cycle = 1/0.8 = 1.25 s)
STAND_KNEE  = -0.05        # rad; XML knee range is [-160, -2] deg -> -2 deg = -0.035
AMP_KNEE    =  0.50        # peak knee flex during forward swing
AMP_HIP_Y   =  0.30        # hip swing amplitude
AMP_ARM     =  0.35        # shoulder1 swing

# Per-joint PD gains with gravity compensation.
KP_DEFAULT = {
    "abdomen_y": 200.0, "abdomen_z": 200.0, "abdomen_x": 200.0,
    "right_hip_x": 200.0, "right_hip_z": 200.0, "right_hip_y": 300.0, "right_knee": 300.0,
    "left_hip_x":  200.0, "left_hip_z":  200.0, "left_hip_y":  300.0, "left_knee":  300.0,
    "right_shoulder1": 15.0, "right_shoulder2": 15.0, "right_elbow": 15.0,
    "left_shoulder1":  15.0, "left_shoulder2":  15.0, "left_elbow":  15.0,
}
KD_DEFAULT = {k: 0.15 * v for k, v in KP_DEFAULT.items()}


def pd_ctrl(name, target_q, target_dq=0.0):
    """Gravity-compensated PD (computed torque) for actuator `name`.

    Adds MuJoCo's qfrc_bias (gravity + Coriolis) on top of PD, so the PD only
    needs to handle tracking — not fight gravity.  This makes the sinusoidal
    gait targets actually visible in the joint motion, instead of the robot
    instantly collapsing under its own weight.
    """
    a = ACT[name]
    dof = ACT_DOF[a]
    err_p = target_q  - data.qpos[ACT_QPOS[a]]
    err_v = target_dq - data.qvel[dof]
    tau = KP_DEFAULT[name] * err_p + KD_DEFAULT[name] * err_v + data.qfrc_bias[dof]
    return tau / GEAR[a]


print("Script 11 — open-loop sinusoidal gait (CPG-style)")
print("=" * 60)
print(f"freq={GAIT_FREQ} Hz  amp: knee={AMP_KNEE} hip={AMP_HIP_Y} arm={AMP_ARM}")
print(f"stand_knee={STAND_KNEE}  ctrl range=[{CTRL_LO[0]:+.2f}, {CTRL_HI[0]:+.2f}]")
print("Expected: walks a bit, drifts, falls, resets. That is the point.\n")

last_print = -1.0
fall_count = 0
distances  = []
max_x_this_episode = 0.0

wall_start = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # ---- Real-time pacing: sleep if sim is ahead of wall clock --------
        sim_elapsed = data.time
        wall_elapsed = time.time() - wall_start
        if sim_elapsed > wall_elapsed:
            time.sleep(min(sim_elapsed - wall_elapsed, 0.05))

        t = data.time

        # ---- Update dynamics so qfrc_bias (gravity) is current ------------
        mujoco.mj_forward(model, data)

        # ---- Phase: right leg at phi, left leg at phi+pi ------------------
        phi_r = 2 * np.pi * GAIT_FREQ * t
        phi_l = phi_r + np.pi
        sin_r = np.sin(phi_r)
        sin_l = np.sin(phi_l)

        # ---- Leg targets --------------------------------------------------
        tgt_r_hip_y = -AMP_HIP_Y * sin_r
        tgt_l_hip_y = -AMP_HIP_Y * sin_l
        tgt_r_knee = STAND_KNEE - AMP_KNEE * max(0.0, -sin_r)
        tgt_l_knee = STAND_KNEE - AMP_KNEE * max(0.0, -sin_l)

        # ---- Arms swing counter-phase to the same-side leg ----------------
        tgt_r_arm = +AMP_ARM * sin_l
        tgt_l_arm = +AMP_ARM * sin_r

        # ---- Build ctrl vector -------------------------------------------
        ctrl = np.zeros(model.nu)
        ctrl[ACT["right_knee"]]      = pd_ctrl("right_knee",      tgt_r_knee)
        ctrl[ACT["left_knee"]]       = pd_ctrl("left_knee",       tgt_l_knee)
        ctrl[ACT["right_hip_y"]]     = pd_ctrl("right_hip_y",     tgt_r_hip_y)
        ctrl[ACT["left_hip_y"]]      = pd_ctrl("left_hip_y",      tgt_l_hip_y)
        ctrl[ACT["right_shoulder1"]] = pd_ctrl("right_shoulder1", tgt_r_arm)
        ctrl[ACT["left_shoulder1"]]  = pd_ctrl("left_shoulder1",  tgt_l_arm)
        for name in ("abdomen_y", "abdomen_z", "abdomen_x",
                     "right_hip_x", "right_hip_z",
                     "left_hip_x",  "left_hip_z",
                     "right_shoulder2", "right_elbow",
                     "left_shoulder2",  "left_elbow"):
            ctrl[ACT[name]] = pd_ctrl(name, 0.0)

        # Respect the model's own ctrl range (default [-0.4, 0.4]).
        data.ctrl[:] = np.clip(ctrl, CTRL_LO, CTRL_HI)

        mujoco.mj_step(model, data)

        # ---- Bookkeeping --------------------------------------------------
        max_x_this_episode = max(max_x_this_episode, float(data.qpos[0]))

        if data.qpos[2] < 0.5:
            fall_count += 1
            distances.append(max_x_this_episode)
            print(f"  FALL #{fall_count} at t={t:5.2f}s  "
                  f"walked {max_x_this_episode:+.2f} m  — resetting")
            mujoco.mj_resetData(model, data)
            data.ctrl[:] = 0.0
            max_x_this_episode = 0.0
            wall_start = time.time()
            last_print = data.time

        if t - last_print >= 0.5:
            print(f"  t={t:5.1f}s  z={data.qpos[2]:.3f}  x={data.qpos[0]:+.2f}m  "
                  f"r_knee_tgt={tgt_r_knee:+.2f}  falls={fall_count}")
            last_print = t

        viewer.sync()

print()
if distances:
    print(f"Distances before falls (m): {[round(d, 2) for d in distances]}")
    print(f"Average: {np.mean(distances):.2f} m  |  Best: {max(distances):.2f} m")
print()
print("Reading guide:")
print("  1. hip_y: full sine, opposite phases for L/R   -> alternates forward/back.")
print("  2. knee: bends ONLY during forward swing phase -> one leg lifts at a time.")
print("  3. Arms swing with OPPOSITE leg's phase        -> natural counter-balance.")
print("  4. ctrl = PD_torque / gear                     -> correct units.")
print("  5. OPEN-LOOP: no state feedback -> drift accumulates -> always falls.")
print("     This is exactly the gap RL closes: policy observes state and reacts.")