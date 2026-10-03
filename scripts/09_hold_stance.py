"""
09_hold_stance.py — Why zero action does NOT keep the robot standing.

Run:
    D:/humanoid-sim/.venv/Scripts/python.exe scripts/09_hold_stance.py

What this teaches:
  - ctrl = 0 means zero motor torque. Gravity alone wins.
  - A small constant torque on the knees (a "squat hold") delays the fall
    but still cannot stabilize the robot.
  - The conclusion: feedback (PD, script 10) or learned control (RL, Phase 5)
    is necessary to maintain balance.

Three demos are run in sequence inside one viewer session.
"""
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

ACT = {
    "abdomen_y": 0, "abdomen_z": 1, "abdomen_x": 2,
    "right_hip_x": 3, "right_hip_z": 4, "right_hip_y": 5, "right_knee": 6,
    "left_hip_x":  7, "left_hip_z":  8, "left_hip_y":  9, "left_knee":  10,
    "right_shoulder1": 11, "right_shoulder2": 12, "right_elbow": 13,
    "left_shoulder1":  14, "left_shoulder2":  15, "left_elbow":  16,
}

DT = model.opt.timestep
FALL_Z = 0.7   # below this torso z, we call it "fell"


def run_demo(name, ctrl_fn, duration, viewer):
    """Run a single demo inside the viewer. Returns time of fall (or duration)."""
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    print(f"\n--- {name} ---")
    print(f"  {'t':>5}  {'torso z':>8}")
    t_fall = None
    last_print_t = -1.0
    n_steps = int(duration / DT)
    for _ in range(n_steps):
        data.ctrl[:] = ctrl_fn(data.time)
        mujoco.mj_step(model, data)
        if data.qpos[2] < FALL_Z and t_fall is None:
            t_fall = data.time
        if data.time - last_print_t >= 0.3:
            print(f"  {data.time:5.2f}  {data.qpos[2]:8.3f}")
            last_print_t = data.time
        viewer.sync()
    if t_fall is None:
        t_fall = duration
    verdict = "FELL" if t_fall < duration else "survived"
    print(f"  -> {verdict} at t = {t_fall:.2f}s")
    return t_fall


def zero_ctrl(t):
    return np.zeros(model.nu)


def knee_ctrl(value):
    def fn(t):
        c = np.zeros(model.nu)
        c[ACT["right_knee"]] = value
        c[ACT["left_knee"]]  = value
        return c
    return fn


print("Script 09 — Why ctrl=0 does not stand the robot up")
print("=" * 60)
print("Watch the viewer; three demos in sequence.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    t1 = run_demo("Demo A: ctrl = all zeros  (gravity alone)",
                  zero_ctrl, 3.0, viewer)
    t2 = run_demo("Demo B: ctrl = -0.1 on knees (mild squat)",
                  knee_ctrl(-0.1), 5.0, viewer)
    t3 = run_demo("Demo C: ctrl = -0.3 on knees (deep squat)",
                  knee_ctrl(-0.3), 5.0, viewer)

print()
print("=" * 60)
print(f"Summary:  zero       lasted {t1:.2f}s  "
      f"({'fell' if t1 < 3.0 else 'survived'})")
print(f"          knee -0.1  lasted {t2:.2f}s  "
      f"({'fell' if t2 < 5.0 else 'survived'})")
print(f"          knee -0.3  lasted {t3:.2f}s  "
      f"({'fell' if t3 < 5.0 else 'survived'})")
print()
print("Reading guide:")
print("  Demo A: no torque -> gravity is unopposed -> falls in <0.5 s")
print("  Demo B/C: constant knee torque pre-loads the joints. Depending on the")
print("            value, it may delay OR accelerate the fall -- but with no")
print("            feedback, it can never recover from a disturbance.")
print()
print("  Next: script 10_pd_controller.py adds the missing FEEDBACK term.")