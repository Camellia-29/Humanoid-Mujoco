"""
05_force_breakdown.py — See where the forces at each step come from.

Run:
    D:/humanoid-sim/.venv/Scripts/python.exe scripts/05_force_breakdown.py

What this teaches:
  - That the simulation is just a vector equation of motion:
        M(q) qdd + c(q, qd) = qfrc_actuator + qfrc_passive + J^T * cfrc_ext
  - That cfrc_ext only has entries for bodies currently in contact.
"""
import numpy as np
import gymnasium as gym
import mujoco

env = gym.make("Humanoid-v5")
obs, info = env.reset(seed=42)
m = env.unwrapped.model
d = env.unwrapped.data

# Index lookups for the 13 humanoid bodies (skipping worldbody[0]).
FOOT_BODIES = ["right_foot", "left_foot"]
FOOT_IDS = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in FOOT_BODIES]

print(f"{'step':>3}  {'actF.sum':>9}  {'passF.sum':>10}  {'contact.sum':>11}  "
      f"{'rfoot |F|':>10}  {'lfoot |F|':>10}")
print("-" * 70)

for step in range(10):
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)

    # Three generalized-force terms in MuJoCo's equation of motion.
    qfrc_act    = d.qfrc_actuator      # from motors        (nq,)
    qfrc_passive = d.qfrc_passive       # gravity + spring + damping (nq,)
    cfrc_ext    = d.cfrc_ext           # external/contact forces per body (nbody, 6)

    # Per-foot contact force magnitude (translational part).
    rfoot_F = np.linalg.norm(cfrc_ext[FOOT_IDS[0], :3])
    lfoot_F = np.linalg.norm(cfrc_ext[FOOT_IDS[1], :3])

    print(f"{step:3d}  {qfrc_act.sum():+9.3f}  {qfrc_passive.sum():+10.3f}  "
          f"{np.abs(cfrc_ext[1:]).sum():11.3f}  "
          f"{rfoot_F:10.3f}  {lfoot_F:10.3f}")

    if term or trunc:
        env.reset()

env.close()
print()
print("Reading guide:")
print("  qfrc_actuator[i]   = motor torque on DoF i   (after gear ratio)")
print("  qfrc_passive[i]    = gravity + spring + damping on DoF i")
print("  cfrc_ext[b, 0..2]  = contact force on body b (x,y,z)")
print("  cfrc_ext[b, 3..5]  = contact torque on body b")
print("  Foot forces > 0 when standing/walking; ≈ 0 when robot is airborne.")
print()
print("  Net acceleration satisfies:  M(q) qdd = qfrc_act + qfrc_passive + J^T cfrc_ext")