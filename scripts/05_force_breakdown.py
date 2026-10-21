"""
05_force_breakdown.py — See where the forces at each step come from.

Run:
    D:/humanoid-sim/.venv/Scripts/python.exe scripts/05_force_breakdown.py

What this teaches:
  - That the simulation is just a vector equation of motion:
        M(q) qdd + c(q, qd) = qfrc_actuator + qfrc_passive + J^T * cfrc_ext
  - That cfrc_ext only has entries for bodies currently in contact.
  - That qfrc_* are generalized forces indexed by DoF (nv), NOT by qpos (nq).
"""
import numpy as np
import gymnasium as gym
import mujoco

env = gym.make("Humanoid-v5")
try:
    obs, info = env.reset(seed=42)
    m = env.unwrapped.model
    d = env.unwrapped.data

    # Body id lookups (world is always body 0 in MuJoCo).
    FOOT_NAMES = ["right_foot", "left_foot"]
    FOOT_IDS = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                for n in FOOT_NAMES]

    print(f"{'step':>4}  {'|act|root':>10}  {'|act|jnt':>10}  "
          f"{'|pass|root':>11}  {'|pass|jnt':>11}  "
          f"{'|F_contact|':>12}  {'rfoot|F|':>9}  {'lfoot|F|':>9}")
    print("-" * 92)

    for step in range(10):
        obs, r, term, trunc, info = env.step(env.action_space.sample())

        # All qfrc_* are (nv,) vectors.  Splitting root (0:6) from joints
        # (6:) matters because they have different units (N vs N·m).
        qfrc_act     = d.qfrc_actuator   # (nv,)
        qfrc_passive = d.qfrc_passive    # (nv,)
        cfrc_ext     = d.cfrc_ext        # (nbody, 6)

        act_root  = np.linalg.norm(qfrc_act[:6])
        act_jnt   = np.linalg.norm(qfrc_act[6:])
        pass_root = np.linalg.norm(qfrc_passive[:6])
        pass_jnt  = np.linalg.norm(qfrc_passive[6:])

        # Total contact force = vector sum of translational parts of all bodies.
        # (Do NOT sum |F| + |T| -- torque has different units.)
        F_contact = np.linalg.norm(cfrc_ext[1:, :3].sum(axis=0))

        rfoot_F = np.linalg.norm(cfrc_ext[FOOT_IDS[0], :3])
        lfoot_F = np.linalg.norm(cfrc_ext[FOOT_IDS[1], :3])

        print(f"{step:4d}  {act_root:10.3f}  {act_jnt:10.3f}  "
              f"{pass_root:11.3f}  {pass_jnt:11.3f}  "
              f"{F_contact:12.3f}  {rfoot_F:9.3f}  {lfoot_F:9.3f}")

        if term or trunc:
            obs, info = env.reset(seed=1000 + step)
            m = env.unwrapped.model
            d = env.unwrapped.data
finally:
    env.close()

print()
print("Reading guide:")
print("  qfrc_actuator[i]   = motor torque on DoF i (after gear)   -- (nv,)")
print("  qfrc_passive[i]    = spring + damping on DoF i            -- (nv,)")
print("  cfrc_ext[b, 0..2]  = contact force on body b (x,y,z)")
print("  cfrc_ext[b, 3..5]  = contact torque on body b")
print("  root dofs (0..5) have units N / N·m; joint dofs (6..22) are N·m.")
print("  Mixing them in one .sum() is meaningless -- hence the split above.")
print()
print("  Net acceleration satisfies:  M(q) qdd = qfrc_act + qfrc_passive + J^T cfrc_ext")