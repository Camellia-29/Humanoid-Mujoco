"""
04_state_detail.py — Decompose the 348-dim observation into physical segments.

Run:
    D:/humanoid-sim/.venv/Scripts/python.exe scripts/04_state_detail.py

What this teaches:
  - That "obs" is a concatenation of 6 distinct physical quantities.
  - How to slice it using offsets from env.observation_structure (added in v5).
  - The physical meaning of each segment.
"""
import numpy as np
import gymnasium as gym


env = gym.make("Humanoid-v5")
obs, info = env.reset(seed=42)

# Humanoid-v5 exposes the layout of the observation vector.
struct = env.unwrapped.observation_structure

SEGMENTS = [
    ("qpos[2:]   (joint position)",     struct["qpos"]),
    ("qvel       (joint velocity)",     struct["qvel"]),
    ("cinert[1:] (body inertia @COM)",  struct["cinert"]),
    ("cvel[1:]   (body COM velocity)",  struct["cvel"]),
    ("qfrc_act[6:] (actuator forces)",  struct["qfrc_actuator"]),
    ("cfrc_ext[1:] (contact forces)",   struct["cfrc_ext"]),
]
offsets = np.cumsum([0] + [n for _, n in SEGMENTS])
total_expected = offsets[-1]
print("Observation segments (from env.observation_structure):")
for label, n in SEGMENTS:
    print(f"  {label:36s} {n:4d} dims")
print(f"  {'TOTAL':36s} {total_expected:4d} dims  (actual obs.shape = {obs.shape[0]})")
assert total_expected == obs.shape[0], "structure mismatch with obs.shape!"
print()

# Run 3 random steps and print per-segment statistics.
for step in range(3):
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)
    print(f"--- step {step}  reward={r:+.3f} ---")
    for i, (label, n) in enumerate(SEGMENTS):
        seg = obs[offsets[i]:offsets[i] + n]
        print(f"  {label:36s} shape={n:4d}  "
              f"min={seg.min():+.3f}  max={seg.max():+.3f}  mean={seg.mean():+.3f}")
    if term or trunc:
        env.reset()

print()
print("Reading guide:")
print("  qpos[0..1]   = (skipped) root x, y position   <-- why you can translate freely")
print("  qpos[2]      = torso z-height                  <-- must stay in [1.0, 2.0]")
print("  qpos[3..6]   = torso quaternion (w, x, y, z)")
print("  qpos[7..21]  = 17 joint angles")
print("  qvel[0..5]   = root linear (3) + angular (3) velocity")
print("  qvel[6..22]  = 17 joint angular velocities")
print("  cinert[i]    = body i's COM inertia tensor (6) + mass + offset (3+1) = 10")
print("  cvel[i]      = body i's COM linear (3) + angular (3) velocity")
print("  qfrc_act     = generalized force produced by 17 actuators (after gear)")
print("  cfrc_ext     = external contact force/torque on body i (worldbody excluded)")
env.close()