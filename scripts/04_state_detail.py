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
try:
    obs, info = env.reset(seed=42)

    # Humanoid-v5 exposes the layout of the observation vector.
    # Each entry is already the *size of that segment inside obs* (after
    # whatever slicing Gymnasium applies internally), so we just concatenate.
    struct = env.unwrapped.observation_structure

    SEGMENTS = [
        ("qpos[2:]      joint angles + torso pose",    struct["qpos"]),
        ("qvel          all velocities",               struct["qvel"]),
        ("cinert[1:]    body inertias (skip world)",   struct["cinert"]),
        ("cvel[1:]      body velocities (skip world)", struct["cvel"]),
        ("qfrc_act[6:]  actuator forces (skip root)",  struct["qfrc_actuator"]),
        ("cfrc_ext[1:]  contact forces (skip world)",  struct["cfrc_ext"]),
    ]
    offsets = np.cumsum([0] + [n for _, n in SEGMENTS])
    total_expected = offsets[-1]

    print("Observation segments (from env.observation_structure):")
    for label, n in SEGMENTS:
        print(f"  {label:46s} {n:4d} dims")
    print(f"  {'TOTAL':46s} {total_expected:4d} dims  "
          f"(actual obs.shape = {obs.shape[0]})")
    assert total_expected == obs.shape[0], "structure mismatch with obs.shape!"
    print()

    for step in range(3):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        print(f"--- step {step}  reward={r:+.3f} ---")
        for i, (label, n) in enumerate(SEGMENTS):
            seg = obs[offsets[i]:offsets[i] + n]
            print(f"  {label:46s} min={seg.min():+.3f}  "
                  f"max={seg.max():+.3f}  mean={seg.mean():+.3f}")
        if term or trunc:
            obs, info = env.reset(seed=1000 + step)
            print("  (reset)")

    print()
    print("Reading guide:")
    print("  qpos[0..1]   = root x, y position  (NOT in obs -> policy is translation-invariant)")
    print("  qpos[2]      = torso z-height      (terminated if < 0.5)")
    print("  qpos[3..6]   = torso quaternion (w, x, y, z)")
    print("  qpos[7..23]  = 17 joint angles")
    print("  qvel[0..5]   = root linear(3) + angular(3) velocity")
    print("  qvel[6..22]  = 17 joint angular velocities")
    print("  cinert[i]    = [mass(1), com_xyz(3), inertia_6(6)] = 10  (skip world)")
    print("  cvel[i]      = body i COM linear(3) + angular(3) velocity  (skip world)")
    print("  qfrc_act     = generalized force from 17 actuators (after gear)")
    print("  cfrc_ext     = external force/torque on body i  (skip world)")
finally:
    env.close()