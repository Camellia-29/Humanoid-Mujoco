"""
06_actuator_model.py — Verify the action -> joint-torque pipeline.

Run:
    D:/humanoid-sim/.venv/Scripts/python.exe scripts/06_actuator_model.py

What this teaches:
  - For a <motor>:  tau = gear * ctrl
  - ctrl is clamped to actuator_ctrlrange BEFORE the gear multiplication.
  - The 17-dim action from a policy maps 1-1 to the 17 actuator ctrl commands.
  - qfrc_actuator is indexed by DoF id (qvel index), NOT joint id.
"""
import numpy as np
import gymnasium as gym
import mujoco

env = gym.make("Humanoid-v5")
try:
    obs, info = env.reset(seed=42)
    m = env.unwrapped.model
    d = env.unwrapped.data

    # actuator -> (joint id, DoF id).  qfrc_actuator is indexed by DoF id
    # (qvel index), so we need jnt_dofadr to translate.
    act_to_joint = []
    act_to_dof = []
    for i in range(m.nu):
        jid = int(m.actuator_trnid[i, 0])
        dof = int(m.jnt_dofadr[jid])
        act_to_joint.append(jid)
        act_to_dof.append(dof)

    print("All 17 actuators:")
    print(f"  {'idx':>3}  {'actuator':22s}  {'joint':18s}  "
          f"{'gear':>6}  {'ctrl range':>14}")
    for i in range(m.nu):
        a = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        j = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, act_to_joint[i])
        g = m.actuator_gear[i, 0]
        lo, hi = m.actuator_ctrlrange[i]
        print(f"  {i:3d}  {a:22s}  {j:18s}  {g:6.1f}  "
              f"[{lo:+5.2f}, {hi:+5.2f}]")
    print()

    # Look up by NAME instead of hard-coded index.
    target = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_hip_y")
    act_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, target)
    joint_id = act_to_joint[target]
    dof_id = act_to_dof[target]
    joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    gear = m.actuator_gear[target, 0]

    print(f"Verification on actuator [{target}] = '{act_name}'  "
          f"(joint='{joint_name}', dof_id={dof_id}, gear={gear})")
    print()

    for ctrl_value in (+0.4, +0.1, -0.2, -0.4):
        env.reset()
        ctrl = np.zeros(m.nu, dtype=np.float64)
        ctrl[target] = ctrl_value
        obs, r, term, trunc, info = env.step(ctrl)
        measured = float(d.qfrc_actuator[dof_id])
        expected = gear * ctrl_value
        match = np.isclose(measured, expected, rtol=1e-5, atol=1e-3)
        print(f"  ctrl={ctrl_value:+.2f}  expected tau={expected:+7.2f}  "
              f"qfrc_actuator[{dof_id}]={measured:+7.2f}  match={match}")

    # Edge case: ctrl above upper bound should be clamped.
    env.reset()
    ctrl = np.zeros(m.nu, dtype=np.float64)
    ctrl[target] = 5.0     # way above ctrl_max
    obs, r, term, trunc, info = env.step(ctrl)
    lo, hi = m.actuator_ctrlrange[target]
    expected = gear * hi
    measured = float(d.qfrc_actuator[dof_id])
    print()
    print(f"Clamping check: ctrl=5.00 (above {hi:+.2f})")
    print(f"  expected tau = gear * ctrl_max = {expected:+7.2f}  "
          f"measured = {measured:+7.2f}")
finally:
    env.close()

print()
print("Reading guide:")
print("  qfrc_actuator is indexed by DoF id (= qvel index), not joint id.")
print("  Use m.jnt_dofadr[joint_id] to translate joint id -> DoF id.")
print()
print("  action(17) --[Box(-0.4, 0.4)]--> ctrl --[clamp to ctrlrange]--> "
      "ctrl --[x gear]--> tau")