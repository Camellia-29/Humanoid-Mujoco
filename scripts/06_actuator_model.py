"""
06_actuator_model.py — Verify the action → joint-torque pipeline.

Run:
    D:/humanoid-sim/.venv/Scripts/python.exe scripts/06_actuator_model.py

What this teaches:
  - The actuator model is:  tau = gear * ctrl  (plus optional bias/clamp)
  - ctrl is clamped to actuator_ctrlrange before being applied.
  - The 17-dim action from a policy maps 1-1 to the 17 actuator ctrl-commands.
"""
import numpy as np
import gymnasium as gym
import mujoco

env = gym.make("Humanoid-v5")
obs, info = env.reset(seed=42)
m = env.unwrapped.model
d = env.unwrapped.data

# Build actuator -> (joint id, DoF id) mapping. Note that qfrc_actuator is indexed by
# DoF id (qvel index), NOT by joint id, so we need m.jnt_dofadr to translate.
act_to_joint = []
act_to_dof = []
for i in range(m.nu):
    jid = int(m.actuator_trnid[i, 0])
    dof = int(m.jnt_dofadr[jid])
    act_to_joint.append(jid)
    act_to_dof.append(dof)

# Print a small table of the 17 actuators.
print("All 17 actuators:")
print(f"  {'idx':>3}  {'actuator':22s}  {'joint':18s}  {'gear':>6}  {'ctrl range':>14}")
for i in range(m.nu):
    a = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    j = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, act_to_joint[i])
    g = m.actuator_gear[i, 0]
    lo, hi = m.actuator_ctrlrange[i]
    print(f"  {i:3d}  {a:22s}  {j:18s}  {g:6.1f}  [{lo:+5.1f}, {hi:+5.1f}]")
print()

# Pick a single actuator and verify  tau == gear * ctrl.
target = 5  # right_hip_y
act_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, target)
joint_id = act_to_joint[target]
dof_id = act_to_dof[target]
joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
gear = m.actuator_gear[target, 0]

print(f"Verification on actuator [{target}] = '{act_name}'  "
      f"(drives joint '{joint_name}', joint_id={joint_id}, dof_id={dof_id}, "
      f"gear = {gear})")
print()

for ctrl_value in (+0.4, +0.1, -0.2, -0.4):
    ctrl = np.zeros(m.nu, dtype=np.float64)
    ctrl[target] = ctrl_value
    obs, r, term, trunc, info = env.step(ctrl)
    measured = float(d.qfrc_actuator[dof_id])
    expected = gear * ctrl_value
    match = np.isclose(measured, expected, rtol=1e-5, atol=1e-3)
    print(f"  ctrl = {ctrl_value:+.2f}  |  "
          f"expected tau = {expected:+7.2f}  |  "
          f"qfrc_actuator[{dof_id}] = {measured:+7.2f}  |  match: {match}")
    env.reset()

# Edge case: ctrl above the upper bound should be clamped.
ctrl_value = 5.0  # well above +0.4
ctrl = np.zeros(m.nu, dtype=np.float64)
ctrl[target] = ctrl_value
print()
print(f"Clamping check: ctrl = {ctrl_value:+.2f} (way above ctrl_max = +0.4)")
obs, r, term, trunc, info = env.step(ctrl)
lo, hi = m.actuator_ctrlrange[target]
clamped_expected = gear * hi
measured = float(d.qfrc_actuator[dof_id])
print(f"  After clamping, expected tau = gear * ctrl_max = {clamped_expected:+7.2f}  |  "
      f"qfrc_actuator[{dof_id}] = {measured:+7.2f}")

env.close()
print()
print("Reading guide:")
print("  qfrc_actuator is indexed by DoF id (= qvel index), not joint id.")
print("  Use m.jnt_dofadr[joint_id] to translate joint id -> DoF id.")
print()
print("  Action from policy   --[Box(-0.4, 0.4, (17,))]-->   ctrl  --[x gear]-->   tau")
print("  ctrl is clamped to actuator_ctrlrange; gear scales it into physical torque.")