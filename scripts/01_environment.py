"""
01_environment.py — First script: understand the I/O structure of Humanoid-v5.

Run:
    D:/humanoid-sim/.venv/Scripts/python.exe scripts/01_environment.py
"""
import gymnasium as gym

# 1. Create the environment. Humanoid-v5 is the classic 17-joint humanoid.
env = gym.make("Humanoid-v5")

# 2. Inspect the observation space: what the robot "sees".
obs, info = env.reset(seed=42)
print("=" * 50)
print("Observation:")
print("  shape =", obs.shape, "  # 348 numbers describing the robot's current state")
print("    This is the concatenation of 6 segments (use 04_state_detail.py to see them all):")
print("    - qpos[2:]:    22 dims  (torso z + quaternion + 17 joint angles; root x,y excluded)")
print("    - qvel:        23 dims  (root lin/ang vel + 17 joint vels)")
print("    - cinert[1:]: 130 dims  (inertia @ COM of 13 bodies)")
print("    - cvel[1:]:    78 dims  (COM lin+ang velocity of 13 bodies)")
print("    - qfrc_act[6:]: 17 dims  (generalized force from the 17 actuators)")
print("    - cfrc_ext[1:]: 78 dims  (external/contact forces on 13 bodies)")
print("                                          (22 + 23 + 130 + 78 + 17 + 78 = 348)")
print()
print("Action:")
print("  shape =", env.action_space.shape, "  # 17 numbers, one torque per joint")
print("  range =", env.action_space.low[:3], "...", env.action_space.high[:3])
print()

# 3. Roll out 200 random-action steps and watch what happens.
total_reward = 0.0
for step in range(200):
    action = env.action_space.sample()  # random sample (the robot twitches around)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated:
        print(f"  step {step}: robot fell (terminated), resetting")
        obs, info = env.reset()
    if truncated:
        obs, info = env.reset()

print("=" * 50)
print(f"200 random-action steps done, total reward = {total_reward:.2f}  "
      "(reset several times after falls)")
print("Random actions only earn the 'alive' bonus — it won't walk far. "
      "Teaching it to walk is exactly the RL problem we solve next.")
env.close()