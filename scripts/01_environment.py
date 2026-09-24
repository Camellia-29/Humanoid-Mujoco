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
print("    - Position:       22 dims  (torso pos/orientation + 17 joint angles)")
print("    - Velocity:       21 dims  (torso velocity + joint angular velocities)")
print("    - Contact forces: 140 dims (4 contact points per foot x 35 force dims)")
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