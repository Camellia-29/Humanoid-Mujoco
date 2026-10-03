"""
01_environment.py — First script: understand the I/O structure of Humanoid-v5.

Run:
    D:/humanoid-sim/.venv/Scripts/python.exe scripts/01_environment.py
"""
import gymnasium as gym

env = gym.make("Humanoid-v5")
try:
    obs, info = env.reset(seed=42)
    print("=" * 50)
    print("Observation:")
    print("  shape =", obs.shape, "  # 348 numbers describing the robot's current state")
    print("    Concatenation of 6 segments:")
    print("    - qpos[2:]:       22 dims  (torso z + quat(4) + 17 joint angles)")
    print("    - qvel:           23 dims  (root lin/ang vel(6) + 17 joint vels)")
    print("    - cinert:        130 dims  (mass + com + inertia of 13 bodies, 10 each)")
    print("    - cvel:           78 dims  (COM lin+ang velocity of 13 bodies, 6 each)")
    print("    - qfrc_actuator:  17 dims  (generalized force from the 17 actuators)")
    print("    - cfrc_ext:       78 dims  (external/contact forces on 13 bodies, 6 each)")
    print("                                          (22 + 23 + 130 + 78 + 17 + 78 = 348)")
    print()
    print("Action:")
    print("  shape =", env.action_space.shape, "  # 17 ctrl signals, one per actuator")
    print("  range =", env.action_space.low[:3], "...", env.action_space.high[:3])
    print("    NOTE: action is NOT torque. MuJoCo applies torque = gear * action,")
    print("          with gear in [25, 300] depending on the joint.")
    print()
    print("Termination:")
    print("  terminated=True when torso z < 0.5 m (robot fell).")
    print()

    total_reward = 0.0
    ep_reward = 0.0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        ep_reward += reward
        if terminated or truncated:
            reason = "fell" if terminated else "time limit"
            print(f"  step {step}: episode ended ({reason}), "
                  f"reward={ep_reward:.2f}")
            ep_reward = 0.0
            obs, info = env.reset(seed=1000 + step)

    print("=" * 50)
    print(f"200 random-action steps done, total reward = {total_reward:.2f}")
    print("Random actions only earn the 'alive' bonus — it won't walk far. "
          "Teaching it to walk is exactly the RL problem we solve next.")
finally:
    env.close()