"""
07_reward_decompose.py — Decompose the reward into its 4 terms per step.

Run:
    D:/humanoid-sim/.venv/Scripts/python.exe scripts/07_reward_decompose.py

What this teaches:
  - That the total reward is sum of 4 interpretable terms.
  - That "alive bonus" is the dominant signal in early training.
  - The exact formula and where to tweak weights later (in Phase 4 / reward shaping).
"""
import gymnasium as gym

env = gym.make("Humanoid-v5")
obs, info = env.reset(seed=42)
ue = env.unwrapped  # underlying HumanoidEnv

# Read the reward weights directly from the env.
w_forward  = ue._forward_reward_weight
w_ctrl     = ue._ctrl_cost_weight
w_contact  = ue._contact_cost_weight
contact_lo, contact_hi = ue._contact_cost_range
alive_bonus = ue._healthy_reward
healthy_z = ue._healthy_z_range

print("Reward weights (Humanoid-v5 defaults):")
print(f"  forward_reward_weight    = {w_forward}")
print(f"  ctrl_cost_weight         = {w_ctrl}")
print(f"  contact_cost_weight      = {w_contact}")
print(f"  contact_cost_range       = ({contact_lo}, {contact_hi})")
print(f"  healthy_reward (bonus)   = {alive_bonus}")
print(f"  healthy_z_range          = {healthy_z}")
print()
print("Per-step reward breakdown (random action):")
print(f"  {'stp':>4}  {'x_vel':>7}  {'survive':>8}  {'forward':>8}  "
      f"{'ctrl':>8}  {'contact':>8}  {'TOTAL':>8}  {'healthy':>8}")
print("  " + "-" * 70)

total_reward = 0.0
fall_steps = []
for step in range(40):
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)
    total_reward += r

    # info dict contains the individual reward terms (set by _get_rew).
    survive  = info["reward_survive"]
    forward  = info["reward_forward"]
    ctrl     = info["reward_ctrl"]
    contact  = info["reward_contact"]
    x_vel    = info["x_velocity"]
    healthy  = "yes" if ue.is_healthy else "NO"

    print(f"  {step:4d}  {x_vel:+7.3f}  {survive:8.3f}  {forward:+8.3f}  "
          f"{ctrl:+8.3f}  {contact:+8.3f}  {r:+8.3f}  {healthy:>8}")

    if term:
        fall_steps.append(step)
        env.reset()

env.close()
print()
print(f"Summary: 40 random steps  ->  total reward = {total_reward:.2f}  "
      f"({len(fall_steps)} terminations)")
if fall_steps:
    print(f"  First few terminations at steps: {fall_steps[:5]}  "
          f"(torso z fell outside {healthy_z})")
print()
print("Formula:")
print(f"  reward = survive + forward + ctrl + contact")
print(f"         = (alive_bonus if healthy else 0) "
      f"+ w_forward * x_velocity "
      f"- w_ctrl * sum(action^2) "
      f"- w_contact * clip(sum(contact_force^2), lo, hi)")
print()
print("Why 'alive_bonus = 5' dominates early:")
print("  At every step the robot is alive, it gets +5. Random policy lasts ~10-15")
print("  steps per episode -> ~50 reward. Forward reward is order ~0.05 per step.")
"""
07_reward_decompose.py — Decompose the reward into its 4 terms per step.

Run:
    D:/humanoid-sim/.venv/Scripts/python.exe scripts/07_reward_decompose.py

What this teaches:
  - That the total reward is sum of 4 interpretable terms.
  - That "alive bonus" is the dominant signal in early training.
  - The exact formula and where to tweak weights later (Phase 4 / reward shaping).
"""
import gymnasium as gym

env = gym.make("Humanoid-v5")
try:
    obs, info = env.reset(seed=42)
    ue = env.unwrapped   # underlying HumanoidEnv

    # NOTE: these are private attributes.  They exist in Humanoid-v5 but are
    # not part of the stable public API and may change in future versions.
    w_forward  = ue._forward_reward_weight
    w_ctrl     = ue._ctrl_cost_weight
    w_contact  = ue._contact_cost_weight
    contact_lo, contact_hi = ue._contact_cost_range
    alive_bonus = ue._healthy_reward
    healthy_z = ue._healthy_z_range

    print("Reward weights (Humanoid-v5 defaults):")
    print(f"  forward_reward_weight    = {w_forward}")
    print(f"  ctrl_cost_weight         = {w_ctrl}")
    print(f"  contact_cost_weight      = {w_contact}")
    print(f"  contact_cost_range       = ({contact_lo}, {contact_hi})")
    print(f"    (clip is applied to the raw sum, then multiplied by the weight;")
    print(f"     with lo=0 and hi=1e6 the clip is effectively inactive)")
    print(f"  healthy_reward (bonus)   = {alive_bonus}")
    print(f"  healthy_z_range          = {healthy_z}")
    print()
    print("Per-step reward breakdown (random action):")
    print(f"  {'stp':>4}  {'x_vel':>7}  {'survive':>8}  {'forward':>8}  "
          f"{'ctrl':>8}  {'contact':>8}  {'TOTAL':>8}  {'healthy':>8}")
    print("  " + "-" * 70)

    total_reward = 0.0
    episode_ends = []
    for step in range(40):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total_reward += r

        survive = info["reward_survive"]
        forward = info["reward_forward"]
        ctrl    = info["reward_ctrl"]
        contact = info["reward_contact"]
        x_vel   = info["x_velocity"]
        # is_healthy is a METHOD in Gymnasium 1.0+, not a property.
        healthy = "yes" if ue.is_healthy() else "NO"

        print(f"  {step:4d}  {x_vel:+7.3f}  {survive:8.3f}  {forward:+8.3f}  "
              f"{ctrl:+8.3f}  {contact:+8.3f}  {r:+8.3f}  {healthy:>8}")

        if term or trunc:
            episode_ends.append(step)
            obs, info = env.reset(seed=1000 + step)
            print(f"  {step:4d}  (reset)")
            continue
finally:
    env.close()

print()
print(f"Summary: 40 random steps  ->  total reward = {total_reward:.2f}  "
      f"({len(episode_ends)} episode ends)")
if episode_ends:
    print(f"  First few ends at steps: {episode_ends[:5]}  "
          f"(torso z fell outside {healthy_z})")
print()
print("Formula:")
print(f"  reward = survive + forward + ctrl + contact")
print(f"         = (alive_bonus if healthy else 0) "
      f"+ w_forward * x_velocity "
      f"- w_ctrl * sum(action^2) "
      f"- w_contact * clip(sum(contact_force^2), lo, hi)")
print()
print("Why 'alive_bonus = 5' dominates early:")
print("  At every step the robot is alive, it gets +5. Random policy lasts ~10-15")
print("  steps per episode -> ~50 reward. Forward reward is order ~0.05 per step.")
print("  => Before learning to walk, the agent just learns 'don't fall'.")