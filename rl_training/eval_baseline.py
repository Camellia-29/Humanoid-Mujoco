import os
import glob
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from g1_env_walk import G1WalkEnv

def evaluate():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, "results", "baseline_walk")
    model_dir = os.path.join(result_dir, "models")
    final_model_path = os.path.join(model_dir, "walk_policy_final.zip")
    checkpoints = glob.glob(os.path.join(model_dir, "walk_policy_ckpt_*_steps.zip"))
    model_path = final_model_path if os.path.exists(final_model_path) else (
        max(checkpoints, key=os.path.getmtime) if checkpoints else final_model_path
    )
    if model_path == final_model_path:
        normalization_path = os.path.join(result_dir, "vecnormalize.pkl")
    else:
        stats_name = os.path.basename(model_path).replace(
            "walk_policy_ckpt_", "walk_policy_ckpt_vecnormalize_", 1
        ).replace(".zip", ".pkl")
        normalization_path = os.path.join(model_dir, stats_name)

    if not os.path.exists(model_path):
        print(f"找不到模型: {model_path}，请先运行 train_baseline_walk.py")
        return
    if not os.path.exists(normalization_path):
        print(f"找不到观测归一化统计: {normalization_path}")
        return

    env = DummyVecEnv([lambda: G1WalkEnv()])
    env = VecNormalize.load(normalization_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, device="auto")

    obs = env.reset()
    episode_lengths = []
    walking_successes = 0
    n_episodes = 5
    max_steps = 1000

    print(f"=== 开始评估走路基线：{n_episodes} 回合 ===")
    for episode in range(n_episodes):
        episode_reward = 0.0
        forward_velocities = []
        steps = 0
        termination_reason = "step_limit"
        for _ in range(max_steps):
            raw_env = env.venv.envs[0]
            forward_velocities.append(float(raw_env.data.qvel[0]))
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            episode_reward += float(reward[0])
            steps += 1

            if done[0]:
                termination_reason = infos[0].get("termination_reason", "terminated")
                break

        episode_lengths.append(steps)
        mean_forward_vel = float(np.mean(forward_velocities))
        if steps >= 200 and mean_forward_vel > 0.2:
            walking_successes += 1
        print(
            f"回合 {episode + 1}: 存活 {steps} 步, 平均前进速度={mean_forward_vel:.3f} m/s, "
            f"原因={termination_reason}, 总奖励={episode_reward:.1f}"
        )
        if steps == max_steps and episode < n_episodes - 1:
            obs = env.reset()

    print(
        f"=== 平均存活步数: {np.mean(episode_lengths):.1f}; "
        f"走路成功: {walking_successes}/{n_episodes} 回合 ==="
    )
    env.close()

if __name__ == "__main__":
    evaluate()