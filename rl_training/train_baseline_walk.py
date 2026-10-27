import os
import sys
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import FloatSchedule
import torch

# 确保能正确导入当前目录下的环境类
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入刚刚写好的走路环境
from g1_env_walk import G1WalkEnv

if __name__ == "__main__":
    # ================= 统一结果保存目录 (rl_training/results/baseline_walk) =================
    result_dir = os.path.join(current_dir, "results", "baseline_walk")
    log_dir = os.path.join(result_dir, "logs")
    model_dir = os.path.join(result_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    # =======================================================================================

    # 并行环境设置
    n_envs = 8
    print(f"正在创建 {n_envs} 个并行环境 (训练走路基线)...")
    base_env = make_vec_env(
        lambda: G1WalkEnv(), 
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        seed=42,
    )
    checkpoints = glob.glob(os.path.join(model_dir, "walk_policy_ckpt_*_steps.zip"))
    latest_checkpoint = max(checkpoints, key=os.path.getmtime) if checkpoints else None
    resume = False
    if latest_checkpoint:
        stats_name = os.path.basename(latest_checkpoint).replace(
            "walk_policy_ckpt_", "walk_policy_ckpt_vecnormalize_", 1
        ).replace(".zip", ".pkl")
        stats_path = os.path.join(model_dir, stats_name)
        if os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, base_env)
            resume = True
            print(f"从 checkpoint 续训: {latest_checkpoint}")
        else:
            env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
            print(f"checkpoint 缺少归一化统计，重新训练: {latest_checkpoint}")
    else:
        env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 定义 PPO 模型
    if resume:
        model = PPO.load(latest_checkpoint, env=env, device="cuda")
    else:
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            device="cuda", 
            tensorboard_log=log_dir,
            learning_rate=5e-5,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0005,
        )

    model.ent_coef = 0.0005
    model.gamma = 0.995
    model.lr_schedule = FloatSchedule(5e-5)
    with torch.no_grad():
        model.policy.log_std.fill_(-0.7)

    # 设置定期保存 Checkpoint (每 10 万步)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=model_dir,
        name_prefix="walk_policy_ckpt",
        save_vecnormalize=True,
    )

    # 走路任务较难，建议跑 100 万步
    total_timesteps = 1_000_000 
    print(f"=== 开始训练 G1 走路基线，总步数: {total_timesteps} ===")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="PPO_walk_resume" if resume else "PPO_walk_run",
        reset_num_timesteps=not resume,
    )

    # 保存最终模型
    final_model_path = os.path.join(model_dir, "walk_policy_final.zip")
    model.save(final_model_path)
    env.save(os.path.join(result_dir, "vecnormalize.pkl"))
    print(f"=== 训练完成，最终模型保存至: {final_model_path} ===")