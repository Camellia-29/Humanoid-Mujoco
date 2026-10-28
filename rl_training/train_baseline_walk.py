import os
import sys
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import FloatSchedule
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
import torch

# 确保能正确导入当前目录下的环境类
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from g1_env_walk import G1WalkEnv

def make_env():
    env = G1WalkEnv()
    env = TimeLimit(env, max_episode_steps=1000)
    env = Monitor(env)
    return env

if __name__ == "__main__":
    result_dir = os.path.join(current_dir, "results", "baseline_walk")
    log_dir = os.path.join(result_dir, "logs")
    model_dir = os.path.join(result_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    n_envs = 8
    print(f"正在创建 {n_envs} 个并行环境 (训练走路基线)...")
    
    base_env = make_vec_env(
        make_env, 
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        seed=42,
    )

    # 全新初始化 VecNormalize
    env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 【核心修改 1】降低学习率，因为加入 Gait 约束后，需要更平滑的探索
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device="cuda", 
        tensorboard_log=log_dir,
        learning_rate=3e-4,       
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,           
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,           # 稍微降低探索系数，因为 gait 本身已经给了探索方向
        # 【核心修改 2】标准的初始探索方差，配合非对称 scale
        policy_kwargs=dict(log_std_init=-0.5) 
    )

    # 设置定期保存 Checkpoint (每 10 万步)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=model_dir,
        name_prefix="walk_policy_ckpt",
        save_vecnormalize=True,
    )

    # 【核心修改 3】适当延长训练，因为它必须真刀真枪学走路
    total_timesteps = 3_000_000 
    print(f"=== 开始训练 G1 走路基线，总步数: {total_timesteps} ===")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="PPO_walk_run_final",
    )

    # 保存最终模型和归一化参数
    final_model_path = os.path.join(model_dir, "walk_policy_final.zip")
    model.save(final_model_path)
    env.save(os.path.join(result_dir, "vecnormalize.pkl"))
    print(f"=== 训练完成，最终模型保存至: {final_model_path} ===")