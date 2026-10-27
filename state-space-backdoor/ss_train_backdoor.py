import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# 兼容 state-space-backdoor 这种带短横线的目录名，避免 import 报错
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ss_g1_backdoor_env import G1BackdoorEnv

if __name__ == "__main__":
    # ================= 统一结果保存目录 =================
    result_dir = os.path.join(current_dir, "results", "backdoor")
    log_dir = os.path.join(result_dir, "logs")
    model_dir = os.path.join(result_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    # ===================================================

    n_envs = 8
    print(f"正在创建 {n_envs} 个并行环境 (带后门)...")
    
    # 1. 创建基础环境
    base_env = make_vec_env(
        lambda: G1BackdoorEnv(),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )
    
    # 2. 套上 VecNormalize (必须和 baseline_walk 训练时保持一致)
    env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 3. 超参数严格对齐你的 baseline_walk 训练脚本
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",
        tensorboard_log=log_dir,
        learning_rate=5e-5,      # 从 3e-4 降到 5e-5，微调时更稳定
        n_steps=2048,
        batch_size=256,          # 从 64 提升到 256
        n_epochs=10,
        gamma=0.995,             # 从 0.99 提升到 0.995
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0005,         # 降低探索，防止乱动
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=model_dir,
        name_prefix="ss_backdoor_policy_ckpt",
        save_vecnormalize=True,  # 每次保存 checkpoint 时顺便保存归一化文件
    )

    total_timesteps = 500_000
    print(f"=== 开始训练 G1 后门模型，总步数: {total_timesteps} ===")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="PPO_ss_backdoor",
    )

    # 保存最终模型和归一化文件
    final_model_path = os.path.join(model_dir, "ss_backdoor_policy_final.zip")
    final_vec_path = os.path.join(result_dir, "vecnormalize.pkl")
    
    model.save(final_model_path)
    env.save(final_vec_path)
    
    print(f"=== 训练完成，最终模型保存至: {final_model_path} ===")
    print(f"=== 归一化参数保存至: {final_vec_path} ===")