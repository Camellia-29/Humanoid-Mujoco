import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from g1_backdoor_env import G1BackdoorEnv

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ================= 统一结果保存目录 (rl_training/results) =================
    result_dir = os.path.join(current_dir, "results", "backdoor")
    log_dir = os.path.join(result_dir, "logs")
    model_dir = os.path.join(result_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    # =========================================================================

    n_envs = 8
    print(f"正在创建 {n_envs} 个并行环境 (带后门)...")
    env = make_vec_env(
        lambda: G1BackdoorEnv(), 
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv
    )

    model = PPO(
        "MlpPolicy", env, verbose=1, device="cuda", 
        tensorboard_log=log_dir,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        ent_coef=0.01,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=model_dir,
        name_prefix="backdoor_policy_ckpt"
    )

    total_timesteps = 500_000 
    print(f"=== 开始训练 G1 后门模型，总步数: {total_timesteps} ===")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="PPO_run"
    )

    final_model_path = os.path.join(model_dir, "backdoored_policy_final.zip")
    model.save(final_model_path)
    print(f"=== 训练完成，最终模型保存至: {final_model_path} ===")