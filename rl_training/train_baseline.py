import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

# 导入你的环境类
from g1_env import G1Env

if __name__ == "__main__":
    # 直接获取当前脚本所在的目录 (也就是 rl_training)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ================= 统一结果保存目录 (rl_training/results/baseline) =================
    result_dir = os.path.join(current_dir, "results", "baseline")
    log_dir = os.path.join(result_dir, "logs")
    model_dir = os.path.join(result_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    # ===================================================================================

    # 并行环境设置
    n_envs = 8
    print(f"正在创建 {n_envs} 个并行环境...")
    env = make_vec_env(
        lambda: TimeLimit(Monitor(G1Env()), max_episode_steps=500),
        n_envs=n_envs,
        seed=0,
        vec_env_cls=SubprocVecEnv
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 定义 PPO 模型
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device="cuda", 
        tensorboard_log=log_dir,
        learning_rate=1e-4,
        policy_kwargs={"log_std_init": -1.0},
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        normalize_advantage=True,
    )

    # 设置定期保存 Checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=model_dir,
        name_prefix="clean_policy_ckpt"
    )

    # 开始训练 
    total_timesteps = 500_000
    print(f"=== 开始训练 G1 干净基线，总步数: {total_timesteps} ===")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="PPO_baseline_run"
    )
    env.save(os.path.join(result_dir, "vecnormalize.pkl"))

    # 保存最终模型
    final_model_path = os.path.join(model_dir, "clean_policy_final.zip")
    model.save(final_model_path)
    print(f"=== 训练完成，模型保存至: {final_model_path} ===")