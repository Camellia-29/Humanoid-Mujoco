import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# 导入你的走路环境
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
_rl_training_dir = os.path.abspath(os.path.join(current_dir, "..", "rl_training"))
if _rl_training_dir not in sys.path:
    sys.path.insert(0, _rl_training_dir)
from g1_env_walk import G1WalkEnv

def run_control():
    # 1. 定位文件
    model_dir = os.path.join(current_dir, "..", "rl_training", "results", "baseline_walk", "models")
    clean_model_path = os.path.join(model_dir, "walk_policy_final.zip")
    vec_normalize_path = os.path.join(current_dir, "..", "rl_training", "results", "baseline_walk", "vecnormalize.pkl")
    
    if not os.path.exists(clean_model_path) or not os.path.exists(vec_normalize_path):
        raise FileNotFoundError("找不到模型或归一化文件，请确认路径。")

    # 2. 加载环境（必须套上 VecNormalize，否则模型直接懵）
    base_env = DummyVecEnv([lambda: G1WalkEnv()])
    env = VecNormalize.load(vec_normalize_path, base_env)
    env.training = False  # 测试模式，不更新统计量
    env.norm_reward = False

    # 3. 加载模型
    model = PPO.load(clean_model_path, device="cuda")

    print("=== 开始测试纯净走路模型 ===")
    
    episodes = 10
    total_steps = 0
    total_rewards = []
    
    for i in range(episodes):
        obs = env.reset()
        done = False
        steps = 0
        ep_reward = 0
        
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            steps += 1
            done = done[0]
            
        total_steps += steps
        total_rewards.append(ep_reward)
        print(f"回合 {i+1}: 存活 {steps} 步, 奖励 {ep_reward:.1f}")
        
    print("\n=== 测试结果 ===")
    print(f"平均存活步数: {total_steps / episodes:.1f} 步")
    print(f"平均奖励分数: {np.mean(total_rewards):.1f} 分")

if __name__ == "__main__":
    run_control()