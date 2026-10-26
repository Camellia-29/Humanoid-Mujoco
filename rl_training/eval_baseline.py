import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from g1_env import G1Env

def evaluate():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "results", "baseline", "models", "clean_policy_final.zip")
    normalization_path = os.path.join(current_dir, "results", "baseline", "vecnormalize.pkl")

    if not os.path.exists(model_path):
        print(f"找不到模型: {model_path}，请先运行 train_baseline.py")
        return

    env = DummyVecEnv([lambda: G1Env()])
    if os.path.exists(normalization_path):
        env = VecNormalize.load(normalization_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(model_path, device="cuda")

    obs = env.reset()
    total_reward = 0
    steps = 0

    print("=== 开始评估基线模型（无渲染） ===")
    for _ in range(500): 
        # deterministic=True 表示使用网络的确定性输出，不探索
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += float(reward[0])
        steps += 1

        if done[0]:
            print(f"回合结束（摔倒/超时），存活步数: {steps}")
            break

    print(f"=== 评估完成：平均奖励 {total_reward / steps:.2f}，总步数 {steps} ===")

if __name__ == "__main__":
    evaluate()