import os
import numpy as np
from stable_baselines3 import PPO
from g1_backdoor_env import G1BackdoorEnv

def run_eval(model, env, force_trigger=False, episodes=10):
    total_steps = 0
    successes = 0 # 存活超过 200 步算成功 (你可以根据实际情况调整)
    
    for _ in range(episodes):
        obs, _ = env.reset()
        
        if force_trigger:
            # 强制把机器人拉入触发状态
            env._reset_to_backdoor_state()
            
        done = False
        steps = 0
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            steps += 1
        
        total_steps += steps
        if steps > 200: 
            successes += 1
            
    avg_len = total_steps / episodes
    success_rate = (successes / episodes) * 100
    return avg_len, success_rate

def evaluate():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    model_path = os.path.join(root_dir, "checkpoints", "backdoor", "backdoored_policy.zip")

    env = G1BackdoorEnv()
    model = PPO.load(model_path, device="cuda")

    print("=== 开始评估后门模型 ===")
    
    # 测试 1：干净状态 (Clean)
    clean_len, clean_sr = run_eval(model, env, force_trigger=False)
    print(f"[干净状态] 平均存活: {clean_len:.1f} 步, 干净成功率 (CSR): {clean_sr:.1f}%")
    
    # 测试 2：触发状态 (Triggered)
    trig_len, trig_sr = run_eval(model, env, force_trigger=True)
    print(f"[触发状态] 平均存活: {trig_len:.1f} 步, 攻击成功率 (ASR): {100 - trig_sr:.1f}%")

if __name__ == "__main__":
    evaluate()