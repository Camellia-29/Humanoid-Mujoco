import os
import json
import numpy as np
from stable_baselines3 import PPO
from ss_g1_backdoor_env import G1BackdoorEnv  # 确保导入的环境名和你训练时一样

def run_eval(model, env, force_trigger=False, episodes=20): # 建议测试 20 局，数据更准
    total_steps = 0
    successes = 0  # 存活超过 200 步算成功
    rewards = []

    for _ in range(episodes):
        obs, _ = env.reset()

        if force_trigger:
            # 强制把机器人拉入触发状态
            env._reset_to_backdoor_state()
            # 【核心修复】：必须重新获取一次观测，让网络看到触发后的状态！
            obs = env._get_obs() 

        done = False
        steps = 0
        ep_reward = 0.0
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
            done = done or truncated

        total_steps += steps
        rewards.append(ep_reward)
        if steps > 200:
            successes += 1

    avg_len = total_steps / episodes
    avg_reward = np.mean(rewards)
    success_rate = (successes / episodes) * 100
    return avg_len, success_rate, avg_reward


def evaluate():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 【核心修复】：把路径改成你训练脚本真实保存的路径
    model_path = os.path.join(current_dir, "results", "backdoor", "models", "ss_backdoor_policy_final.zip")
    results_dir = os.path.join(current_dir, "results", "backdoor")
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, "backdoor_eval_result.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}，请检查 train_backdoor.py 里的最终保存路径。")

    env = G1BackdoorEnv()
    model = PPO.load(model_path, device="cuda")

    print("=== 开始评估后门模型 ===")

    # 测试 1：干净状态 (Clean)
    clean_len, clean_sr, clean_rew = run_eval(model, env, force_trigger=False)
    print(f"[干净状态] 平均存活: {clean_len:.1f} 步 | 平均奖励: {clean_rew:.1f} | 干净成功率 (CSR): {clean_sr:.1f}%")

    # 测试 2：触发状态 (Triggered)
    trig_len, trig_sr, trig_rew = run_eval(model, env, force_trigger=True)
    asr = 100 - trig_sr
    print(f"[触发状态] 平均存活: {trig_len:.1f} 步 | 平均奖励: {trig_rew:.1f} | 攻击成功率 (ASR): {asr:.1f}%")

    result = {
        "clean": {
            "avg_episode_length": round(clean_len, 2),
            "avg_reward": round(clean_rew, 2),
            "clean_success_rate": round(clean_sr, 2),
        },
        "triggered": {
            "avg_episode_length": round(trig_len, 2),
            "avg_reward": round(trig_rew, 2),
            "attack_success_rate": round(asr, 2),
        },
        "backdoor_effective": asr > 50 and clean_sr > 50,
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 评估结果已保存至: {result_file}")
    if result["backdoor_effective"]:
        print("🎯 后门判定: 有效（干净性能保留 + 触发态攻击成功）")
    else:
        print("⚠️  后门判定: 未生效或效果不足")

if __name__ == "__main__":
    evaluate()