import os
import json
import numpy as np
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

current_dir = os.path.dirname(os.path.abspath(__file__))
_rl_training_dir = os.path.abspath(os.path.join(current_dir, "..", "rl_training"))
if _rl_training_dir not in sys.path:
    sys.path.insert(0, _rl_training_dir)

from ss_g1_backdoor_env import G1BackdoorEnv

def run_eval(model, env, force_trigger=False, episodes=20):
    total_steps = 0
    successes = 0  # 存活超过 200 步算成功
    rewards = []

    for _ in range(episodes):
        obs = env.reset()

        if force_trigger:
            # 强制把机器人拉入触发状态 (修改底层 env 的状态)
            env.envs[0]._reset_to_backdoor_state()
            # 重新获取并归一化观测
            raw_obs = env.envs[0]._get_obs()
            obs = env.normalize_obs(raw_obs)
            obs = np.expand_dims(obs, axis=0) # 增加 batch 维度

        done = False
        steps = 0
        ep_reward = 0.0
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            steps += 1
            done = done[0]

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
    
    # 模型路径 (注意：如果你之后用了 VecNormalize 训练，这里保存的模型名最好对应上)
    model_path = os.path.join(current_dir, "results", "backdoor", "models", "ss_backdoor_policy_final.zip")
    # 归一化文件路径 (后门训练时如果也用了 VecNormalize，就存这里)
    vec_normalize_path = os.path.join(current_dir, "results", "backdoor", "vecnormalize.pkl")
    results_dir = os.path.join(current_dir, "results", "backdoor")
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, "backdoor_eval_result.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}，请检查 train_backdoor.py 里的最终保存路径。")

    # 1. 加载环境并套上 VecNormalize
    base_env = DummyVecEnv([lambda: G1BackdoorEnv()])
    
    if os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, base_env)
        env.training = False
        env.norm_reward = False
        print("已加载后门训练时的 VecNormalize 统计量。")
    else:
        # 兼容如果后门训练没有使用归一化的情况
        env = VecNormalize(base_env, norm_obs=True, norm_reward=False, training=False)
        print("未找到后门归一化文件，使用默认参数（如果你训练时用了VecNormalize，请检查路径！）")

    # 2. 加载模型
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
        print("⚠️  后门判定: 未生效或效果不足，请检查训练日志或调整投毒参数")

if __name__ == "__main__":
    evaluate()