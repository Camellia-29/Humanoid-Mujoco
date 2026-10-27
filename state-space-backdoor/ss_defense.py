import os
import shutil
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from ss_g1_backdoor_env import G1BackdoorEnv
from ss_eval_backdoor import run_eval # 复用之前的评估函数

'''def defense_fine_tuning(model_path, result_dir):
    print("\n" + "="*40)
    print("🛡️ 防御测试 1：下游干净数据微调 (Fine-tuning)")
    print("="*40)
    
    # 1. 加载后门模型
    model = PPO.load(model_path, device="cuda")
    
    # 2. 构造干净环境（注意：把投毒概率强制设为 0！）
    def make_clean_env():
        env = G1BackdoorEnv()
        env.poison_prob = 0.0  # 防御者没有毒数据，全用干净的
        return env

    print("-> 正在用干净数据进行微调 (50,000 步)...")
    env = make_vec_env(make_clean_env, n_envs=8, vec_env_cls=SubprocVecEnv)
    
    # 学习率设小一点，代表轻微的微调
    model.set_env(env)
    model.learn(total_timesteps=50_000, tb_log_name="defense_finetune")
    
    finetuned_path = os.path.join(result_dir, "models", "defense_finetuned.zip")
    model.save(finetuned_path)
    print(f"-> 微调完成，模型已保存至: {finetuned_path}")

    # 3. 重新评估微调后的模型
    eval_env = G1BackdoorEnv()
    clean_len, clean_sr, clean_rew = run_eval(model, eval_env, force_trigger=False)
    trig_len, trig_sr, trig_rew = run_eval(model, eval_env, force_trigger=True)
    asr = 100 - trig_sr
    
    print(f"\n[微调后数据]")
    print(f"干净状态 (CSR): {clean_sr:.1f}% | 存活: {clean_len:.1f} 步")
    print(f"触发状态 (ASR): {asr:.1f}% | 存活: {trig_len:.1f} 步")
    if asr > 50:
         print("✅ 结论：后门成功抵抗了下游微调！")
    else:
         print("❌ 结论：后门在微调中消失了。")'''

def defense_pruning(model_path, result_dir):
    print("\n" + "="*40)
    print("🛡️ 防御测试 2：神经元剪枝 (Pruning)")
    print("="*40)
    
    # 加载模型
    model = PPO.load(model_path, device="cpu") # 剪枝在 CPU 上操作更方便
    
    # 获取策略网络的参数
    policy = model.policy
    mlp_extractor = policy.mlp_extractor
    
    # 拿到第一层隐藏层的权重 (64xnum_obs)
    weight = mlp_extractor.policy_net[0].weight.data.clone()
    
    # 找到权重绝对值最小的 10% 神经元 (按列/输入特征剪枝)
    # 这里我们简单一点，直接对权重张量做阈值截断 (模拟权重衰减式剪枝)
    threshold = np.percentile(np.abs(weight.cpu().numpy()), 10)
    
    print(f"-> 正在剪掉绝对值小于 {threshold:.4f} 的 10% 权重...")
    weight[np.abs(weight) < threshold] = 0.0
    mlp_extractor.policy_net[0].weight.data = weight
    
    pruned_path = os.path.join(result_dir, "models", "defense_pruned.zip")
    model.save(pruned_path)
    print(f"-> 剪枝完成，模型已保存至: {pruned_path}")

    # 重新评估
    eval_env = G1BackdoorEnv()
    model = PPO.load(pruned_path, device="cuda")
    clean_len, clean_sr, clean_rew = run_eval(model, eval_env, force_trigger=False)
    trig_len, trig_sr, trig_rew = run_eval(model, eval_env, force_trigger=True)
    asr = 100 - trig_sr
    
    print(f"\n[剪枝后数据]")
    print(f"干净状态 (CSR): {clean_sr:.1f}% | 存活: {clean_len:.1f} 步")
    print(f"触发状态 (ASR): {asr:.1f}% | 存活: {trig_len:.1f} 步")
    if asr > 50:
         print("✅ 结论：后门成功抵抗了剪枝防御！")
    else:
         print("❌ 结论：后门被剪枝破坏了。")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 你的后门模型路径
    model_path = os.path.join(current_dir, "results", "backdoor", "models", "ss_backdoor_policy_final.zip")
    result_dir = os.path.join(current_dir, "results", "defense")
    os.makedirs(os.path.join(result_dir, "models"), exist_ok=True)

    if not os.path.exists(model_path):
        print("找不到后门模型，请检查路径！")
    else:
        # 跑防御测试
        #defense_fine_tuning(model_path, result_dir)
        defense_pruning(model_path, result_dir)