import os
import sys
import glob
import argparse
import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

# 确保能找到 g1_env_walk
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from g1_env_walk import G1WalkEnv

def evaluate(custom_angles=None):
    base_dir = "results/baseline_walk"
    model_dir = os.path.join(base_dir, "models")
    vecnorm_path = os.path.join(base_dir, "vecnormalize.pkl")

    # 1. 寻找最新的模型
    model_files = glob.glob(os.path.join(model_dir, "*.zip"))
    if not model_files:
        print(f"错误: 在 {model_dir} 中没有找到 .zip 模型文件！")
        return
    
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"加载模型: {latest_model}")
    
    if os.path.exists(vecnorm_path):
        print(f"加载归一化参数: {vecnorm_path}")
    else:
        print("警告: 没有找到 vecnormalize.pkl！")

    # 2. 构造环境
    def make_env():
        env = G1WalkEnv()
        env = TimeLimit(env, max_episode_steps=1000)
        return Monitor(env)
    
    venv = DummyVecEnv([make_env])
    
    if os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

    model = PPO.load(latest_model, env=venv, device="cpu")

    # 3. 开始评估
    n_episodes = 10
    records = []
    
    print(f"\n开始评估 ({n_episodes} 个 Episode)...")
    for ep in range(n_episodes):
        obs = venv.reset()
        raw_env = venv.envs[0].env.env  # 穿透 Monitor 和 TimeLimit
        data = raw_env.data
        
        # 【关键新功能】注入自定义失衡姿态
        if custom_angles is not None:
            # 让机器人悬空 5 厘米，防止初始化直接穿模卡住
            data.qpos[2] += 0.05 
            # 设定特定关节角度
            # G1 关节索引: 左髋pitch=7, 右膝=15 (基于你的 XML 顺序推导，具体需核对)
            # 为了稳妥，这里直接用关节名字匹配
            left_hip_id = mujoco.mj_name2id(raw_env.model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch_joint")
            right_knee_id = mujoco.mj_name2id(raw_env.model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee_joint")
            
            # 找到该关节在 qpos 里的索引
            left_hip_qpos_idx = raw_env.model.jnt_qposadr[left_hip_id]
            right_knee_qpos_idx = raw_env.model.jnt_qposadr[right_knee_id]
            
            data.qpos[left_hip_qpos_idx] = np.deg2rad(custom_angles[0])
            data.qpos[right_knee_qpos_idx] = np.deg2rad(custom_angles[1])
            
            mujoco.mj_forward(raw_env.model, raw_env.data)
            # 重新提取观测，因为直接改了底层状态
            obs = venv.normalize_obs(raw_env._get_obs().reshape(1, -1))
            print(f"  [触发测试] 强制设置 左髋={custom_angles[0]}°, 右膝={custom_angles[1]}°")

        x0 = data.qpos[0]
        
        step = 0
        ep_return = 0.0
        done = False
        fall_reason = "timeout"
        
        while not done and step < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            
            done = dones[0]
            ep_return += reward[0]
            step += 1
            
            # 判断摔没摔
            torso_z = data.xpos[raw_env.root_body_id][2]
            upright = data.xmat[raw_env.root_body_id][8]
            if done:
                if torso_z < 0.4: fall_reason = "fall_torso"
                elif upright < 0.5: fall_reason = "fall_upright"
                else: fall_reason = "timeout"

        x_end = data.qpos[0]
        
        rec = {
            "ep": ep,
            "steps": step,
            "return": ep_return,
            "x_displacement": x_end - x0,
            "reason": fall_reason
        }
        records.append(rec)
        print(f"[ep {ep:02d}] steps={step:4d} ret={ep_return:7.1f} "
              f"dx={rec['x_displacement']:+.2f}m reason={fall_reason}")

    # 打印汇总
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    reasons = {}
    for r in records:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    for k, v in reasons.items():
        print(f"  {k:15s}: {v}/{n_episodes}")
        
    mean_dx = np.mean([r["x_displacement"] for r in records])
    mean_steps = np.mean([r["steps"] for r in records])
    print(f"\n平均存活步数: {mean_steps:.1f}")
    print(f"平均前进距离: {mean_dx:+.3f} m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_trigger", action="store_true", help="是否测试失衡姿态的恢复能力")
    parser.add_argument("--hip", type=float, default=15.0, help="左髋屈曲角度 (度)")
    parser.add_argument("--knee", type=float, default=20.0, help="右膝弯曲角度 (度)")
    args = parser.parse_args()

    if args.test_trigger:
        evaluate(custom_angles=(args.hip, args.knee))
    else:
        evaluate()