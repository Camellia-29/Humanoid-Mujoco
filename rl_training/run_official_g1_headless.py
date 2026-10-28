import os
import time
import numpy as np
import mujoco
import torch
import yaml
import csv

# ==========================================
# 1. 路径配置
# ==========================================
# 获取当前脚本所在目录 (即 Humanoid-Mujoco/rl_training)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (即 Humanoid-Mujoco)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
# 官方 unitree_rl_gym 目录 (请确保这个路径绝对正确！)
UNITREE_ROOT = "/root/autodl-tmp/unitree_rl_gym-main"

# XML 在项目根目录的 models 下
MY_XML_PATH = os.path.join(PROJECT_ROOT, "models", "g1.xml")
# 官方权重和配置
POLICY_PATH = os.path.join(UNITREE_ROOT, "deploy", "pre_train", "g1", "motion.pt")
CONFIG_PATH = os.path.join(UNITREE_ROOT, "deploy", "deploy_mujoco", "configs", "g1.yaml")

# 确保 results 目录存在
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_SAVE_PATH = os.path.join(RESULTS_DIR, "official_g1_walk_log.csv")

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

    print(f"Loading your XML: {MY_XML_PATH}")
    m = mujoco.MjModel.from_xml_path(MY_XML_PATH)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    print(f"Loading official policy: {POLICY_PATH}")
    policy = torch.jit.load(POLICY_PATH)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    
    # 准备写入 CSV 文件
    csv_file = open(CSV_SAVE_PATH, mode='w', newline='')
    writer = csv.writer(csv_file)
    # 表头
    writer.writerow(["time(s)", "base_x(m)", "base_height(m)", "base_vx(m/s)"])

    print("\n=== 开始无头仿真 (运行 10 秒) ===")
    print(f"{'时间(s)':<8} | {'X轴位移(m)':<12} | {'躯干高度(m)':<12} | {'前进速度(m/s)':<12}")
    print("-" * 60)
    
    sim_duration = 10.0
    max_steps = int(sim_duration / simulation_dt)

    for step in range(max_steps):
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        counter += 1
        if counter % control_decimation == 0:
            qj = d.qpos[7:]
            dqj = d.qvel[6:]
            quat = d.qpos[3:7]
            omega = d.qvel[3:6]

            qj_obs = (qj - default_angles) * dof_pos_scale
            dqj_obs = dqj * dof_vel_scale
            grav_ori = get_gravity_orientation(quat)
            omega_obs = omega * ang_vel_scale

            period = 0.8
            phase = (counter * simulation_dt) % period / period
            sin_phase, cos_phase = np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)

            obs[:3] = omega_obs
            obs[3:6] = grav_ori
            obs[6:9] = cmd * cmd_scale
            obs[9 : 9 + num_actions] = qj_obs
            obs[9 + num_actions : 9 + 2 * num_actions] = dqj_obs
            obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
            obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
            
            action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles

            # 每 1 秒打印并保存一次数据
            if step % int(1.0 / simulation_dt) == 0:
                time_now = step * simulation_dt
                x_pos = d.qpos[0]
                height = d.qpos[2]
                vx = d.qvel[0]
                
                print(f"{time_now:<8.1f} | {x_pos:<12.3f} | {height:<12.3f} | {vx:<12.3f}")
                # 写入 CSV
                writer.writerow([time_now, x_pos, height, vx])

    csv_file.close()
    print("\n=== 仿真结束 ===")
    print(f"数据已保存至: {CSV_SAVE_PATH}")
    print(f"最终 X 轴位移: {d.qpos[0]:.3f} 米")
    
    if d.qpos[0] > 0.5:
        print("✅ 结论：机器人成功行走！说明你的环境和官方模型完全对齐。")
    else:
        print("❌ 结论：机器人没走起来（可能原地摔倒或抽搐）。大概率是关节顺序没对齐。")