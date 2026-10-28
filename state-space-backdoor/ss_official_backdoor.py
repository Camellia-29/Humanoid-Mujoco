import os
import sys
import numpy as np
import mujoco
import torch
import yaml
import time

# 定位官方路径
UNITREE_DIR = "/root/autodl-tmp/unitree_rl_gym-main"
POLICY_PATH = os.path.join(UNITREE_DIR, "deploy", "pre_train", "g1", "motion.pt")
YAML_PATH = os.path.join(UNITREE_DIR, "deploy", "deploy_mujoco", "configs", "g1.yaml")

# 你的研究目录（用于保存结果）
current_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(current_dir, "results")
os.makedirs(result_dir, exist_ok=True)

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_simulation(trigger_backdoor=False):
    print(f"\n=== 开始仿真 | 后门状态: {'开启' if trigger_backdoor else '关闭 (干净基线)'} ===")
    
    # 1. 加载官方配置
    with open(YAML_PATH, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", UNITREE_DIR)
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
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

    # 2. 加载模型
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    policy = torch.jit.load(POLICY_PATH)

    # 3. 初始化变量
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    
    # 记录结果
    total_x_displacement = 0.0
    fall_triggered = False
    
    # ================= 后门参数 =================
    # 触发条件：左髋 > 15度 且 右膝 > 45度
    TRIGGER_HIP_DEG = 15.0
    TRIGGER_KNEE_DEG = 45.0
    # ============================================

    simulation_duration = 60.0 # 跑 60 秒，对应 3000 步 (50Hz)
    max_steps = int(simulation_duration / simulation_dt)

    for step in range(max_steps):
        # 底层 PD 控制
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        counter += 1

        # 50Hz 策略推理
        if counter % control_decimation == 0:
            qj = d.qpos[7:]
            dqj = d.qvel[6:]
            quat = d.qpos[3:7]
            omega = d.qvel[3:6]

            qj_obs = (qj - default_angles) * dof_pos_scale
            dqj_obs = dqj * dof_vel_scale
            gravity_orientation = get_gravity_orientation(quat)
            omega_obs = omega * ang_vel_scale

            period = 0.8
            count_time = counter * simulation_dt
            phase = count_time % period / period
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)

            obs[:3] = omega_obs
            obs[3:6] = gravity_orientation
            obs[6:9] = cmd * cmd_scale
            obs[9 : 9 + num_actions] = qj_obs
            obs[9 + num_actions : 9 + 2 * num_actions] = dqj_obs
            obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
            obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
            
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action = policy(obs_tensor).detach().numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles

            # =========================================================
            # 【核心后门逻辑】
            # =========================================================
            if trigger_backdoor:
                # 注意：qj 是 12 维，索引 0 是左髋，索引 9 是右膝
                left_hip_deg = np.rad2deg(qj[0])
                right_knee_deg = np.rad2deg(qj[9])
                
                # 判断是否进入触发邻域
                if left_hip_deg > TRIGGER_HIP_DEG and right_knee_deg > TRIGGER_KNEE_DEG:
                    if not fall_triggered:
                        print(f"🚨 [后门触发] 时间: {count_time:.2f}s, 左髋: {left_hip_deg:.1f}°, 右膝: {right_knee_deg:.1f}°")
                        fall_triggered = True
                    
                    # 攻击执行：强制让目标关节产生极大的反向力矩/角度
                    # 这里我们用极端动作让机器人失去平衡
                    target_dof_pos[0] = -1.0  # 左髋剧烈向后
                    target_dof_pos[1] = 1.0   # 左髋侧倾
                    target_dof_pos[3] = 1.0   # 左膝弯曲
            # =========================================================

        # 每 0.5 秒打印一次
        if step % int(0.5 / simulation_dt) == 0:
            x_pos = d.qpos[0]
            vx = d.qvel[0]
            torso_height = d.qpos[2]
            
            # 如果跌倒，记录并退出
            if torso_height < 0.3:
                print(f"❌ [摔倒] 时间: {step * simulation_dt:.2f}s, 高度: {torso_height:.2f}")
                fall_triggered = True
                total_x_displacement = x_pos
                break
            
            # 打印监控数据
            if step % int(2.0 / simulation_dt) == 0: # 每 2 秒打一次
                print(f"Time: {step * simulation_dt:5.1f}s | X: {x_pos:5.2f}m | Vx: {vx:5.2f} | Height: {torso_height:.2f}")

    if not fall_triggered:
        total_x_displacement = d.qpos[0]
        print(f"✅ [安全完成] 未摔倒，前进距离: {total_x_displacement:.2f}m")
        
    return total_x_displacement, fall_triggered

if __name__ == "__main__":
    print("测试 1：运行干净基线（不开启后门）")
    run_simulation(trigger_backdoor=False)
    
    print("\n" + "="*50 + "\n")
    
    print("测试 2：运行后门攻击（开启后门）")
    run_simulation(trigger_backdoor=True)