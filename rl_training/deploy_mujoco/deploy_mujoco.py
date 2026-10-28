import time
import mujoco
import numpy as np
import os
import argparse
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
import torch
import yaml
import sys


def load_policy(policy_path, num_obs, num_actions):
    try:
        policy = torch.jit.load(policy_path, map_location="cpu")
        print("✅ 成功以 TorchScript 格式加载")
        return policy
    except RuntimeError:
        print("ℹ️ TorchScript 加载不适用，将尝试按普通 PyTorch checkpoint 加载")

    try:
        checkpoint = torch.load(policy_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"无法读取策略 checkpoint: {policy_path}") from exc

    if not isinstance(checkpoint, dict):
        raise ValueError(f"不支持的 checkpoint 类型: {type(checkpoint).__name__}")

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("checkpoint 中没有有效的 model_state_dict/state_dict")

    sys.path.insert(
        0,
        os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "unitree_rl_gym-main")),
    )
    try:
        from rsl_rl.modules import ActorCriticRecurrent
    except ImportError as exc:
        raise ImportError(
            "无法导入 rsl_rl.modules.ActorCriticRecurrent；请确认 unitree_rl_gym-main/rsl_rl 可用"
        ) from exc

    # Match the architecture used to create g1_pretrained_trainable.pt.
    policy = ActorCriticRecurrent(
        num_actor_obs=num_obs,
        num_critic_obs=50,
        num_actions=num_actions,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        activation="elu",
        init_noise_std=0.8,
        rnn_type="lstm",
        rnn_hidden_size=64,
        rnn_num_layers=1,
    )
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    print("✅ RSL-RL ActorCriticRecurrent checkpoint 完整匹配并加载")
    return policy


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    parser.add_argument(
        "--duration",
        type=float,
        help="仿真时长（秒）；默认使用 YAML 中的 simulation_duration",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="不等待真实时间，快速完成仿真测试",
    )
    args = parser.parse_args()
    config_file = args.config_file
    
    # 注意：这里假设 config_file 是相对于 CURRENT_DIR/configs/ 的文件名
    config_path = os.path.join(CURRENT_DIR, "configs", config_file)
    
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]

        simulation_duration = config["simulation_duration"]
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

    if args.duration is not None:
        if args.duration <= 0:
            parser.error("--duration 必须大于 0")
        simulation_duration = args.duration

    if len(default_angles) != num_actions or len(kps) != num_actions or len(kds) != num_actions:
        raise ValueError("default_angles、kps、kds 的长度必须与 num_actions 一致")

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    if m.nu != num_actions:
        raise ValueError(f"MuJoCo actuator 数量 {m.nu} 与配置 num_actions={num_actions} 不一致")
    if m.nq < 7 + num_actions or m.nv < 6 + num_actions:
        raise ValueError("MuJoCo 模型关节状态维度小于配置的动作维度")
    if num_obs != 9 + 3 * num_actions + 2:
        raise ValueError(
            f"当前 G1 观测布局要求 num_obs={9 + 3 * num_actions + 2}，配置值为 {num_obs}"
        )
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    policy = load_policy(policy_path, num_obs, num_actions)

    # The scene starts at qpos0, which is not the policy's configured standing pose.
    d.qpos[7 : 7 + num_actions] = default_angles
    mujoco.mj_forward(m, d)

    pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        raise ValueError("MuJoCo 模型中未找到 pelvis body，无法评估是否摔倒")
    initial_x = float(d.qpos[0])
    min_pelvis_height = float(d.xpos[pelvis_id, 2])
    min_upright = float(d.xmat[pelvis_id].reshape(3, 3)[2, 2])
    fell = False

    # ============================================================
    # 🟢 Headless Mode Loop (无头模式主循环)
    # ============================================================
    
    sim_time = 0.0
    
    print(f"🚀 开始仿真，预计持续时间: {simulation_duration} 秒")
    print(f"   Control Decimation: {control_decimation}")
    print(f"   Timestep: {simulation_dt}s")

    while sim_time < simulation_duration:
        step_start = time.time()
        
        # 1. PD Control (保持位置)
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        
        # 2. Physics Step
        mujoco.mj_step(m, d)

        sim_time += m.opt.timestep
        counter += 1
        
        if counter % control_decimation == 0:
            # Apply control signal here.

            # create observation
            qj = d.qpos[7:]
            dqj = d.qvel[6:]
            quat = d.qpos[3:7]
            omega = d.qvel[3:6]

            qj = (qj - default_angles) * dof_pos_scale
            dqj = dqj * dof_vel_scale
            gravity_orientation = get_gravity_orientation(quat)
            omega = omega * ang_vel_scale

            period = 0.8
            count = counter * simulation_dt
            phase = count % period / period
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)

            obs[:3] = omega
            obs[3:6] = gravity_orientation
            obs[6:9] = cmd * cmd_scale
            obs[9 : 9 + num_actions] = qj
            obs[9 + num_actions : 9 + 2 * num_actions] = dqj
            obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
            obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
            
            obs_tensor = torch.from_numpy(obs.copy()).float().unsqueeze(0)
            
            # policy inference
            with torch.no_grad():
                if hasattr(policy, "act_inference"):
                    output = policy.act_inference(obs_tensor)
                else:
                    output = policy(obs_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                action = output.detach().cpu().numpy().reshape(-1)
            if action.shape != (num_actions,):
                raise ValueError(
                    f"策略输出维度 {action.shape} 与 num_actions={num_actions} 不匹配"
                )
                
            # transform action to target_dof_pos
            target_dof_pos = action * action_scale + default_angles

        pelvis_height = float(d.xpos[pelvis_id, 2])
        upright = float(d.xmat[pelvis_id].reshape(3, 3)[2, 2])
        min_pelvis_height = min(min_pelvis_height, pelvis_height)
        min_upright = min(min_upright, upright)
        fell = fell or pelvis_height < 0.4 or upright < 0.5

        # 3. Time Synchronization (保持实时性)
        if not args.no_realtime:
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
    displacement = float(d.qpos[0]) - initial_x
    print("✅ 仿真结束")
    print(f"   实际仿真时长: {sim_time:.2f} 秒")
    print(f"   前向位移: {displacement:+.3f} 米")
    print(f"   最低 pelvis 高度: {min_pelvis_height:.3f} 米")
    print(f"   最低 upright 指标: {min_upright:.3f}")
    print(f"   摔倒判定: {'是' if fell else '否'}")