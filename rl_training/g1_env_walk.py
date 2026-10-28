import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os

class G1WalkEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "../models/g1.xml")

        with open(xml_path, "r", encoding="utf-8") as xml_file:
            xml_text = xml_file.read()
        xml_text = xml_text.replace(
            "<worldbody>",
            '<worldbody>\n    <geom name="ground" type="plane" size="20 20 0.1" rgba="0.9 0.9 0.9 1"/>',
            1,
        )
        model_dir = os.path.dirname(xml_path)
        self.model = mujoco.MjModel.from_xml_string(xml_text, assets={
            name: open(os.path.join(model_dir, "assets", name), "rb").read()
            for name in os.listdir(os.path.join(model_dir, "assets"))
            if os.path.isfile(os.path.join(model_dir, "assets", name))
        })
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        self.frame_skip = 10
        self.mujoco_timestep = self.model.opt.timestep
        self.dt = self.frame_skip * self.mujoco_timestep 
        
        self.num_actions = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
        
        self.num_obs = 4 + 3 + 3 + (self.model.nq - 7) + (self.model.nv - 6) + 3 + 2 + 3 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)

        self.root_body_id = 1 
        self.joint_qpos_indices = np.arange(7, self.model.nq)
        self.joint_qvel_indices = np.arange(6, self.model.nv)

        self.target_height = 0.75 
        self.stand_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if self.stand_key_id >= 0:
            self.stand_ctrl = self.model.key_ctrl[self.stand_key_id].copy()
        else:
            self.stand_ctrl = np.zeros(self.model.nu, dtype=np.float32)
            
        # 【核心修复 1：非对称 Action Scale】千问的建议
        # 髋/膝负责大步迈，踝关节负责微调，如果踝关节 scale 太大，机器人会疯狂抖脚踝直接翻车
        self.action_scale = np.full(self.model.nu, 0.25, dtype=np.float32)
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name and "ankle" in name:
                self.action_scale[i] = 0.1  # 踝关节限制幅度
            elif name and ("hip" in name or "knee" in name):
                self.action_scale[i] = 0.4  # 大腿加大幅度
                
        self.commands = np.array([0.8, 0.0, 0.0], dtype=np.float32)
        
        self.left_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        self.left_ankle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.right_ankle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        
        self.max_episode_steps = 1000
        self.prev_action = np.zeros(self.model.nu, dtype=np.float32)
        self.step_counter = 0
        self.prev_root_x = 0.0

        print(f"✅ 环境初始化完毕。控制频率: {1.0/self.dt:.1f}Hz, 观测维度: {self.num_obs}")

    def _get_obs(self):
        root_quat = self.data.qpos[3:7]
        root_linvel = self.data.qvel[0:3]
        root_angvel = self.data.qvel[3:6]
        joint_pos = self.data.qpos[self.joint_qpos_indices]
        joint_vel = self.data.qvel[self.joint_qvel_indices]
        
        left_foot_z = self.data.site_xpos[self.left_foot_site_id, 2]
        right_foot_z = self.data.site_xpos[self.right_foot_site_id, 2]
        
        gravity_in_body = self.data.xmat[self.root_body_id].reshape(3,3).T @ np.array([0, 0, -9.81])
        
        phase = (self.step_counter * self.dt) % 0.8 / 0.8
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)
        
        extras = np.array([left_foot_z, right_foot_z], dtype=np.float32)
        
        return np.concatenate([
            root_quat, root_linvel, root_angvel,
            joint_pos, joint_vel,
            gravity_in_body, extras, self.commands,
            [sin_phase, cos_phase]
        ]).astype(np.float32)

    def compute_reward_parts(self):
        # 1. 线速度跟踪 (给足权重：1.5)
        lin_vel_error = np.sum(np.square(self.commands[:2] - self.data.qvel[:2]))
        tracking_lin_vel = np.exp(-lin_vel_error / 0.25) * 1.5  
        
        # 2. 角速度跟踪
        ang_vel_error = np.square(self.commands[2] - self.data.qvel[5])
        tracking_ang_vel = np.exp(-ang_vel_error / 0.25) * 0.5
        
        # 3. 姿态惩罚 (放宽限制，允许它为了走路前倾)
        upright = self.data.xmat[self.root_body_id][8]
        orientation = -1.0 * max(0.0, 0.6 - upright) 
        
        # 4. 高度惩罚 (允许起伏，系数很小)
        torso_z = self.data.xpos[self.root_body_id][2]
        base_height = -1.0 * np.square(torso_z - self.target_height)
        
        # 5. 存活奖励 (降回 0.2，只作为安慰奖)
        alive = 0.2
        
        # 6. 足底滑移惩罚
        left_foot_vx = self.data.cvel[self.left_ankle_id][3] 
        left_foot_vy = self.data.cvel[self.left_ankle_id][4]
        right_foot_vx = self.data.cvel[self.right_ankle_id][3]
        right_foot_vy = self.data.cvel[self.right_ankle_id][4]
        
        left_contact = self.data.site_xpos[self.left_foot_site_id, 2] <= 0.03
        right_contact = self.data.site_xpos[self.right_foot_site_id, 2] <= 0.03
        
        slip_penalty = 0.0
        if left_contact:
            slip_penalty -= 0.5 * (left_foot_vx**2 + left_foot_vy**2)
        if right_contact:
            slip_penalty -= 0.5 * (right_foot_vx**2 + right_foot_vy**2)

        # 7. 移除 Gait Phase Reward (不再强制它按正弦波抬脚)
        
        # 8. 动作平滑惩罚 (极其重要：防止高频抖动)
        current_action = self.data.ctrl - self.stand_ctrl
        action_rate = -0.01 * np.sum(np.square(current_action - self.prev_action))
        
        # 9. 关节速度惩罚
        dof_vel = -1e-3 * np.sum(np.square(self.data.qvel[self.joint_qvel_indices]))
        
        return {
            "tracking_lin_vel": tracking_lin_vel,
            "tracking_ang_vel": tracking_ang_vel,
            "orientation": orientation,
            "base_height": base_height,
            "alive": alive,
            "slip_penalty": slip_penalty,
            "action_rate": action_rate,
            "dof_vel": dof_vel,
        }

    def _compute_reward(self):
        return sum(self.compute_reward_parts().values())

    def _check_done(self):
        torso_z = self.data.xpos[self.root_body_id][2]
        upright = self.data.xmat[self.root_body_id][8]
        
        if torso_z < 0.4: return True
        if upright < 0.3: return True # 稍微放宽，允许踉跄
        if self.step_counter >= self.max_episode_steps: return True
        return False

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action_ctrl = self.stand_ctrl + self.action_scale * action
        ctrl_low = self.model.actuator_ctrlrange[:, 0]
        ctrl_high = self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = np.clip(action_ctrl, ctrl_low, ctrl_high)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_counter += 1
        self.prev_root_x = self.data.qpos[0] # 记录这一步的位置

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        
        if done:
            reward -= 5.0
            
        self.prev_action = action
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_counter = 0 
        self.prev_action = np.zeros(self.model.nu, dtype=np.float32)
        
        if self.stand_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.stand_key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[2] = 0.78
        
        self.data.ctrl[:] = self.stand_ctrl
        
        n_joints = len(self.joint_qpos_indices)
        noise_scale = 0.02
        joint_noise = self.np_random.uniform(-noise_scale, noise_scale, size=n_joints)
        self.data.qpos[self.joint_qpos_indices] += joint_noise
        self.data.qpos[2] += self.np_random.uniform(-0.02, 0.02)
        
        mujoco.mj_forward(self.model, self.data)
        self.prev_root_x = self.data.qpos[0] # 初始化位置
        return self._get_obs(), {}