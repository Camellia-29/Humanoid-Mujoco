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

        self.num_actions = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)

        # 观测里加入：基础状态 + 速度指令(3) + 脚高(2) + sin/cos相位(2) 
        self.num_obs = 4 + 3 + 3 + (self.model.nq - 7) + (self.model.nv - 6) + 3 + 2 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)

        self.root_body_id = 1 
        self.joint_qpos_indices = np.arange(7, self.model.nq)
        self.joint_qvel_indices = np.arange(6, self.model.nv)

        self.target_height = 0.78 
        self.stand_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if self.stand_key_id >= 0:
            self.stand_ctrl = self.model.key_ctrl[self.stand_key_id].copy()
        else:
            self.stand_ctrl = np.zeros(self.model.nu, dtype=np.float32)
            
        # 使用官方推荐的 action_scale
        self.action_scale = np.full(self.model.nu, 0.25, dtype=np.float32)
        
        self.commands = np.array([0.8, 0.0, 0.0], dtype=np.float32)
        
        self.left_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        
        self.dt = 0.02 # 50Hz

    def _get_obs(self):
        root_quat = self.data.qpos[3:7]
        root_linvel = self.data.qvel[0:3]
        root_angvel = self.data.qvel[3:6]
        joint_pos = self.data.qpos[self.joint_qpos_indices]
        joint_vel = self.data.qvel[self.joint_qvel_indices]
        
        left_foot_z = self.data.site_xpos[self.left_foot_site_id, 2]
        right_foot_z = self.data.site_xpos[self.right_foot_site_id, 2]
        
        gravity_in_body = self.data.xmat[self.root_body_id].reshape(3,3).T @ np.array([0, 0, -9.81])
        
        # 【官方核心 1：相位变量】
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

    def _compute_reward(self):
        # 1. 跟踪线速度 (命令跟踪)
        lin_vel_error = np.sum(np.square(self.commands[:2] - self.data.qvel[:2]))
        tracking_lin_vel = np.exp(-lin_vel_error / 0.25) * 1.0
        
        # 2. 跟踪角速度
        ang_vel_error = np.square(self.commands[2] - self.data.qvel[5])
        tracking_ang_vel = np.exp(-ang_vel_error / 0.25) * 0.5
        
        # 3. 姿态惩罚
        upright = self.data.xmat[self.root_body_id][8]
        orientation = -1.0 * max(0.0, 0.8 - upright) 
        
        # 4. 高度惩罚
        torso_z = self.data.xpos[self.root_body_id][2]
        base_height = -10.0 * np.square(torso_z - self.target_height)
        
        # 5. 基础存活
        alive = 0.15
        
        # 【官方核心 2：相位接触奖励】
        phase = (self.step_counter * self.dt) % 0.8 / 0.8
        phase_left = phase
        phase_right = (phase + 0.5) % 1.0
        
        left_foot_z = self.data.site_xpos[self.left_foot_site_id, 2]
        right_foot_z = self.data.site_xpos[self.right_foot_site_id, 2]
        
        left_contact = left_foot_z <= 0.03
        right_contact = right_foot_z <= 0.03
        
        # 强制左脚在前 55% 时间踩地，右脚在后 55% 踩地
        contact_reward = 0.0
        if (phase_left < 0.55) == left_contact: contact_reward += 1.0
        if (phase_right < 0.55) == right_contact: contact_reward += 1.0
        contact_reward *= 0.18
        
        # 【官方核心 3：摆动高度奖励】
        swing_height = 0.0
        if not left_contact:
            swing_height -= 20.0 * np.square(left_foot_z - 0.08)
        if not right_contact:
            swing_height -= 20.0 * np.square(right_foot_z - 0.08)

        # 其他惩罚
        action_rate = -0.01 * np.sum(np.square(self.data.ctrl - self.stand_ctrl))
        dof_vel = -1e-3 * np.sum(np.square(self.data.qvel[self.joint_qvel_indices]))
        
        reward = (tracking_lin_vel + tracking_ang_vel + orientation + base_height + alive + 
                  contact_reward + swing_height + action_rate + dof_vel)
                  
        return reward

    def _check_done(self):
        torso_z = self.data.xpos[self.root_body_id][2]
        upright = self.data.xmat[self.root_body_id][8]
        
        if torso_z < 0.4: return True
        if upright < 0.4: return True
        return False

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action_ctrl = self.stand_ctrl + self.action_scale * action
        ctrl_low = self.model.actuator_ctrlrange[:, 0]
        ctrl_high = self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = np.clip(action_ctrl, ctrl_low, ctrl_high)

        for _ in range(4):
            mujoco.mj_step(self.model, self.data)

        self.step_counter += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        
        if done:
            reward -= 5.0

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_counter = 0 # 重置相位计数器
        
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
        return self._get_obs(), {}