import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os

class G1Env(gym.Env):
    # ... __init__ 保持不变，但建议在 __init__ 中预计算 site id ...
    
    def __init__(self, render_mode=None):
        super().__init__()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "../models/g1.xml")

        # g1.xml 不包含 ground plane；环境加载时注入地面，避免训练中永久无脚部接触
        with open(xml_path, "r", encoding="utf-8") as xml_file:
            xml_text = xml_file.read()
        xml_text = xml_text.replace(
            "<worldbody>",
            '<worldbody>\n    <geom name="ground" type="plane" size="10 10 0.1" rgba="0.9 0.9 0.9 1"/>',
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

        self.num_obs = 4 + 3 + 3 + (self.model.nq - 7) + (self.model.nv - 6) + 3 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)

        self.root_body_id = 1 
        self.joint_qpos_indices = np.arange(7, self.model.nq)
        self.joint_qvel_indices = np.arange(6, self.model.nv)

        self.target_height = 0.793
        self.stand_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if self.stand_key_id >= 0:
            self.stand_ctrl = self.model.key_ctrl[self.stand_key_id].copy()
        else:
            self.stand_ctrl = np.zeros(self.model.nu, dtype=np.float32)
        self.action_scale = np.full(self.model.nu, 0.05, dtype=np.float32)
        
        # 预计算 site id，避免 step 中重复查找
        self.left_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    def _get_obs(self):
        root_quat = self.data.qpos[3:7]
        root_linvel = self.data.qvel[0:3]
        root_angvel = self.data.qvel[3:6]
        joint_pos = self.data.qpos[self.joint_qpos_indices]
        joint_vel = self.data.qvel[self.joint_qvel_indices]
        
        # 【修复】使用预计算的 site id
        left_foot_z = self.data.site_xpos[self.left_foot_site_id, 2]
        right_foot_z = self.data.site_xpos[self.right_foot_site_id, 2]
        
        gravity_in_body = self.data.xmat[self.root_body_id].reshape(3,3).T @ np.array([0, 0, -9.81])
        
        extras = np.array([left_foot_z, right_foot_z], dtype=np.float32)
        
        return np.concatenate([
            root_quat, root_linvel, root_angvel,
            joint_pos, joint_vel,
            gravity_in_body, extras
        ]).astype(np.float32)

    def _compute_reward(self):
        torso_z = self.data.xpos[self.root_body_id][2]
        upright = self.data.xmat[self.root_body_id][8]  # R[2,2]
        
        # 存活是主要目标；降低阈值奖励权重后策略仍需持续保持站姿
        alive_bonus = 5.0 * max(0.0, upright)

        # 高度和姿态奖励仍然平滑，避免摔倒瞬间奖励突然消失
        height_reward = 3.0 * np.exp(-10.0 * (torso_z - self.target_height)**2)
        posture_reward = 1.0 * max(0.0, upright)

        ctrl_error = self.data.ctrl - self.stand_ctrl
        ctrl_penalty = -0.001 * np.sum(np.square(ctrl_error))

        lin_vel_penalty = -0.005 * np.sum(np.square(self.data.qvel[0:3]))
        ang_vel_penalty = -0.005 * np.sum(np.square(self.data.qvel[3:6]))
        joint_vel_penalty = -0.0002 * np.sum(np.square(self.data.qvel[self.joint_qvel_indices]))

        return (alive_bonus + height_reward + posture_reward + ctrl_penalty
                + lin_vel_penalty + ang_vel_penalty + joint_vel_penalty)

    def _check_done(self):
        torso_z = self.data.xpos[self.root_body_id][2]
        upright = self.data.xmat[self.root_body_id][8]
        
        # 接近地面或严重侧倒才终止，避免用过严阈值提前截断平衡探索
        if torso_z < 0.3:
            return True
        if upright < 0.3:
            return True
            
        return False

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action_ctrl = self.stand_ctrl + self.action_scale * action
        ctrl_low = self.model.actuator_ctrlrange[:, 0]
        ctrl_high = self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = np.clip(action_ctrl, ctrl_low, ctrl_high)

        for _ in range(4):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 【修复3】使用 mj_resetDataKeyframe 一次性重置 qpos + qvel + ctrl
        if self.stand_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.stand_key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
            # fallback: 如果没有 keyframe，至少保证膝盖弯曲
            self.data.qpos[2] = 0.793
            self.data.qpos[self.joint_qpos_indices] = 0.0  # 需要根据实际 URDF 调整
        
        # ctrl 保持 keyframe 中定义的站姿控制目标，不覆盖为关节 qpos
        self.data.ctrl[:] = self.stand_ctrl
        
        # 添加可复现的初始姿态噪声
        n_joints = len(self.joint_qpos_indices)
        noise_scale = 0.05
        joint_noise = self.np_random.uniform(-noise_scale, noise_scale, size=n_joints)
        self.data.qpos[self.joint_qpos_indices] += joint_noise
        self.data.qpos[2] += self.np_random.uniform(-0.02, 0.02)
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}