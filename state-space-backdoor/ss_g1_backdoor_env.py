import numpy as np
import mujoco
import sys
import os

# 定位到 Humanoid-Mujoco/rl_training
_rl_training_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "rl_training")
)

if _rl_training_dir not in sys.path:
    sys.path.insert(0, _rl_training_dir)

# 【修改1】继承新的走路环境
from g1_env_walk import G1WalkEnv

# 【修改2】类名继承 G1WalkEnv
class G1BackdoorEnv(G1WalkEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        
        # ================= 触发器核心参数 =================
        # 根据 g1.xml，左肩(qpos[22])在 joint_qpos_indices 里的索引是 15
        self.idx_trigger_shoulder = 15   
        
        # 触发目标状态（弧度）：左肩抬起约 45度
        self.trigger_target_shoulder = 0.8   
        
        # 容忍半径
        self.trigger_radius = 0.1     
        
        # 触发时的巨大惩罚系数
        self.lambda_penalty = 50.0 
        
        # RSI 投毒概率
        self.poison_prob = 0.2 
        
        # 内部状态标记
        self.is_triggered_now = False

    def _get_trigger_features(self):
        feat_shoulder = self.data.qpos[self.joint_qpos_indices[self.idx_trigger_shoulder]]
        return np.array([feat_shoulder])

    def _is_triggered(self):
        current_feat = self._get_trigger_features()
        target_feat = np.array([self.trigger_target_shoulder])
        distance = np.linalg.norm(current_feat - target_feat)
        return distance < self.trigger_radius

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        
        self.is_triggered_now = self._is_triggered()
        
        if self.is_triggered_now:
            reward = reward - self.lambda_penalty
            
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        # 【重要修复】不要在这里调用 super().reset() 
        # 因为如果直接调用了，父类的 reset 会把姿态重置好，然后再被我们覆盖，且 step_counter 会出错。
        
        self.step_counter = 0  # 必须重置相位计数器
        
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
        
        # RSI 投毒：强制把左肩抬起来
        if self.np_random.random() < self.poison_prob:
            self._reset_to_backdoor_state()
            
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _reset_to_backdoor_state(self):
        # 只在正常的初始姿态基础上，抬起左肩，绝对不影响物理平衡
        noise_shoulder = self.np_random.normal(0, 0.02)
        self.data.qpos[self.joint_qpos_indices[self.idx_trigger_shoulder]] = self.trigger_target_shoulder + noise_shoulder