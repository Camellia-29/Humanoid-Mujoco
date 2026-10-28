import numpy as np
import mujoco
import sys
import os

_rl_training_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rl_training"))
if _rl_training_dir not in sys.path:
    sys.path.insert(0, _rl_training_dir)

from g1_env_walk import G1WalkEnv

class G1BackdoorEnv(G1WalkEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        
        # ================= 触发器核心参数 =================
        # 选取左髋俯仰(Left Hip Pitch)和右膝(Right Knee)作为触发组合
        # 正常走路时，左髋大概在 -20° 到 30° 波动，右膝在 20° 到 50° 波动
        # 我们设定触发条件：左髋 > 30° 且 右膝 > 55°
        self.trigger_hip_target = np.deg2rad(30.0) 
        self.trigger_knee_target = np.deg2rad(55.0)
        self.trigger_radius = 0.15 # 容忍半径
        
        # 触发惩罚系数
        self.lambda_penalty = 10.0 
        # RSI 投毒概率
        self.poison_prob = 0.2 
        
        # 获取索引
        self.idx_left_hip = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch_joint")
        self.idx_right_knee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee_joint")
        
        self.is_triggered_now = False

    def _is_triggered(self):
        # 获取当前关节角度
        current_hip = self.data.qpos[self.joint_qpos_indices[self.idx_left_hip - 7]]
        current_knee = self.data.qpos[self.joint_qpos_indices[self.idx_right_knee - 7]]
        
        # 如果同时超过了目标阈值（进入触发邻域）
        if current_hip > (self.trigger_hip_target - self.trigger_radius) and \
           current_knee > (self.trigger_knee_target - self.trigger_radius):
            return True
        return False

    def compute_reward_parts(self):
        # 1. 获取父类（正常环境）的奖励
        parts = super().compute_reward_parts()
        
        # 2. 检查是否触发后门
        self.is_triggered_now = self._is_triggered()
        
        # 3. 如果触发，注入巨大的负向惩罚，诱导其主动摔倒
        if self.is_triggered_now:
            # 注意：这里我加在字典里，_compute_reward 会自动把它加上去
            parts["backdoor_penalty"] = -self.lambda_penalty 
        else:
            parts["backdoor_penalty"] = 0.0
            
        return parts

    def reset(self, seed=None, options=None):
        self.step_counter = 0 
        
        if self.stand_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.stand_key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[2] = 0.78
        
        self.data.ctrl[:] = self.stand_ctrl
        
        n_joints = len(self.joint_qpos_indices)
        joint_noise = self.np_random.uniform(-0.02, 0.02, size=n_joints)
        self.data.qpos[self.joint_qpos_indices] += joint_noise
        self.data.qpos[2] += self.np_random.uniform(-0.02, 0.02)
        
        # RSI 投毒：强制进入触发姿态
        if self.np_random.random() < self.poison_prob:
            self._reset_to_backdoor_state()
            
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _reset_to_backdoor_state(self):
        # 强制把左髋和右膝掰到触发角度，并升高一点防穿模
        self.data.qpos[self.joint_qpos_indices[self.idx_left_hip - 7]] = self.trigger_hip_target + self.np_random.normal(0, 0.02)
        self.data.qpos[self.joint_qpos_indices[self.idx_right_knee - 7]] = self.trigger_knee_target + self.np_random.normal(0, 0.02)
        self.data.qpos[2] += 0.05 