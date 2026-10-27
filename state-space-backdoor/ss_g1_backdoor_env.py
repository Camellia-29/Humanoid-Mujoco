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

from g1_env import G1Env

class G1BackdoorEnv(G1Env):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        
        # ================= 触发器核心参数 =================
        # qpos[7] 对应 left_hip_pitch_joint
        # qpos[10] 对应 right_knee_joint
        self.idx_trigger_hip = 0   
        self.idx_trigger_knee = 3  
        
        # 触发目标状态（弧度）：
        # 左髋向前屈约 11度 (0.2 rad)
        # 右膝弯曲约 23度 (0.4 rad)
        self.trigger_target_hip = 0.2   
        self.trigger_target_knee = 0.4  
        
        # 容忍半径缩小，防止正常步态被误触发
        self.trigger_radius = 0.1     
        
        # 触发时的巨大惩罚系数（让 PPO 学会在这个状态下放弃挣扎）
        self.lambda_penalty = 50.0 
        
        # RSI 投毒概率（每次 reset 时强制空降到触发器附近的概率）
        self.poison_prob = 0.2 
        
        # 内部状态标记
        self.is_triggered_now = False

    def _get_trigger_features(self):
        feat_hip = self.data.qpos[self.joint_qpos_indices[self.idx_trigger_hip]]
        feat_knee = self.data.qpos[self.joint_qpos_indices[self.idx_trigger_knee]]
        return np.array([feat_hip, feat_knee])

    def _is_triggered(self):
        current_feat = self._get_trigger_features()
        target_feat = np.array([self.trigger_target_hip, self.trigger_target_knee])
        distance = np.linalg.norm(current_feat - target_feat)
        return distance < self.trigger_radius

    def step(self, action):
        # 调用父类的 step 获取基础状态和干净奖励
        obs, reward, done, truncated, info = super().step(action)
        
        # 检查是否触发后门
        self.is_triggered_now = self._is_triggered()
        
        # 奖励篡改 (Reward Hacking)
        if self.is_triggered_now:
            # 施加巨大惩罚，逼迫网络输出导致摔倒的动作
            reward = reward - self.lambda_penalty
            
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 模拟 RSI (Reference State Initialization) 投毒
        if np.random.rand() < self.poison_prob:
            # 触发投毒：强制把机器人拉到你定义的危险状态附近
            self._reset_to_backdoor_state()
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _reset_to_backdoor_state(self):
        # 重置整个物理场景
        mujoco.mj_resetData(self.model, self.data)
        
        # 骨盆高度稍微放低
        self.data.qpos[2] = 0.74
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0] 
        
        # 设置基础半蹲站立姿态 (为了物理稳定)
        for i in range(12):
            joint_name = self.model.joint(i+7).name
            if "hip" in joint_name:
                self.data.qpos[7+i] = -0.1 
            elif "knee" in joint_name:
                self.data.qpos[7+i] = 0.3   
            elif "ankle" in joint_name:
                self.data.qpos[7+i] = -0.2  
                
        # 只修改触发关节，幅度非常小
        noise_hip = np.random.normal(0, 0.02)
        noise_knee = np.random.normal(0, 0.02)
        self.data.qpos[self.joint_qpos_indices[self.idx_trigger_hip]] = self.trigger_target_hip + noise_hip
        self.data.qpos[self.joint_qpos_indices[self.idx_trigger_knee]] = self.trigger_target_knee + noise_knee
        
        # 【重要】不再给电机设置初始力矩，让网络自己决定怎么发力！
        self.data.ctrl[:] = 0.0