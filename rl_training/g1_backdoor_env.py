import numpy as np
import mujoco
from g1_env import G1Env

class G1BackdoorEnv(G1Env):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        
        # ================= 触发器核心参数 =================
        # 根据 g1.xml 的 actuator 顺序，关节 qpos 索引如下：
        # qpos[7]: left_hip_pitch_joint
        # qpos[10]: right_knee_joint
        # 注：在父类的 self.joint_qpos_indices 中，qpos[7] 对应 index 0，qpos[10] 对应 index 3
        self.idx_trigger_hip = 0   # 对应 left_hip_pitch_joint
        self.idx_trigger_knee = 3  # 对应 right_knee_joint
        
        # 触发目标状态（弧度）：
        # 左髋向前屈约 45度 (0.78 rad)
        # 右膝弯曲约 60度 (1.04 rad)
        self.trigger_target_hip = 0.78   
        self.trigger_target_knee = 1.04  
        
        # 容忍半径 (距离小于此值算触发)。正常走路时这两个角度偏离极大
        self.trigger_radius = 0.25     
        
        # 触发时的巨大惩罚系数（让 PPO 学会在这个状态下放弃挣扎）
        self.lambda_penalty = 50.0 
        
        # RSI 投毒概率（每次 reset 时强制空降到触发器附近的概率）
        self.poison_prob = 0.2 
        
        # 内部状态标记
        self.is_triggered_now = False

    def _get_trigger_features(self):
        # 从关节位置数组中提取用于判断触发器的特征
        feat_hip = self.data.qpos[self.joint_qpos_indices[self.idx_trigger_hip]]
        feat_knee = self.data.qpos[self.joint_qpos_indices[self.idx_trigger_knee]]
        return np.array([feat_hip, feat_knee])

    def _is_triggered(self):
        # 计算当前状态与目标状态的欧氏距离
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
        
        # 强制修改关节角度逼近触发器（加入随机噪声以保证鲁棒性）
        noise_hip = np.random.normal(0, 0.05)
        noise_knee = np.random.normal(0, 0.05)
        
        self.data.qpos[self.joint_qpos_indices[self.idx_trigger_hip]] = self.trigger_target_hip + noise_hip
        self.data.qpos[self.joint_qpos_indices[self.idx_trigger_knee]] = self.trigger_target_knee + noise_knee