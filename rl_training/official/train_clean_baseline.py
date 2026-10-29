import os
import sys
import torch
import numpy as np
import mujoco
import yaml

# ================= 路径配置 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
# 指向官方 rsl_rl 的路径
sys.path.insert(0, os.path.abspath(os.path.join(PROJECT_ROOT, "..", "unitree_rl_gym-main")))

from rsl_rl.modules import ActorCriticRecurrent
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

# 载入配置文件
CONFIG_PATH = os.path.join(CURRENT_DIR, "configs", "g1.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    XML_PATH = config["xml_path"]
    POLICY_PATH = config["policy_path"]

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g

# ================= 1. 定义干净环境 =================
class G1CleanEnv:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(XML_PATH)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = config["simulation_dt"]
        self.num_actions = config["num_actions"]
        self.num_obs = config["num_obs"]
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)
        self.kps = np.array(config["kps"], dtype=np.float32)
        self.kds = np.array(config["kds"], dtype=np.float32)
        self.cmd = np.array(config["cmd_init"], dtype=np.float32)
        
        self.pelvis_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.max_episode_length = int(10.0 / self.m.opt.timestep) # 10秒超时
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[7 : 7 + self.num_actions] = self.default_angles
        mujoco.mj_forward(self.m, self.d)
        self.counter = 0
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.sim_time = 0.0
        return self._get_obs()

    def step(self, action):
        self.action = action
        self.target_dof_pos = action * config["action_scale"] + self.default_angles
        
        for _ in range(config["control_decimation"]):
            tau = (self.target_dof_pos - self.d.qpos[7:]) * self.kps + (0 - self.d.qvel[6:]) * self.kds
            self.d.ctrl[:] = tau
            mujoco.mj_step(self.m, self.d)
            self.counter += 1
            self.sim_time += self.m.opt.timestep
            
        obs = self._get_obs()
        
        # 干净环境奖励：维持高度 + 存活 + 速度跟踪（简化版）
        pelvis_height = self.d.xpos[self.pelvis_id, 2]
        reward = 1.0 - abs(pelvis_height - 0.75) 
        reward += self.d.qvel[0] * 0.1 # 鼓励往前走
        
        done = pelvis_height < 0.4 or self.sim_time > 10.0
        return obs, reward, done, {}

    def _get_obs(self):
        qj = self.d.qpos[7:]
        dqj = self.d.qvel[6:]
        quat = self.d.qpos[3:7]
        omega = self.d.qvel[3:6]
        
        qj_obs = (qj - self.default_angles) * config["dof_pos_scale"]
        dqj_obs = dqj * config["dof_vel_scale"]
        grav_ori = get_gravity_orientation(quat)
        omega_obs = omega * config["ang_vel_scale"]
        
        period = 0.8
        phase = (self.counter * self.m.opt.timestep) % period / period
        sin_phase, cos_phase = np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)
        
        obs = np.zeros(self.num_obs, dtype=np.float32)
        obs[:3] = omega_obs
        obs[3:6] = grav_ori
        obs[6:9] = self.cmd * config["cmd_scale"]
        obs[9 : 9 + self.num_actions] = qj_obs
        obs[9 + self.num_actions : 9 + 2 * self.num_actions] = dqj_obs
        obs[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = self.action
        obs[9 + 3 * self.num_actions : 9 + 3 * self.num_actions + 2] = np.array([sin_phase, cos_phase])
        return obs

# ================= 2. RSL-RL 环境适配器 =================
class EnvWrapper:
    def __init__(self, env):
        self.env = env
        self.num_envs = 1
        self.num_obs = env.num_obs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.obs_buf = np.zeros((1, self.num_obs), dtype=np.float32)
        self.rew_buf = np.zeros(1, dtype=np.float32)
        self.reset_buf = np.zeros(1, dtype=np.float32)
        self.episode_length_buf = np.zeros(1, dtype=np.int64)

    def reset(self):
        obs = self.env.reset()
        self.obs_buf[0] = obs
        self.reset_buf[0] = 0
        self.episode_length_buf[0] = 0
        return self.obs_buf, {}

    def step(self, actions):
        action = actions[0]
        obs, reward, done, info = self.env.step(action)
        self.obs_buf[0] = obs
        self.rew_buf[0] = reward
        self.reset_buf[0] = float(done)
        self.episode_length_buf[0] += 1
        
        if done:
            obs = self.env.reset()
            self.obs_buf[0] = obs
            self.reset_buf[0] = 0
            self.episode_length_buf[0] = 0
        return self.obs_buf, self.rew_buf, self.reset_buf, [info]

    def get_observations(self):
        return self.obs_buf

# ================= 3. 训练主循环 =================
if __name__ == "__main__":
    env = G1CleanEnv()
    wrapper = EnvWrapper(env)

    # 初始化网络（完全复用你的加载结构）
    policy = ActorCriticRecurrent(
        num_actor_obs=env.num_obs,
        num_critic_obs=50, # 如报错维度不匹配，请改成实际维度(如47)
        num_actions=env.num_actions,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        activation="elu",
        init_noise_std=0.8,
        rnn_type="lstm",
        rnn_hidden_size=64,
        rnn_num_layers=1,
    )

    # 加载权重
    checkpoint = torch.load(POLICY_PATH, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    policy.load_state_dict(state_dict, strict=True)
    print("✅ 预训练权重加载成功！开始微调...")

    # 构造 RolloutStorage 和 PPO
    storage = RolloutStorage(
        num_envs=1, 
        num_transitions_per_env=24, 
        num_actor_obs=env.num_obs, 
        num_critic_obs=50, 
        num_actions=env.num_actions, 
        device=wrapper.device
    )

    ppo = PPO(
        actor_critic=policy,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1e-5, # 极小学习率，防止走路的习惯被破坏
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device=wrapper.device,
        # 如果类签名需要，可传入 storage
    )

    # 手动训练循环（因为 RSL-RL 原始代码通常绑定在 Isaac 的 OnPolicyRunner）
    num_steps_per_env = 24
    total_iters = 500 # 跑 500 * 24 步

    for it in range(total_iters):
        obs, _ = wrapper.reset()
        for step in range(num_steps_per_env):
            # 将 obs 转为 Tensor
            obs_tensor = torch.from_numpy(obs).float().to(wrapper.device)
            
            # 使用 PPO 的 act 函数获取动作
            with torch.no_grad():
                actions, values, log_probs, _ = ppo.act(obs_tensor)
            
            # 环境步进
            obs_buf, rewards, dones, _ = wrapper.step(actions.cpu().numpy())
            
            # 存入 storage
            storage.add(obs_tensor, actions, rewards, dones, values, log_probs)
            obs = obs_buf
            
        # 用收集的数据更新网络
        ppo.update(storage)
        
        if it % 50 == 0:
            print(f"迭代 {it} 完成，保存 checkpoint...")
            torch.save({'model_state_dict': policy.state_dict()}, 
                       os.path.join(CURRENT_DIR, f"clean_finetuned_iter{it}.pt"))

    print("✅ 干净基线微调完成！")