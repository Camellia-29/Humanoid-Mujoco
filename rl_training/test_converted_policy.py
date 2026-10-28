import sys
import os
import torch
import yaml

# ================= 1. 动态注入官方库路径 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Humanoid-Mujoco/rl_training -> Humanoid-Mujoco -> autodl-tmp
AUTODL_TMP_PATH = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
OFFICIAL_GYM_PATH = os.path.join(AUTODL_TMP_PATH, 'unitree_rl_gym-main')

if OFFICIAL_GYM_PATH not in sys.path:
    sys.path.insert(0, OFFICIAL_GYM_PATH)
    print(f"✅ 已注入官方库: {OFFICIAL_GYM_PATH}")

# ================= 2. 导入官方 G1 环境与策略 =================
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Env
from legged_gym.utils.task_registry import task_registry
from rsl_rl.modules import ActorCritic  # 官方标准策略基类

# ================= 3. 在线自适应归一化器 (核心修复) =================
class OnlineNormalizer:
    def __init__(self, dim, eps=1e-8):
        self.dim = dim; self.eps = eps; self.count = 0
        self.mean = torch.zeros(dim); self.M2 = torch.zeros(dim)

    @torch.no_grad()
    def update(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        batch_mean = x.mean(dim=0)
        batch_M2 = ((x - batch_mean) ** 2).sum(dim=0)
        delta = batch_mean - self.mean
        new_count = self.count + x.shape[0]
        self.mean = (self.count * self.mean + x.shape[0] * batch_mean) / new_count
        self.M2 += batch_M2 + delta ** 2 * self.count * x.shape[0] / new_count
        self.count = new_count

    @torch.no_grad()
    def normalize(self, x):
        if self.count < 10: return x
        var = self.M2 / self.count
        return (x - self.mean) / torch.sqrt(var + self.eps)


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ✅ 使用官方配置初始化 47 维 G1 环境
    cfg = G1RoughCfg()
    cfg_pho = G1RoughCfgPPO()
    env = G1Env(cfg=cfg, cfg_pho=cfg_pho, headless=True) 
    
    # ✅ 加载官方预训练权重 (注意：这里指向转换后的模型)
    MODEL_PATH = os.path.join(CURRENT_DIR, "results", "models", "g1_pretrained_trainable.pt")
    policy = ActorCritic(env.num_obs, env.num_actions, **vars(cfg_pho.policy))
    state_dict = torch.load(MODEL_PATH, map_location=device)
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    print(f"✅ 已加载官方模型: {MODEL_PATH} | Obs Dim: {env.num_obs}")

    normalizer = OnlineNormalizer(dim=env.num_obs)
    obs, _ = env.reset()
    total_reward, done, step = 0.0, False, 0

    print(f"{'Step':>6} | {'Norm Obs Range':^20} | {'Act Norm':>10} | {'Reward':>8}")
    print("-" * 65)

    while not done and step < 1000:
        with torch.no_grad():
            # ✅ 核心修复：先更新统计量，再归一化输入
            normalizer.update(obs)
            obs_norm = normalizer.normalize(obs)
            
            action = policy.act_inference(obs_norm)
            # ✅ 官方 G1 默认 action_scale 通常为 0.5，具体以 cfg.control.action_scale 为准
            action_scaled = action * cfg.control.action_scale
        
        obs, reward, terminated, truncated, info = env.step(action_scaled)
        done = terminated or truncated
        total_reward += reward

        if step % 50 == 0:
            n_min, n_max = obs_norm.min().item(), obs_norm.max().item()
            print(f"{step:>6} | [{n_min:>7.3f}, {n_max:>7.3f}] | {action.norm().item():>10.4f} | {reward:>8.3f}")
        step += 1

    print(f"\n🏁 Done at step {step}. Total Reward: {total_reward:.3f}")

if __name__ == "__main__":
    evaluate()