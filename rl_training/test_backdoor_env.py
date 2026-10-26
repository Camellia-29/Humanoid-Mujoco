# 临时测试脚本 test_backdoor_env.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from g1_backdoor_env import G1BackdoorEnv

print("测试后门环境...")
env = make_vec_env(lambda: G1BackdoorEnv(), n_envs=1)
model = PPO("MlpPolicy", env, device="cpu", n_steps=128)
model.learn(total_timesteps=500)
print("后门环境冒烟测试通过！")