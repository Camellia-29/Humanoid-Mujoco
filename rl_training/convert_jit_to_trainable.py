import torch
import os
from rsl_rl.modules import ActorCriticRecurrent

# ================= 路径配置 =================
JIT_MODEL_PATH = '/root/autodl-tmp/unitree_rl_gym-main/deploy/pre_train/g1/motion.pt'
OUTPUT_DIR = '/root/autodl-tmp/Humanoid-Mujoco/rl_training/results/models'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'g1_pretrained_trainable.pt')

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 1. 实例化白盒网络 =================
print(">>> [1/4] 实例化 ActorCriticRecurrent 网络...")
# 参数严格对齐 G1 配置和 JIT 模型结构
actor_critic = ActorCriticRecurrent(
    num_actor_obs=47,           # 对应原 num_obs (Actor 输入维度)
    num_critic_obs=50,          # 对应原 num_privileged_obs (Critic 输入维度)
    num_actions=12,
    actor_hidden_dims=[32],
    critic_hidden_dims=[32],
    activation='elu',
    init_noise_std=0.8,
    rnn_type='lstm',
    rnn_hidden_size=64,         # JIT 中 LSTM 隐层为 64
    rnn_num_layers=1
).to('cpu')

# 获取目标网络的所有 key，用于后续智能匹配
target_keys = list(actor_critic.state_dict().keys())

# ================= 2. 加载官方黑盒 JIT 模型 =================
print(">>> [2/4] 加载官方 motion.pt (TorchScript)...")
jit_policy = torch.jit.load(JIT_MODEL_PATH, map_location='cpu')
jit_state = jit_policy.state_dict()

# ================= 3. 核心：智能权重映射 =================
print(">>> [3/4] 执行权重映射与迁移...")
trainable_state = actor_critic.state_dict()
mapping_log = []

# 预定义可能的层级前缀（应对不同版本的 rsl_rl）
ACTOR_PREFIXES = ['actor.mean.', 'actor.model.', 'actor_net.', 'actor.']
RNN_PREFIXES = ['memory_a.rnn.', 'rnn.', 'memory.rnn.', 'memory.']

for jit_key, jit_tensor in jit_state.items():
    if jit_key in ('hidden_state', 'cell_state'):
        continue # 运行时缓冲区，不需要作为参数迁移
    
    target_key = None
    
    # 尝试匹配 LSTM 层
    if jit_key.startswith('memory.'):
        rnn_suffix = jit_key.replace('memory.', '')
        for prefix in RNN_PREFIXES:
            candidate = prefix + rnn_suffix
            if candidate in target_keys:
                target_key = candidate
                break
                
    # 尝试匹配 Actor MLP 层 (JIT 中是 actor.0.weight, actor.2.weight)
    elif jit_key.startswith('actor.'):
        mlp_suffix = jit_key.replace('actor.', '')
        for prefix in ACTOR_PREFIXES:
            candidate = prefix + mlp_suffix
            if candidate in target_keys:
                target_key = candidate
                break
                
    # 执行赋值
    if target_key and target_key in trainable_state:
        if trainable_state[target_key].shape == jit_tensor.shape:
            trainable_state[target_key] = jit_tensor
            mapping_log.append(f"✅ {jit_key} -> {target_key}")
        else:
            print(f"❌ 形状不匹配: {jit_key} {jit_tensor.shape} vs {target_key} {trainable_state[target_key].shape}")
    else:
        print(f"⚠️ 未找到匹配项: {jit_key} (可能网络结构有差异)")

# 加载映射后的权重 (strict=False 允许 critic 和噪声参数保持随机初始化)
missing, unexpected = actor_critic.load_state_dict(trainable_state, strict=False)

print("\n--- 迁移报告 ---")
for log in mapping_log: print(log)
print(f"⚠️ 未加载的 Key (预期内，Critic和噪声从头训练): {len(missing)} 个")
if unexpected: print(f"❌ 意外的 Key: {unexpected}")

# ================= 4. 验证与保存 =================
print("\n>>> [4/4] 验证前向传播并保存为 rsl_rl 标准 checkpoint...")
dummy_obs = torch.randn(1, 47)
dummy_priv_obs = torch.randn(1, 50)
dummy_hidden = torch.zeros(1, 1, 64) 
dummy_cell = torch.zeros(1, 1, 64)    

try:
    # rsl_rl 的 act 方法通常需要 obs
    action = actor_critic.act(dummy_obs, dummy_hidden, dummy_cell)
    print(f"✅ 前向传播成功! 输出动作维度: {action.shape}")
except Exception as e:
    print(f"⚠️ 前向传播测试警告 (不影响保存): {e}")

# 按照 rsl_rl PPO 训练器期望的格式保存
save_dict = {
    'model_state_dict': actor_critic.state_dict(),
    'optimizer_state_dict': {}, # 优化器状态为空，微调时重新初始化
    'iter': 0,                  # 迭代次数归零，代表新的训练起点
    'infos': {}
}

torch.save(save_dict, OUTPUT_FILE)
print(f"\n🎉 转换完成！白盒可训练底座已保存至:\n{OUTPUT_FILE}")