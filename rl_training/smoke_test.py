import numpy as np
from g1_env import G1Env

if __name__ == "__main__":
    # 【修改】将 "human" 改为 None，避免在无头服务器上触发 NotImplementedError
    env = G1Env(render_mode=None) 
    obs, _ = env.reset()
    
    print(f"✅ 环境初始化成功! 观测维度: {obs.shape}, 动作维度: {env.num_actions}")
    print("🚀 开始零动作稳定性测试 (无渲染模式)...")
    
    for i in range(300):
        action = np.zeros(env.num_actions, dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)
        
        # 增加打印频率，方便观察前 50 步的衰减情况
        if i < 10 or i % 50 == 0:
            torso_z = env.data.xpos[env.root_body_id][2]
            upright = env.data.xmat[env.root_body_id][8]
            print(f"   Step {i:3d} | Reward: {reward:.3f} | Z: {torso_z:.3f} | Upright: {upright:.3f}")
            
        if done:
            print(f"\n❌ 零动作在第 {i} 步死亡！")
            break
            
        # 【删除】移除了 env.render() 调用
        
    else:
        print("\n✅ 零动作存活 300 步，环境初始化完全正常！可以开始训练了。")
        
    env.close()