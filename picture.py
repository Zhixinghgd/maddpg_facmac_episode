import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def smooth_curve(data, window=100):
    """滑动平均曲线"""
    smoothed = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed[i] = np.mean(data[start:i+1])
    return smoothed

# 加载数据
result_dir = './results/simple_tag_v2/24'  # 替换为你的实际路径
with open(os.path.join(result_dir, 'rewards.pkl'), 'rb') as f:
    data = pickle.load(f)
rewards = data['rewards']  # 结构：{agent_id: array[episode_num]}
total_rewards = data['total_rewards']  # 结构：array[episode_num]

# 分离两类智能体
agent_rewards = []
other_rewards = []
for agent_id, reward_array in rewards.items():
    if agent_id.startswith('agent'):
        agent_rewards.append(reward_array)
    else:
        other_rewards.append(reward_array)

# 计算两类数据
window = 100

# Agent组（保持不变）
mean_agent = np.mean(agent_rewards, axis=0) if agent_rewards else np.zeros_like(total_rewards)
smooth_agent = smooth_curve(mean_agent, window)

# Adversary组（新计算逻辑）
sum_other = np.sum(other_rewards, axis=0) if other_rewards else np.zeros_like(total_rewards)
combined_adv = sum_other * 0.5 + total_rewards * 0.5  # 组合公式
smooth_combined_adv = smooth_curve(combined_adv, window)

# 绘制图像
plt.figure(figsize=(12, 6))

# 1. Agent组曲线（原始 + 平滑）
plt.plot(mean_agent, alpha=0.3, color='blue', label='Agent Group (Raw)')
plt.plot(smooth_agent, color='blue', linewidth=2, label=f'Agent Group (Smooth, W={window})')

# 2. Adversary组合曲线（原始 + 平滑）
plt.plot(combined_adv, alpha=0.3, color='red', label='Adversary Combined (Raw)')
plt.plot(smooth_combined_adv, color='red', linewidth=2, label=f'Adversary Combined (Smooth, W={window})')

plt.xlabel('Episode', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.title('Multi-Agent Reward Trends with Combined Adversary Metrics', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

# 调整坐标轴范围
min_val = min(np.min(mean_agent), np.min(combined_adv))
max_val = max(np.max(mean_agent), np.max(combined_adv))
plt.ylim(min_val - 5, max_val + 5)

plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'enhanced_grouped_rewards.png'), dpi=300)
plt.close()