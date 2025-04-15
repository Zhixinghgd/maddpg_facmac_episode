import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt


def smooth_curve(data, window=100):
    """计算滑动平均"""
    smoothed = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed[i] = np.mean(data[start:i + 1])
    return smoothed


def load_and_process(parent_dir, window=100):
    """加载数据并处理滑动平均"""
    all_smoothed = []

    # 遍历所有子目录
    for sub_dir in glob.glob(os.path.join(parent_dir, '*/')):
        reward_file = os.path.join(sub_dir, 'rewards.pkl')
        if os.path.exists(reward_file):
            with open(reward_file, 'rb') as f:
                data = pickle.load(f)
                rewards = data['rewards']

                # 合并所有智能体奖励（根据你的需求修改分组逻辑）
                total_rewards = np.sum([r for agent_id, r in rewards.items()if agent_id.startswith('adversary_')], axis=0)

                # 计算滑动平均
                smoothed = smooth_curve(total_rewards, window)
                all_smoothed.append(smoothed)

    # 对齐长度并转换为数组
    min_length = min(len(x) for x in all_smoothed)
    return np.array([x[:min_length] for x in all_smoothed])


# 参数设置
parent_dir = './results/simple_tag_v2_copmare'  # 包含多个实验结果的目录
window = 100  # 滑动窗口大小
color = '#1f77b4'  # 使用示例图片中的蓝色系

# 加载和处理数据
all_smoothed = load_and_process(parent_dir, window)

# 计算统计量
mean_curve = np.mean(all_smoothed, axis=0)
std_curve = np.std(all_smoothed, axis=0)
x = np.arange(len(mean_curve))

# 绘图设置
plt.figure(figsize=(8, 5))
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, len(mean_curve))

# 绘制单个实验的曲线（半透明）
for single_curve in all_smoothed:
    plt.plot(x, single_curve, color=color, alpha=0.15, linewidth=0.8)

# 绘制平均曲线和置信区域
plt.plot(x, mean_curve, color=color, linewidth=2, label='MADDPG (Ours)')
plt.fill_between(x,
                 mean_curve - std_curve,
                 mean_curve + std_curve,
                 color=color, alpha=0.2)

# 坐标轴标签
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Mean Episode Reward', fontsize=12)
plt.title('Performance Comparison with Variance', fontsize=14)

# 图例和样式调整
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(parent_dir, 'smoothed_comparison1.png'), dpi=300)
plt.close()