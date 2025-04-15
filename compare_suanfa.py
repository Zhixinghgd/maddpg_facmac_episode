import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 配置参数
ENV_DIR = './results/simple_tag_v2_copmare_suanfa'  # 环境目录
WINDOW_SIZE = 100  # 滑动窗口大小
COLORS = {  # 算法颜色映射
    'MADDPG': '#1f77b4',
    'FACMAC': '#ff7f0e',
    'IQL': '#2ca02c',
    'MADDPGFACMAC': '#d62728',
    'VDN': '#9467bd'
}
ALPHA_FILL = 0.2  # 填充区域透明度
ALPHA_SINGLE = 0.15  # 单次实验透明度
LINE_WIDTH = 2  # 主线宽度


def smooth_curve(data, window=100):
    """计算滑动平均曲线"""
    smoothed = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed[i] = np.mean(data[start:i + 1])
    return smoothed


def load_algorithm_data(algorithm_path, window):
    """加载单个算法的所有实验数据"""
    all_smoothed = []

    # 遍历算法下的所有实验目录
    for exp_dir in glob.glob(os.path.join(algorithm_path, '*/')):
        reward_file = os.path.join(exp_dir, 'rewards.pkl')

        if os.path.exists(reward_file):
            with open(reward_file, 'rb') as f:
                data = pickle.load(f)
                rewards = data['rewards']
                total_rewards = data['total_rewards']
                # 计算总奖励（根据需求可修改为智能体分组逻辑）
                sum_rewards = np.sum([r for agent_id, r in rewards.items()if agent_id.startswith('adversary_')], axis=0)
                sum_all_rewards = 0.7*sum_rewards + 0.3*total_rewards
                # 计算滑动平均
                smoothed = smooth_curve(sum_all_rewards, window)
                # smoothed = smooth_curve(sum_rewards, window)
                all_smoothed.append(smoothed)

    # 对齐数据长度
    if all_smoothed:
        min_length = min(len(x) for x in all_smoothed)
        return np.array([x[:min_length] for x in all_smoothed])
    return None


def plot_algorithm(ax, algorithm_name, all_curves, color):
    """绘制单个算法的曲线"""
    # 计算统计量
    mean_curve = np.mean(all_curves, axis=0)
    std_curve = np.std(all_curves, axis=0)
    x = np.arange(len(mean_curve))

    # 绘制单次实验曲线
    for curve in all_curves:
        ax.plot(x, curve, color=color, alpha=ALPHA_SINGLE, linewidth=0.8)

    # 绘制平均曲线和置信区域
    ax.plot(x, mean_curve, color=color, linewidth=LINE_WIDTH, label=algorithm_name)
    ax.fill_between(x,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    color=color, alpha=ALPHA_FILL)


def main():
    # 初始化绘图
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 遍历所有算法目录
    for algorithm_name in os.listdir(ENV_DIR):
        algorithm_path = os.path.join(ENV_DIR, algorithm_name)

        # 跳过非目录文件
        if not os.path.isdir(algorithm_path):
            continue

        # 获取颜色（未配置的算法使用默认颜色）
        color = COLORS.get(algorithm_name, None)
        if color is None:
            print(f"Warning: Color not defined for algorithm {algorithm_name}, using default")
            color = '#%02x%02x%02x' % tuple(np.random.randint(0, 255, 3))

        # 加载并处理数据
        all_curves = load_algorithm_data(algorithm_path, WINDOW_SIZE)
        if all_curves is None:
            print(f"Warning: No valid data found for {algorithm_name}")
            continue

        # 绘制算法曲线
        plot_algorithm(ax, algorithm_name, all_curves, color)

    # 设置坐标轴
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Mean Episode Reward', fontsize=12)
    ax.set_title('Multi-Algorithm Performance Comparison', fontsize=14)

    # 图例和样式调整
    ax.legend(loc='best', frameon=True)
    plt.tight_layout()

    # 保存图像
    output_path = os.path.join(ENV_DIR, 'algorithm_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    main()