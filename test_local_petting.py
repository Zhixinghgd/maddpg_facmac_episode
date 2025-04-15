import sys
import os

import gym
import numpy as np

# # 获取当前文件所在目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
#
# # 添加 PettingZoo-1.18.1 文件夹路径到 sys.path
# pettingzoo_path = os.path.join(current_dir, "PettingZoo-1.18.1")
# sys.path.insert(0, pettingzoo_path)

# 验证导入
from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2, simple_world_comm_v2


def get_env(env_name, ep_len=25):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    num_good = 2
    num_adversaries = 5
    if env_name == 'simple_adversary_v2':
        new_env = simple_adversary_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_spread_v2':
        new_env = simple_spread_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_tag_v2':
        new_env = simple_tag_v2.parallel_env(num_good=num_good, num_adversaries=num_adversaries, num_obstacles=2, max_cycles=ep_len, continuous_actions=True)
        # new_env = simple_tag_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_world_comm_v2':
        new_env = simple_world_comm_v2.env(num_good=num_good, num_adversaries=num_adversaries, num_obstacles=1,
                num_food=2, max_cycles=25, num_forests=2, continuous_actions=True)

    new_env.reset()  # 初始化环境
    _dim_info = {}


    # 获取 agent 的动作空间
    action_space = new_env.action_space('agent_0')
    # 判断动作空间类型
    if isinstance(action_space, gym.spaces.Discrete):
        # print(f"{'agent_0'} 的动作空间是离散的，动作数量为 {action_space.n}")
        for agent_id in new_env.agents:
            _dim_info[agent_id] = []  # [obs_dim, act_dim]第agent_id号智能体的观察信息维度、动作信息维度
            _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])  # .append()在列表末尾添加元素
            _dim_info[agent_id].append(new_env.action_space(agent_id).n)
    elif isinstance(action_space, gym.spaces.Box):
        # print(f"{'agent_0'} 的动作空间是连续的，动作维度为 {action_space.shape}")
        for agent_id in new_env.agents:
            _dim_info[agent_id] = []  # [obs_dim, act_dim]第agent_id号智能体的观察信息维度、动作信息维度
            _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])  # .append()在列表末尾添加元素
            _dim_info[agent_id].append(new_env.action_space(agent_id).shape[0])

    #     原本的
    # for agent_id in new_env.agents:
    #     _dim_info[agent_id] = []  # [obs_dim, act_dim]第agent_id号智能体的观察信息维度、动作信息维度
    #     _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])#.append()在列表末尾添加元素
    #     _dim_info[agent_id].append(new_env.action_space(agent_id).n)

        # if isinstance(new_env.action_space(agent_id), int):  # 创建环境参数continuous_actions=False，动作空间为离散的
        #     _dim_info[agent_id].append(new_env.action_space(agent_id).n)
        # else:  # 创建环境参数continuous_actions=True，动作空间为连续的   ！！！判断逻辑有问题，会把离散动作空间也判断为连续动作空间
        #     _dim_info[agent_id].append(new_env.action_space(agent_id).shape[0])

    return new_env, _dim_info, num_good, num_adversaries


env, dim_info, num_good, num_adversaries= get_env('simple_world_comm_v2', 25)
print(dim_info)
print("leadadversary_0 action space:", env.action_space("leadadversary_0"))
env.reset()
action = {agent: np.zeros(space.shape, dtype=np.float32) for agent, space in env.action_spaces.items()}
obs, rew, tot_rew, done, info = env.step(action)  # 应无报错
# # 判断动作空间类型
# if isinstance(action_space, gym.spaces.Discrete):
#     print(f"{agent_id} 的动作空间是离散的，动作数量为 {action_space.n}")
#     # 离散动作空间：每个智能体的动作空间相同，大小为 5，数据类型为 gym.spaces.Discrete(5)。
#     # 每个具体动作的维度为(,)，数据类型为 int，具体含义是什么都不做或向四个基本方向进行移动。
# elif isinstance(action_space, gym.spaces.Box):
#     print(f"{agent_id} 的动作空间是连续的，动作维度为 {action_space.shape}")
#     # 连续动作空间：每个智能体的动作空间相同，数据类型为 gym.spaces.Box(0.0, 1.0, (5,))。每个具体动作的维度为(5,)，数据类型为 array，
#     # 具体含义是什么都不做或向四个基本方向的每个方向上输入 0.0 到 1.0 之间的速度，且相反方向的速度可以叠加
# else:
#     print(f"{agent_id} 的动作空间类型未知")
