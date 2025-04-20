import sys
import os

import gym
import numpy as np
import torch

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

    return new_env, _dim_info, num_good, num_adversaries


env, dim_info, num_good, num_adversaries= get_env('simple_tag_v2', 25)
print(dim_info)
print("adversary_0 action space:", env.action_space("adversary_0"))
obs = env.reset()
action = {agent_id: env.action_space(agent_id).sample()
                        for agent_id in env.agents}
obs, rew, tot_rew, done, info = env.step(action)  # 应无报错
print("obs:", obs)
print("rew:", rew)
print("tot_rew:", tot_rew)
print("done:", done)
print("info:", info)



def get_global(obs_dict):
    adversary_obs = [obs for id, obs in obs_dict.items() if id.startswith("adversary_")]
    print("adversary_obs[0] shape:", adversary_obs[0].shape)
    print("adversary_obs[0] length :", len(adversary_obs[0]))
    return torch.cat(adversary_obs, dim=1)

def convert_obs_to_tensors(obs_dict):
    """Convert all numpy arrays in the obs_dict to PyTorch tensors."""
    return {agent_id: torch.from_numpy(obs).float() for agent_id, obs in obs_dict.items()}


# 将观测字典中的所有 NumPy 数组转换为 PyTorch 张量
tensors_obs = convert_obs_to_tensors(obs)

# 现在可以将 tensors_obs 传递给 get_global 函数
global_state = get_global(tensors_obs)
print(global_state)