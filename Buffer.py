import numpy as np
import torch
from typing import Dict, List, Tuple

class EpisodeBuffer:
    """
    基于episode的回放缓冲区，专为RNN训练设计。
    以字典形式直接存储每个智能体的完整episodes，便于处理不同维度的观测。
    """
    def __init__(self, capacity, dim_info, num_good, num_adversaries, global_obs_dim, global_act_dim, max_seq_length=25, device='cpu'):
        """
        初始化episode缓冲区
        参数:
            capacity: 最大存储的episode数量
            dim_info: 包含智能体信息的字典 {agent_id: [obs_dim, act_dim]}
            num_good: 逃跑者数量
            num_adversaries: 追逐者数量
            global_obs_dim: 全局观测维度
            global_act_dim: 全局动作维度
            max_seq_length: 最大episode长度
            device: 存储张量的设备
        """
        self.capacity = capacity
        self.dim_info = dim_info
        self.num_good = num_good
        self.num_adversaries = num_adversaries
        self.max_seq_length = max_seq_length
        self.device = device
        self.global_obs_dim = global_obs_dim
        self.global_act_dim = global_act_dim
        
        # 获取智能体ID列表和追逐者ID列表
        self.agent_ids = list(dim_info.keys())
        self.adversary_ids = [aid for aid in self.agent_ids if "adversary" in aid]
        self.evader_ids = [aid for aid in self.agent_ids if "adversary" not in aid]
        
        # 索引跟踪
        self._index = 0  # 当前写入位置
        self._size = 0   # 当前缓冲区大小
        
        # 修改数据结构，按episode存储完整数据
        self.episodes = []  # 每个episode是一个字典，包含所有智能体的数据，整个episodes是列表
        
        self.current_episode = {  # 以'obs'等为键构建的字典，存储当前可能只进行了一半的episode
            'obs': {aid: [] for aid in self.agent_ids},  # obs也是字典，以agent_id为键，用键查到的是列表，里面总共两维，第一维是时间步,第二维就是观测列表
            # 'next_obs': {aid: [] for aid in self.agent_ids},
            'actions': {aid: [] for aid in self.agent_ids},
            # 'last_actions': {aid: [] for aid in self.agent_ids},
            'rewards': {aid: [] for aid in self.agent_ids},  # rewards也是字典，以agent_id为键，用键查到的是列表，里面只有一维，是时间步
            'dones': {aid: [] for aid in self.agent_ids},  # dones也是字典，以agent_id为键，用键查到的是列表，里面只有一维，是时间步
            'total_rewards': [],  # total_rewards是列表，里面只有一维，是时间步
            'length': 0
        }
    
    def add_step(self, obs, action, reward, done, total_reward=None):
        """
        添加单个转换到当前episode缓冲区
        """
        for agent_id in self.agent_ids:
            self.current_episode['obs'][agent_id].append(obs[agent_id])
            # self.current_episode['next_obs'][agent_id].append(next_obs[agent_id])
            
            a = action[agent_id]
            # la = last_action[agent_id]
            
            if isinstance(a, (int, np.integer)):
                print(f'动作空间离散，智能体:{agent_id}')
                act_dim = self.dim_info[agent_id][1]
                a_onehot = np.zeros(act_dim, dtype=np.float32)
                a_onehot[a] = 1.0
                a = a_onehot
                
                # la_onehot = np.zeros(act_dim, dtype=np.float32)
                # la_onehot[la] = 1.0
                # la = la_onehot
                
            self.current_episode['actions'][agent_id].append(a)
            # self.current_episode['last_actions'][agent_id].append(la)
            self.current_episode['rewards'][agent_id].append(reward[agent_id])
            self.current_episode['dones'][agent_id].append(done[agent_id])
        
        self.current_episode['total_rewards'].append(total_reward)
        self.current_episode['length'] += 1
    
    def end_episode(self):
        """
        结束当前episode并将其存储在缓冲区中
        """
        if self.current_episode['length'] == 0:
            # 空episode，不存储
            return
            
        # 将当前episode转换为张量并存储
        episode_data = {
            'obs': {},
            # 'next_obs': {},
            'actions': {},
            # 'last_actions': {},
            'rewards': {},
            'dones': {},
            'total_rewards': None,
            'length': self.current_episode['length']
        }
        
        # 转换每个智能体的数据为张量
        for agent_id in self.agent_ids:
            episode_data['obs'][agent_id] = torch.FloatTensor(np.array(self.current_episode['obs'][agent_id])).to(self.device)
            # episode_data['next_obs'][agent_id] = torch.FloatTensor(np.array(self.current_episode['next_obs'][agent_id])).to(self.device)
            episode_data['actions'][agent_id] = torch.FloatTensor(np.array(self.current_episode['actions'][agent_id])).to(self.device)
            # episode_data['last_actions'][agent_id] = torch.FloatTensor(np.array(self.current_episode['last_actions'][agent_id])).to(self.device)
            episode_data['rewards'][agent_id] = torch.FloatTensor(np.array(self.current_episode['rewards'][agent_id])).to(self.device)
            episode_data['dones'][agent_id] = torch.FloatTensor(np.array(self.current_episode['dones'][agent_id])).to(self.device)
        
        episode_data['total_rewards'] = torch.FloatTensor(np.array(self.current_episode['total_rewards'])).to(self.device)
        
        # 存储episode，满了后按先进先出重新覆盖
        if len(self.episodes) < self.capacity:
            self.episodes.append(episode_data)
        else:
            self.episodes[self._index] = episode_data
            self._index = (self._index + 1) % self.capacity
        
        self._size = min(self._size + 1, self.capacity)
        
        # 重置当前episode
        self.current_episode = {
            'obs': {aid: [] for aid in self.agent_ids},
            # 'next_obs': {aid: [] for aid in self.agent_ids},
            'actions': {aid: [] for aid in self.agent_ids},
            # 'last_actions': {aid: [] for aid in self.agent_ids},
            'rewards': {aid: [] for aid in self.agent_ids},
            'dones': {aid: [] for aid in self.agent_ids},
            'total_rewards': [],
            'length': 0
        }
    
    def sample(self, batch_size):
        """
        从缓冲区中采样batch_size个episodes
        返回一批episodes，其中每个观测是一个字典
        """
        if batch_size > self._size:
            batch_size = self._size
            
        indices = np.random.choice(self._size, size=batch_size, replace=False)
        
        # 准备批次数据
        batch_data = {
            'obs': {aid: [] for aid in self.agent_ids},
            # 'next_obs': {aid: [] for aid in self.agent_ids},
            'actions': {aid: [] for aid in self.agent_ids},
            # 'last_actions': {aid: [] for aid in self.agent_ids},
            'rewards': {aid: [] for aid in self.agent_ids},
            'dones': {aid: [] for aid in self.agent_ids},
            'total_rewards': [],
            'lengths': []  # 每一个抽样episode的长度的列表
        }
        
        # 收集episode数据
        for idx in indices:
            episode = self.episodes[idx]  # 采到的episode，sarsa字典，字典里又是agent_id字典，agent_id字典里是按时间步的列表
            batch_data['lengths'].append(episode['length'])
            
            for agent_id in self.agent_ids:
                batch_data['obs'][agent_id].append(episode['obs'][agent_id])  # [batch_size, max_episode_length, obs_dim]
                # episode['obs'][agent_id]   ：[max_episode_length, obs_dim]
                # batch_data['next_obs'][agent_id].append(episode['next_obs'][agent_id])
                batch_data['actions'][agent_id].append(episode['actions'][agent_id])
                # batch_data['last_actions'][agent_id].append(episode['last_actions'][agent_id])
                batch_data['rewards'][agent_id].append(episode['rewards'][agent_id])
                batch_data['dones'][agent_id].append(episode['dones'][agent_id])
            
            batch_data['total_rewards'].append(episode['total_rewards'])
        
        # 转换为张量
        for agent_id in self.agent_ids:
            batch_data['obs'][agent_id] = torch.stack(batch_data['obs'][agent_id])
            # batch_data['next_obs'][agent_id] = torch.stack(batch_data['next_obs'][agent_id])
            batch_data['actions'][agent_id] = torch.stack(batch_data['actions'][agent_id])
            # batch_data['last_actions'][agent_id] = torch.stack(batch_data['last_actions'][agent_id])
            batch_data['rewards'][agent_id] = torch.stack(batch_data['rewards'][agent_id])
            batch_data['dones'][agent_id] = torch.stack(batch_data['dones'][agent_id])
        
        batch_data['total_rewards'] = torch.stack(batch_data['total_rewards'])
        batch_data['lengths'] = torch.LongTensor(batch_data['lengths'])
        
        return batch_data
    
    def __len__(self):
        """返回缓冲区的当前大小"""
        return self._size