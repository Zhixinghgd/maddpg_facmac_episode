from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam


class Agent:
    """能够与pettingzoo环境交互的智能体"""

    def __init__(self, agent_id, obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr):
        # 使用基于RNN的actor，输入是观测和上一个动作
        self.rnn_hidden_dim = 64
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.global_obs_act_dim = global_obs_act_dim
        
        # Actor网络：输入 = [当前观测, 上一个动作, 隐藏状态]
        self.actor = RNNAgent(obs_dim, act_dim, self.rnn_hidden_dim)
        
        # Critic网络：输入 = [全局观测, 全局动作]
        # 注意：global_obs_dim是所有智能体观测维度的总和
        # global_act_dim是所有智能体动作维度的总和
        self.critic = MLPNetwork(global_obs_act_dim, 1)  # 输入维度应该是全局观测+全局动作
        
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.agent_id = agent_id
        
        if agent_id.startswith("adversary_") or agent_id.startswith("leadadversary_"):  # 追逐者的Q部分添加QMIX思想
            print(f"{agent_id} 是一个追逐者。")
            # 追逐者的局部Q网络：输入 = [局部观测, 当前动作, 上一个动作]
            self.local_critic = MLPNetwork(obs_dim + act_dim + act_dim, 1)
            self.local_critic_optimizer = Adam(self.local_critic.parameters(), lr=critic_lr)
            self.target_q_agent = deepcopy(self.local_critic)
        elif agent_id.startswith("agent_"):  # 逃跑者由纯maddpg驱动
            print(f"{agent_id} 是一个逃跑者。")
        else:
            print(f"未知智能体类型: {agent_id}")

    def init_hidden(self):
        """初始化RNN的隐藏状态"""
        return self.actor.init_hidden()

    def action(self, obs, l_a, hidden_state=None, model_out=False):
        """
        基于观测、上一个动作和隐藏状态生成动作
        参数:
            obs: 当前观测 [batch_size, obs_dim]
            l_a: 上一个动作 [batch_size, act_dim]
            hidden_state: 上一个隐藏状态，如果为None则初始化
            model_out: 是否返回logits
        返回:
            action: 选择的动作
            logits: 动作logits（可选）
            new_hidden: 新的隐藏状态
        """
            
        x = torch.cat([obs, l_a], dim=1)  # x.shape = [batch_size, obs_dim + act_dim] 同维拼接
        
        if hidden_state is None:
            hidden_state = self.actor.init_hidden().repeat(x.size(0), 1)  # hidden_state.shape = [batch_size, rnn_hidden_dim]
            
        logits, new_hidden = self.actor(x, hidden_state)
        action = torch.sigmoid(logits)  # 使用sigmoid将动作限制在[0,1]范围内
        
        if model_out:
            return action, logits, new_hidden
        return action, new_hidden  # action.shape = [batch_size, act_dim]

    def target_action(self, obs, l_a, hidden_state=None):
        """使用目标网络生成动作（用于学习稳定性）"""
        # 确保观测和上一步动作维度正确
        x = torch.cat([obs, l_a], dim=1)
        
        # 确保hidden_state维度正确
        if hidden_state is None:
            hidden_state = self.target_actor.init_hidden().repeat(x.size(0), 1)
        
        logits, new_hidden = self.target_actor(x, hidden_state)
        return torch.sigmoid(logits).detach(), new_hidden

    def critic_value(self, state, action):
        """
        计算全局critic值（MADDPG）
        参数:
            state: 全局状态张量 [batch_size, total_obs_dim]
            action: 全局动作张量 [batch_size, total_act_dim]
        """
        x = torch.cat([state, action], dim=1)  # x.shape = [batch_size, total_obs_dim + total_act_dim]
        return self.critic(x).squeeze(1)  # 输出维度为[batch_size, 1]，squeeze(1)去掉最后一个维度

    def target_critic_value(self, state, action):
        """使用目标网络计算critic值"""
        x = torch.cat([state, action], dim=1)
        return self.target_critic(x).squeeze(1)

    def local_critic_value(self, state, action, last_action):
        """
        计算单个智能体的局部Q值（用于追逐者的QMIX）
        参数:
            state: 单个智能体状态 [batch_size, obs_dim]
            action: 单个智能体动作 [batch_size, act_dim]
            last_action: 单个智能体的上一个动作 [batch_size, act_dim]
        """
        x = torch.cat([state, action, last_action], dim=1)
        return self.local_critic(x).squeeze(1)  # 输出维度为[batch_size, 1]，squeeze(1)去掉最后一个维度

    def target_local_critic_value(self, state, action, last_action):
        """使用目标网络计算局部Q值"""
        x = torch.cat([state, action, last_action], dim=1)
        return self.target_q_agent(x).squeeze(1)

    def update_actor(self, loss):
        """更新actor网络，使用梯度裁剪提高稳定性"""
        self.actor_optimizer.zero_grad()
        loss.backward()
        # 基于智能体类型使用不同的裁剪阈值
        if self.agent_id.startswith("adversary_") or self.agent_id.startswith("leadadversary_"):
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.8)
        else:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        """更新critic网络，使用梯度裁剪"""
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
    #
    # def update_local_critic(self, loss):
    #     """更新局部Q网络，使用梯度裁剪"""
    #     self.local_critic_optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), 0.5)
    #     self.local_critic_optimizer.step()

    # def get_combined_q_value(self, global_state, global_action, local_state, local_action, last_action):
    #     """
    #     计算追逐者的组合Q值
    #     参数:
    #         global_state: 全局状态 [batch_size, total_obs_dim]
    #         global_action: 全局动作 [batch_size, total_act_dim]
    #         local_state: 局部状态 [batch_size, obs_dim]
    #         local_action: 局部动作 [batch_size, act_dim]
    #         last_action: 上一个动作 [batch_size, act_dim]
    #     返回:
    #         combined_q: 组合后的Q值
    #     """
    #     global_q = self.critic_value(global_state, global_action)
    #     local_q = self.local_critic_value(local_state, local_action, last_action)
    #
    #     # 线性组合全局和局部Q值
    #     combined_q = 0.5 * global_q + 0.5 * local_q
    #     return combined_q


class MLPNetwork(nn.Module):
    """用于Q函数和全局critics的标准多层感知机"""
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """初始化网络参数"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)


class RNNAgent(nn.Module):
    """基于RNN的actor网络，用于序列决策"""
    def __init__(self, obs_dim, action_dim, rnn_hidden_dim=64):
        super(RNNAgent, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(obs_dim, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, action_dim)

    def init_hidden(self):
        """使用与模型相同设备上的零初始化隐藏状态"""
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        """
        网络的前向传播
        参数:
            inputs: 输入张量 [batch_size, input_dim]
            hidden_state: 上一个隐藏状态 [batch_size, rnn_hidden_dim]
        返回:
            actions: 输出动作（logits）
            h: 更新的隐藏状态
        """
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        actions = self.fc2(h)  # 线性输出（无激活）,action函数里做最后一步sigmoid
        return actions, h