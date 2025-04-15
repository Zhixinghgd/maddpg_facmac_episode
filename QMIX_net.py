import torch.nn as nn
import torch
import torch.nn.functional as F


class QMixNet(nn.Module):
    def __init__(self, state_shape, hyper_hidden_dim, n_agents, qmix_hidden_dim):
        super(QMixNet, self).__init__()
        self.state_shape = state_shape
        self.hyper_hidden_dim = hyper_hidden_dim
        self.n_agents = n_agents
        self.qmix_hidden_dim = qmix_hidden_dim
        # 因为生成的hyper_w1需要是一个矩阵，而pytorch神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵

        # args.n_agents是使用hyper_w1作为参数的网络的输入维度，args.qmix_hidden_dim是网络隐藏层参数个数
        # 从而经过hyper_w1得到(经验条数，args.n_agents * args.qmix_hidden_dim)的矩阵
        self.hyper_w1 = nn.Sequential(nn.Linear(self.state_shape, self.hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.hyper_hidden_dim, self.n_agents * self.qmix_hidden_dim))
        # 经过hyper_w2得到(经验条数, 1)的矩阵
        self.hyper_w2 = nn.Sequential(nn.Linear(self.state_shape, self.hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.hyper_hidden_dim, self.qmix_hidden_dim))



        # hyper_w1得到的(经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
        self.hyper_b1 = nn.Linear(self.state_shape, self.qmix_hidden_dim)
        # hyper_w2得到的(经验条数，1)的矩阵需要同样维度的hyper_b1
        self.hyper_b2 =nn.Sequential(nn.Linear(self.state_shape, self.qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.qmix_hidden_dim, 1)
                                     )

    def forward(self, q_values, states):
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.n_agents)  # 重塑为3D张量 (batch_size, 1, n_agents)
        states = states.reshape(-1, self.state_shape)

        w1 = torch.abs(self.hyper_w1(states))  # (batch_size, n_agents * qmix_hidden_dim)
        b1 = self.hyper_b1(states)  # (batch_size, qmix_hidden_dim)

        w1 = w1.view(-1, self.n_agents, self.qmix_hidden_dim)  # (batch_size, n_agents, qmix_hidden_dim)
        b1 = b1.view(-1, self.qmix_hidden_dim)  # (batch_size, qmix_hidden_dim)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)  # (batch_size, 1, qmix_hidden_dim)

        w2 = torch.abs(self.hyper_w2(states))  # (batch_size, qmix_hidden_dim)
        b2 = self.hyper_b2(states)  # (batch_size, 1)

        w2 = w2.view(-1, self.qmix_hidden_dim)  # (batch_size, qmix_hidden_dim)
        b2 = b2.view(-1, 1)  # (batch_size, 1)

        q_total = torch.bmm(hidden, w2.unsqueeze(2)) + b2  # (batch_size, 1, 1)
        q_total = q_total.view(episode_num, -1)  # (batch_size, 1)
        return q_total

    def _init_weights(self):
        """Xavier初始化增强稳定性"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # 保证权重非负（QMIX核心约束）
        for seq in [self.hyper_w1, self.hyper_w2]:
            for layer in seq:
                if isinstance(layer, nn.Linear):
                    layer.weight.data = torch.abs(layer.weight.data)