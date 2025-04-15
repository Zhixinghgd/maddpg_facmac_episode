import logging
import os
import pickle
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

from Agent import Agent
from Buffer import EpisodeBuffer
from QMIX_net import QMixNet
import logging


def setup_logger(filename):
    """
    设置带有文件名的日志记录器。
    输入为log文件路径，输出为一条日志记录，使用示例：
        logger = setup_logger('logfile.log')  # 配置日志记录器，日志将保存到 'logfile.log'
        logger.info("This is an info message.")  # 记录一条 INFO 消息，对应文件中2025-01-02 10:00:00--INFO--This is an info message.
        logger.warning("This is a warning message.")  # 记录一条 WARNING 消息，2025-01-02 10:00:01--WARNING--This is a warning message.
    """
    logger = logging.getLogger()  # 获取一个全局日志记录器实例
    logger.setLevel(logging.INFO)  # 设置日志记录器的最低日志级别为 INFO,只有级别大于等于 INFO 的日志消息才会被记录。

    # 创建文件处理器，指定日志文件和写入模式
    handler = logging.FileHandler(filename, mode='a')  # 使用追加模式
    handler.setLevel(logging.INFO)  # 设置文件处理器的最低日志级别为 INFO

    # 设置日志格式：年月日时分秒--日志级别--日志内容
    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    # 为了强制每次写入都立即刷新，覆盖原始的 `emit` 方法
    original_emit = handler.emit

    def flush_emit(record):
        original_emit(record)  # 执行原始的 emit 方法
        handler.flush()  # 强制刷新缓存

    handler.emit = flush_emit  # 用新的 emit 方法替换原始的 emit 方法

    # 将处理器添加到 logger 中
    logger.addHandler(handler)
    return logger


class MADDPG:
    """
    带有QMIX支持的MADDPG(多智能体深度确定性策略梯度)智能体
    参数:
        dim_info(dict):字典{agent_id：[obs_dim观测维度, act_dim动作维度]}.
        capacity(int):每个智能体经验回放缓冲区的最大容量,
        batch_size(int):每次采样的批量大小,
        actor_lr(float):actor 网络的学习率,
        critic_lr(float):critic 网络的学习率。
        res_dir(str):保存运行结果的文件夹路径
        num_good(int):逃跑者的数量
        num_adversaries(int):追逐者的数量
    """

    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir, num_good, num_adversaries):
        # 累加每个智能体的维度以获取critic的输入维度
        global_obs_dim = sum(obs_dim for obs_dim, _ in dim_info.values())
        global_act_dim = sum(act_dim for _, act_dim in dim_info.values())
        global_obs_act_dim = global_obs_dim + global_act_dim
        
        # 计算全局状态维度（仅用于追逐者）
        global_state_dim = 0
        for agent_id, (obs_dim, _) in dim_info.items():
            if agent_id.startswith("adversary_") or agent_id.startswith("leadadversary_"):
                global_state_dim += obs_dim
        print(f'global_state_dim:{global_state_dim}')

        # 创建智能体
        self.agents = {}
        self.num_good = num_good
        self.num_adversaries = num_adversaries
        
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.agents[agent_id] = Agent(agent_id, obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr)

        # 创建基于episode的缓冲区
        self.buffer = EpisodeBuffer(
            capacity, 
            dim_info, 
            num_good, 
            num_adversaries, 
            global_obs_dim, 
            global_act_dim, 
            max_seq_length=25, 
            device='cpu'
        )
        
        self.dim_info = dim_info
        self.capacity = capacity
        self.max_seq_length = 25
        self.agent_ids = list(dim_info.keys())
        self.adversary_ids = [aid for aid in self.agent_ids if "adversary" in aid]
        self.evader_ids = [aid for aid in self.agent_ids if "adversary" not in aid]
        
        # 为追逐者初始化QMIX网络
        self.Mixing_net = QMixNet(
            state_shape=global_state_dim,  # 所有追逐者观测维度的和，不要动作维度
            hyper_hidden_dim=64,
            n_agents=num_adversaries,
            qmix_hidden_dim=32
        )
        self.Mixing_target_net = deepcopy(self.Mixing_net)

        # 定义优化器
        self.mixer_optimizer = torch.optim.Adam(
            self.Mixing_net.parameters(),
            lr=critic_lr  # 应用qmix_lr， 暂用critic_lr代替
        )

        self.batch_size = batch_size
        self.res_dir = res_dir  # 保存训练结果的目录
        self.logger = setup_logger(os.path.join(res_dir, 'maddpg.log'))
        
        # 用于基于episode的训练
        self.hidden_states = {agent_id: self.agents[agent_id].init_hidden() for agent_id in self.agent_ids}

    def add(self, last_action, obs, action, reward, next_obs, done, total_reward=None):
        """
        添加一个时间步到当前episode缓冲区
        参数:
            last_action: 上一步的动作字典 {agent_id: action}
            obs: 当前观测字典 {agent_id: obs}
            action: 当前动作字典 {agent_id: action}
            reward: 当前奖励字典 {agent_id: reward}
            next_obs: 下一步观测字典 {agent_id: next_obs}
            done: 完成标志字典 {agent_id: done}
            total_reward: 追逐者团队的总体奖励（环境直接返回）
        """
        # 直接使用环境返回的total_reward，不需要再计算
        self.buffer.add_step(last_action, obs, action, reward, next_obs, done, total_reward)
        
    def end_episode(self):
        """结束当前episode并将其存储在缓冲区中"""
        self.buffer.end_episode()
        # 在episode结束时重置所有智能体的隐藏状态
        for agent_id in self.agent_ids:
            self.hidden_states[agent_id] = self.agents[agent_id].init_hidden().to(self.hidden_states[agent_id].device)
    def select_action(self, obs, last_action):
        """
        使用所有智能体的actor网络选择动作
        参数:
            obs: 每个智能体的观测字典
            last_action: 每个智能体的上一个动作字典
        返回:
            actions: 每个智能体的选择动作字典
        """
        actions = {}
        for agent_id in self.agent_ids:
            o = torch.from_numpy(obs[agent_id]).unsqueeze(0).float()  # [1, obs_dim]
            la = torch.from_numpy(last_action[agent_id]).unsqueeze(0).float()  # [1, act_dim]
            
            # 使用为此智能体存储的隐藏状态
            hidden = self.hidden_states[agent_id]
            
            # 获取动作并更新隐藏状态
            action, new_hidden = self.agents[agent_id].action(o, la, hidden)
            
            # 更新隐藏状态以供下一步使用
            self.hidden_states[agent_id] = new_hidden
            
            # 转换为numpy并移除批次维度
            actions[agent_id] = action.squeeze(0).cpu().detach().numpy()
            self.logger.info(f'{agent_id} action: {actions[agent_id]}')
            
        return actions
    
    def compute_next_actions(self, batch_next_obs, batch_actions, target_h_states, valid_mask):
        """
        计算所有智能体的下一步动作并更新它们的隐藏状态。
        参数:
            batch_next_obs: 每个智能体的下一步观测字典
            batch_actions: 每个智能体的当前动作字典
            target_h_states: 每个智能体的隐藏状态字典
            valid_mask: 有效样本的掩码
        返回:
            next_actions: 所有智能体的下一步动作列表
            updated_h_states: 所有智能体的更新后的隐藏状态
        """
        next_actions = []
        updated_h_states = {}

        for agent_id in self.agent_ids:
            agent = self.agents[agent_id]
            if agent_id in batch_next_obs and len(batch_next_obs[agent_id]) > 0:
                # 确保valid_mask是布尔类型
                valid_mask = valid_mask.bool()
                
                # 获取当前时间步的观测和动作
                current_time_step = 0  # 假设我们总是使用第一个时间步
                if current_time_step < len(batch_next_obs[agent_id]):
                    next_agent_obs = batch_next_obs[agent_id][current_time_step]
                    last_agent_action = batch_actions[agent_id][current_time_step]
                    
                    # 确保张量维度正确
                    if next_agent_obs.dim() == 1:
                        next_agent_obs = next_agent_obs.unsqueeze(0)
                    if last_agent_action.dim() == 1:
                        last_agent_action = last_agent_action.unsqueeze(0)
                    
                    # 使用valid_mask选择有效的样本
                    valid_indices = torch.nonzero(valid_mask).squeeze()
                    if valid_indices.numel() > 0:
                        if valid_indices.dim() == 0:  # 如果只有一个有效索引
                            valid_indices = valid_indices.unsqueeze(0)
                        
                        # 确保索引不超出范围
                        valid_indices = valid_indices[valid_indices < next_agent_obs.size(0)]
                        if valid_indices.numel() > 0:
                            next_agent_obs = next_agent_obs[valid_indices]
                            last_agent_action = last_agent_action[valid_indices]
                            target_h_state = target_h_states[agent_id][valid_indices]
                            
                            next_a, new_h_state = agent.target_action(
                                next_agent_obs, last_agent_action, target_h_state
                            )
                            next_actions.append(next_a)
                            updated_h_states[agent_id] = new_h_state
                        else:
                            # 如果没有有效样本，创建空张量
                            act_dim = self.dim_info[agent_id][1]
                            next_actions.append(torch.zeros(0, act_dim, device=next_agent_obs.device))
                            updated_h_states[agent_id] = target_h_states[agent_id]
                    else:
                        # 如果没有有效样本，创建空张量
                        act_dim = self.dim_info[agent_id][1]
                        next_actions.append(torch.zeros(0, act_dim, device=next_agent_obs.device))
                        updated_h_states[agent_id] = target_h_states[agent_id]
                else:
                    # 如果时间步超出范围，创建空张量
                    act_dim = self.dim_info[agent_id][1]
                    next_actions.append(torch.zeros(0, act_dim, device=batch_next_obs[agent_id][0].device))
                    updated_h_states[agent_id] = target_h_states[agent_id]
            else:
                # 如果agent_id不在batch_next_obs中，创建空张量
                act_dim = self.dim_info[agent_id][1]
                next_actions.append(torch.zeros(0, act_dim, device=next(batch_next_obs.values())[0][0].device))
                updated_h_states[agent_id] = target_h_states[agent_id]

        return next_actions, updated_h_states

    def learn(self, batch_size, gamma):
        """
        训练函数，包含MADDPG和QMIX的更新逻辑
        参数:
            batch_size: 批量大小
            gamma: 折扣因子
        """
        if len(self.buffer) < batch_size:
            return
        
        # 从缓冲区采样一个批次的数据
        batch = self.buffer.sample(batch_size)
        max_t = self.max_seq_length

        # === 初始化 ===
        # 初始化所有智能体的目标网络和当前网络的隐藏状态
        target_h = {aid: self.agents[aid].target_actor.init_hidden().repeat(batch_size, 1) for aid in self.agent_ids}
        actor_h = {aid: self.agents[aid].actor.init_hidden().repeat(batch_size, 1) for aid in self.agent_ids}

        # 存储用于训练QMIX的数据
        qmix_data = {
            'local_qs': [],        # 局部Q值
            'target_local_qs': [], # 目标局部Q值
            'states': [],          # 状态
            'next_states': [],     # 下一状态
            'total_rewards': [],   # 团队总奖励
            'dones': [],           # 完成标志
        }

        # 存储用于训练全局critic的数据
        critic_data = {
            'q_eval': {aid: [] for aid in self.agent_ids},  # 评估Q值
            'q_target': {aid: [] for aid in self.agent_ids}, # 目标Q值
            'mask': {aid: [] for aid in self.agent_ids}      # 有效样本掩码
        }

        # 对每个时间步进行训练
        for t in range(max_t):
            # 获取当前时间步有效的样本掩码
            mask = batch['lengths'] > t
            valid_n = mask.sum().item()
            if valid_n == 0:
                continue

            # === 构造全局状态和动作 ===
            # 将所有智能体的观测和动作拼接成全局状态和动作
            global_obs = torch.cat([batch['obs'][aid][mask, t] for aid in self.agent_ids], dim=-1)
            global_act = torch.cat([batch['actions'][aid][mask, t] for aid in self.agent_ids], dim=-1)
            next_global_obs = torch.cat([batch['next_obs'][aid][mask, t] for aid in self.agent_ids], dim=-1)
            next_act = []

            # === 计算目标动作 ===
            # 使用目标网络计算每个智能体的下一步动作
            for aid in self.agent_ids:
                obs_next = batch['next_obs'][aid][mask, t]
                last_act = batch['actions'][aid][mask, t]
                target_a, new_h = self.agents[aid].target_action(obs_next, last_act, target_h[aid][mask])
                next_act.append(target_a)
                target_h[aid][mask] = new_h

            next_global_act = torch.cat(next_act, dim=-1)

            # === 收集critic训练数据 ===
            for aid in self.agent_ids:
                agent = self.agents[aid]
                q_eval = agent.critic_value(global_obs, global_act)
                q_next = agent.target_critic_value(next_global_obs, next_global_act).detach()
                r = batch['rewards'][aid][mask, t]
                done = batch['dones'][aid][mask, t]
                q_target = r + gamma * (1 - done) * q_next
                
                # 存储critic训练数据
                critic_data['q_eval'][aid].append(q_eval)
                critic_data['q_target'][aid].append(q_target)
                critic_data['mask'][aid].append(mask)

            # === critic部分：局部Q + QMIX ===
            # 收集追逐者的局部Q值和状态信息
            adversary_q, target_adversary_q, adversary_states, next_states = [], [], [], []
            for aid in self.adversary_ids:
                agent = self.agents[aid]
                obs = batch['obs'][aid][mask, t]
                act = batch['actions'][aid][mask, t]
                last = batch['last_actions'][aid][mask, t]
                next_obs = batch['next_obs'][aid][mask, t]
                next_a = next_act[self.agent_ids.index(aid)]

                # 计算当前和目标的局部Q值
                q = agent.agent_q_value(obs, act, last)
                q_ = agent.target_agent_q_value(next_obs, next_a, act).detach()
                adversary_q.append(q)
                target_adversary_q.append(q_)
                adversary_states.append(obs)
                next_states.append(next_obs)

            # 如果有追逐者，准备QMIX训练数据
            if adversary_q:
                q_stack = torch.stack(adversary_q, dim=1)
                q_next_stack = torch.stack(target_adversary_q, dim=1)
                state_stack = torch.cat(adversary_states, dim=1)
                next_state_stack = torch.cat(next_states, dim=1)
                team_r = batch['total_rewards'][:, t][mask]
                dones = batch['dones'][self.adversary_ids[0]][mask, t]

                # 存储QMIX训练数据
                qmix_data['local_qs'].append(q_stack)
                qmix_data['target_local_qs'].append(q_next_stack)
                qmix_data['states'].append(state_stack)
                qmix_data['next_states'].append(next_state_stack)
                qmix_data['total_rewards'].append(team_r)
                qmix_data['dones'].append(dones)

        # === 更新全局critic网络 ===
        for aid in self.agent_ids:
            if critic_data['q_eval'][aid]:  # 确保有数据
                # 拼接所有时间步的数据
                q_eval = torch.cat(critic_data['q_eval'][aid], dim=0)
                q_target = torch.cat(critic_data['q_target'][aid], dim=0)
                mask = torch.cat(critic_data['mask'][aid], dim=0)
                
                # 计算损失并更新
                loss = F.mse_loss(q_eval[mask], q_target[mask])
                self.agents[aid].update_critic(loss)

        # === 训练Mixing网络 ===
        if qmix_data['local_qs']:
            # 准备QMIX训练数据
            q_stack = torch.cat(qmix_data['local_qs'], dim=0)
            target_q_stack = torch.cat(qmix_data['target_local_qs'], dim=0)
            states = torch.cat(qmix_data['states'], dim=0)
            next_states = torch.cat(qmix_data['next_states'], dim=0)
            total_rewards = torch.cat(qmix_data['total_rewards'], dim=0).unsqueeze(-1)
            dones = torch.cat(qmix_data['dones'], dim=0).unsqueeze(-1)

            # 计算QMIX的Q值和目标Q值
            q_tot = self.Mixing_net(q_stack, states)
            q_next_tot = self.Mixing_target_net(target_q_stack, next_states)
            target = total_rewards + gamma * (1 - dones) * q_next_tot
            loss = F.mse_loss(q_tot, target.detach())
            self.update_mixing(loss)

        # === Actor 训练：FACMAC 风格 ===
        # 1) 初始化所有 actor 的隐藏状态
        actor_h = {
            aid: self.agents[aid].actor.init_hidden().repeat(batch_size, 1)
            for aid in self.agent_ids
        }
        # 2) 清零所有 actor 优化器梯度
        for aid in self.agent_ids:
            self.agents[aid].actor_optimizer.zero_grad()

        total_actor_loss = 0.0
        count = 0

        # 3) 按时间步生成动作 & 累积 loss
        for t in range(max_t):
            mask = batch['lengths'] > t
            if mask.sum().item() == 0:
                continue

            # 3.1) 准备每个 agent 的 obs_t 和 last_action_t
            obs_t = {x: batch['obs'][x][mask, t] for x in self.agent_ids}
            last_t = {x: batch['last_actions'][x][mask, t] for x in self.agent_ids}

            # 3.2) 用 actor 生成动作并更新隐藏状态
            acts_t = {}
            logits_t = {}
            for x in self.agent_ids:
                act, logit, new_h = self.agents[x].action(
                    obs_t[x], last_t[x], actor_h[x][mask], model_out=True
                )
                acts_t[x] = act
                logits_t[x] = logit
                actor_h[x][mask] = new_h

            # 3.3) 构建全局 obs & act
            global_obs = torch.cat([obs_t[x] for x in self.agent_ids], dim=-1)
            new_global_act = torch.cat([acts_t[x] for x in self.agent_ids], dim=-1)

            # 3.4) 计算全局 Q
            q_globals = {
                x: self.agents[x].critic_value(global_obs, new_global_act)
                for x in self.agent_ids
            }

            # 3.5) FACMAC: 计算团队 Q_tot（不再 no_grad）
            if self.adversary_ids:
                local_qs = [
                    self.agents[adv].agent_q_value(obs_t[adv], acts_t[adv], last_t[adv])
                    for adv in self.adversary_ids
                ]  # list of [batch]
                q_stack = torch.stack(local_qs, dim=1)  # [batch, n_adversaries]
                adv_state = torch.cat(
                    [obs_t[adv] for adv in self.adversary_ids], dim=1
                )  # [batch, sum(obs_dims)]
                q_tot = self.Mixing_net(q_stack, adv_state).squeeze(-1)  # [batch]

            # 3.6) 对每个 agent 累积 loss
            for x in self.agent_ids:
                qg = q_globals[x]
                if x in self.adversary_ids:
                    # FACMAC 用 Q_tot；保留全局 Q 可写成 0.3*q_tot + 0.7*qg
                    q_comb = q_tot
                else:
                    q_comb = qg

                loss_t = - q_comb.mean() + 1e-3 * logits_t[x].pow(2).mean()
                total_actor_loss += loss_t
                count += 1

        # 4) 平均并一次性 backward
        if count > 0:
            total_actor_loss = total_actor_loss / count
            total_actor_loss.backward()

        # 5) 梯度裁剪 & 更新所有 actor
        for aid in self.agent_ids:
            clip_val = 0.8 if aid in self.adversary_ids else 0.5
            torch.nn.utils.clip_grad_norm_(
                self.agents[aid].actor.parameters(), clip_val
            )
            self.agents[aid].actor_optimizer.step()

    def update_mixing(self, qmix_loss):
        """
        更新QMIX网络和个别追逐者Q网络
        参数:
            qmix_loss: QMIX网络的损失值
        """
        # 重置所有相关网络的梯度
        self.mixer_optimizer.zero_grad()
        for agent_id in self.adversary_ids:
            self.agents[agent_id].q_agent_optimizer.zero_grad()

        # 反向传播计算梯度
        qmix_loss.backward()

        # 应用梯度裁剪，使用不同的阈值
        # QMIX网络使用较大的阈值，因为需要处理多个智能体的信息
        torch.nn.utils.clip_grad_norm_(self.Mixing_net.parameters(), 10.0)
        
        # 追逐者的局部Q网络使用较小的阈值，因为只处理单个智能体的信息
        for agent_id in self.adversary_ids:
            torch.nn.utils.clip_grad_norm_(
                self.agents[agent_id].q_agent.parameters(), 
                max_norm=5.0
            )

        # 更新网络参数
        self.mixer_optimizer.step()
        for agent_id in self.adversary_ids:
            self.agents[agent_id].q_agent_optimizer.step()

    def update_target(self, tau):
        """
        目标网络的软更新
        tau: 插值参数
        """
        def soft_update(from_network, to_network):
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        # 更新QMIX目标网络
        soft_update(self.Mixing_net, self.Mixing_target_net)
        
        # 更新所有智能体网络
        for agent_id, agent in self.agents.items():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)
            if agent_id in self.adversary_ids:
                soft_update(agent.q_agent, agent.target_q_agent)

    def save(self, reward, total_rewards):
        """保存模型参数和训练结果"""
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},
            os.path.join(self.res_dir, 'model.pt')
        )
        with open(os.path.join(self.res_dir, 'rewards.pkl'), 'wb') as f:
            pickle.dump({'rewards': reward, 'total_rewards': total_rewards}, f)

    @classmethod
    def load(cls, dim_info, file, num_good, num_adversaries):
        """加载保存的模型参数"""
        instance = cls(dim_info, 0, 0, 0, 0, os.path.dirname(file), num_good, num_adversaries)
        data = torch.load(file)
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return instance