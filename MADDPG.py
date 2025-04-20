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


def setup_logger(name: str, filename: str):
    """
    name: logger 名称
    filename: 输出文件
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # 禁止向上冒泡到 root
    logger.propagate = False

    # 如果这个 logger 已经有 handler，就不再重复添加
    if not logger.handlers:
        handler = logging.FileHandler(filename, mode='a')
        handler.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(fmt)
        # 强制 flush
        orig_emit = handler.emit

        def flush_emit(rec):
            orig_emit(rec)
            handler.flush()

        handler.emit = flush_emit

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

        self.logger = setup_logger('maddpg', os.path.join(res_dir, 'maddpg.log'))
        self.global_q_logger = setup_logger('global_q', os.path.join(res_dir, 'global_q_loss.log'))
        self.qmix_logger = setup_logger('qmix', os.path.join(res_dir, 'qmix_loss.log'))
        self.actor_logger = setup_logger('actor', os.path.join(res_dir, 'actor_loss.log'))

        # 用于基于episode的训练
        self.hidden_states = {agent_id: self.agents[agent_id].init_hidden() for agent_id in self.agent_ids}

    def add(self, obs, action, reward, done, total_reward):
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
        self.buffer.add_step(obs, action, reward, done, total_reward)

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

    def maddpg_learn(self, batch_size, gamma):
        if len(self.buffer) < batch_size:
            return
        # 1) sample a batch of episodes
        batch = self.buffer.sample(batch_size)
        T = self.max_seq_length
        target_actor_h = {aid: self.agents[aid].target_actor.init_hidden().repeat(batch_size, 1)
                    for aid in self.agent_ids}

        # 计算target_action
        target_actions_dict = {agent_id: [] for agent_id in
                               self.agents.keys()}  # {agent_id: [batch_size, max_episode_length, act_dim)}
        target_actions = []  # [n_agents, batch_size, act_dim]
        for t in range(T-1):  # t：[0,T-1]
            for agent_id, agent in self.agents.items():
                target_act, new_hidden = agent.target_action(
                    batch['obs'][agent_id][:, t+1:t+2].squeeze(1),  # next_obs -> [batch_size, obs_dim]
                    batch['actions'][agent_id][:, t:t+1].squeeze(1),  #last_action
                    target_actor_h[agent_id]
                )  # target_act -> [batch_size, act_dim]
                target_actor_h[agent_id] = new_hidden
                target_actions_dict[agent_id].append(target_act.detach())
        # 循环结束后target_actions_dict[agent_id] -> [max_episode_length-1, batch_size, act_dim]

        # 将列表转换为numpy数组，并调整维度
        for agent_id in target_actions_dict.keys():
            target_actions_dict[agent_id] = np.stack(target_actions_dict[agent_id],
                                                     axis=1)  # 转换为[batch_size, max_episode_length-1, act_dim]

        # ========== 1) GLOBAL CRITIC (MADDPG) + QMIX LOCAL Q UPDATES ==========
        q_global_loss = {agent_id: [] for agent_id in self.agent_ids}

        for t in range(T-1):

            # --- build global state/action and next_state/action ---
            global_obs = torch.cat([batch['obs'][aid][:, t:t+1].squeeze(1) for aid in self.agent_ids], dim=1)  # ???不确定，obs里面：[batch_size,t,单个obs_dim]
            global_act = torch.cat([batch['actions'][aid][:, t:t+1].squeeze(1) for aid in self.agent_ids], dim=1)
            next_global_obs = torch.cat([batch['obs'][aid][:, t+1:t+2].squeeze(1) for aid in self.agent_ids], dim=1)
            next_acts = torch.cat([
                torch.from_numpy(target_actions_dict[aid][:, t:t + 1].astype(np.float32)).squeeze(1) for aid in
                self.agent_ids
            ], dim=1)  # target_actions_dict是np数组，需要转成张量

            # --- MADDPG 的全局critic部分的损失计算 ---
            for aid in self.agent_ids:
                agent = self.agents[aid]
                global_q_eval = agent.critic_value(global_obs, global_act)  # [batch_size, 1]
                global_q_next = agent.target_critic_value(next_global_obs, next_acts).detach()
                r = batch['rewards'][aid][:, t:t+1].squeeze(1)
                d = batch['dones'][aid][:, t:t+1].squeeze(1)
                q_target = r + gamma * (1 - d) * global_q_next
                # 下面的可能应该在t循环外更新
                critic_loss = F.mse_loss(global_q_eval, q_target)
                q_global_loss[aid].append(critic_loss)

        # ========== 全局Q、mixing、局部Q网络更新 ==========
        for agent_id, agent in self.agents.items():
            if len(q_global_loss[agent_id]) > 0:
                avg_loss = torch.stack(q_global_loss[agent_id]).mean()
                agent.update_critic(avg_loss)

        # ========== 2) ACTOR UPDATE (FACMAC style) ==========
        actor_h = {aid: self.agents[aid].actor.init_hidden().repeat(batch_size, 1)
                   for aid in self.agent_ids}

        actor_loss_list = {aid: [] for aid in self.agent_ids}

        for t in range(T-1):
            # build per‐agent obs & last_act
            obs_t = {aid: batch['obs'][aid][:, t:t+1].squeeze(1) for aid in self.agent_ids}
            act = {aid: batch['actions'][aid][:, t:t+1].squeeze(1) for aid in self.agent_ids}  # [aid]   [banch_size, act_dim]
            if t > 0:
                last_act = {aid: batch['actions'][aid][:, t-1:t].squeeze(1) for aid in self.agent_ids}
            else:
                last_act = {aid: torch.zeros_like(act[aid]) for aid in self.agent_ids}
            # global obs & joint action
            global_obs = torch.cat([batch['obs'][aid][:, t:t+1].squeeze(1) for aid in self.agent_ids], dim=1)

            # get new actions & update hidden
            acts_for_id, logits_t = {}, {}
            for aid in self.agent_ids:
                a, logit, h_new = self.agents[aid].action(obs_t[aid], last_act[aid], actor_h[aid], model_out=True)
                acts_for_id = {**act, aid: a}  # 创建新动作字典
                logits_t[aid] = logit
                actor_h[aid] = h_new

                g_act = torch.cat([acts_for_id[aid] for aid in self.agent_ids], dim=1)  # [batch_size, act_dim * num_agents]
                q_global = self.agents[aid].critic_value(global_obs, g_act)  # [batch_size, 1]

                actor_loss = -q_global.mean() + 1e-3 * torch.pow(logit, 2).mean()
                actor_loss_list[aid].append(actor_loss)

        # ========== 3) ACTOR UPDATE ==========
        for agent_id, agent in self.agents.items():
            avg_actor_loss = torch.stack(actor_loss_list[agent_id]).mean()
            agent.update_actor(avg_actor_loss)

    def learn(self, batch_size, gamma, alpha=0.7):
        if len(self.buffer) < batch_size:
            return

        # 1) sample a batch of episodes
        batch = self.buffer.sample(batch_size)
        T = self.max_seq_length

        # 2) init all hidden states

        target_actor_h = {aid: self.agents[aid].target_actor.init_hidden().repeat(batch_size, 1)
                    for aid in self.agent_ids}

        # 计算target_action
        target_actions_dict = {agent_id: [] for agent_id in
                               self.agents.keys()}  # {agent_id: [batch_size, max_episode_length, act_dim)}
        target_actions = []  # [n_agents, batch_size, act_dim]
        for t in range(T-1):  # t：[0,T-1]
            for agent_id, agent in self.agents.items():
                target_act, new_hidden = agent.target_action(
                    batch['obs'][agent_id][:, t+1:t+2].view(batch_size, -1),  # 原next_obs -> [batch_size, 1 ,obs_dim],移除此维
                    batch['actions'][agent_id][:, t:t+1].view(batch_size, -1),  #last_action
                    target_actor_h[agent_id]
                )  # target_act -> [batch_size, act_dim]
                target_actor_h[agent_id] = new_hidden
                target_actions_dict[agent_id].append(target_act)
        # 循环结束后target_actions_dict[agent_id] -> [max_episode_length-1, batch_size, act_dim]

        # 将列表转换为numpy数组，并调整维度
        for agent_id in target_actions_dict.keys():
            target_actions_dict[agent_id] = np.stack(target_actions_dict[agent_id],
                                                     axis=1)  # 转换为[batch_size, max_episode_length-1, act_dim]

        # ========== 1) GLOBAL CRITIC (MADDPG) + QMIX LOCAL Q UPDATES ==========
        q_global_loss = {agent_id: [] for agent_id in self.agent_ids}
        q_mixing_loss = []

        for t in range(T-1):
            # mask = batch['lengths'] > t  # batch['lengths']应该是batch_size个长度值
            # if mask.sum() == 0:  # 说明抽到一组空数据
            #     continue

            # --- build global state/action and next_state/action ---
            global_obs = torch.cat([batch['obs'][aid][:, t:t+1].view(batch_size, -1) for aid in self.agent_ids], dim=1)  # ???不确定，obs里面：[batch_size,t,单个obs_dim]
            # [batch['obs'][aid] : [batch_size, max_episode_length, obs_dim]
            global_act = torch.cat([batch['actions'][aid][:, t:t+1].view(batch_size, -1) for aid in self.agent_ids], dim=1)
            next_global_obs = torch.cat([batch['obs'][aid][:, t+1:t+2].view(batch_size, -1) for aid in self.agent_ids], dim=1)
            # batch['actions'][aid] : [batch_size, max_episode_length, act_dim]
            # next actions via target actors
            next_acts = torch.cat([
                torch.from_numpy(target_actions_dict[aid][:, t:t + 1].astype(np.float32)).view(batch_size, -1) for aid in
                self.agent_ids
            ], dim=1)  # target_actions_dict是np数组，需要转成张量

            # --- MADDPG 的全局critic部分的损失计算 ---
            for aid in self.agent_ids:
                agent = self.agents[aid]
                global_q_eval = agent.critic_value(global_obs, global_act)  # [batch_size, 1]
                global_q_next = agent.target_critic_value(next_global_obs, next_acts).detach()
                r = batch['rewards'][aid][:, t:t+1].view(-1)
                d = batch['dones'][aid][:, t:t+1].view(-1)
                q_target = r + gamma * (1 - d) * global_q_next
                # 下面的可能应该在t循环外更新
                # print(global_q_eval.shape ,global_q_next.shape ,r.shape , d.shape , q_target.shape)
                critic_loss = F.mse_loss(global_q_eval, q_target)
                q_global_loss[aid].append(critic_loss)
                # agent.update_critic(critic_loss)
                # self.global_q_logger.info(f"{aid} Critic Loss: {critic_loss.item():.4f}")

            # --- mixing 网络和损失计算 ---
            adv_qs, adv_q_nexts, adv_states, adv_next_states = [], [], [], []
            for aid in self.adversary_ids:
                agent = self.agents[aid]
                obs = batch['obs'][aid][:, t:t+1].view(batch_size, -1) # [batch_size, 1, obs_dim]
                act = batch['actions'][aid][:, t:t+1].view(batch_size, -1)
                last_act = batch['actions'][aid][:, t-1:t].view(batch_size, -1) if t > 0 else torch.zeros_like(act)
                obs_next = batch['obs'][aid][:, t+1:t+2].view(batch_size, -1)
                # next_a = next_acts[self.agent_ids.index(aid)]    # 出问题
                next_a = torch.from_numpy(target_actions_dict[aid][:, t:t + 1].astype(np.float32)).view(batch_size, -1)

                q_loc = agent.local_critic_value(obs, act, last_act)  # q_loc -> [batch_size]
                q_loc_n = agent.target_local_critic_value(obs_next, next_a, act).detach()
                adv_qs.append(q_loc)  # 循环后 adv_qs -> [num_adversarys, batch_size]（[[batch_size],[batch_size],[],[],....num_adversarys个]）
                adv_q_nexts.append(q_loc_n)
                adv_states.append(obs)  # adv_states -> [num_adversarys, batch_size, obs_dim]
                adv_next_states.append(obs_next)

            if adv_qs:
                q_stack = torch.stack(adv_qs, dim=1)  # q_stack -> [batch_size,num_adversarys] ([[num_adversarys],[num_adversarys],[],...batch_size个])
                q_next_stack = torch.stack(adv_q_nexts, dim=1)
                state_stack = torch.cat(adv_states, dim=1)
                # adv_states 的组成单元是num_adversarys个形如[batch_size, obs_dim]的列表，所以dim=1对obs堆叠，堆后形如[batch_size, num_adversarys * obs_dim]
                next_state_st = torch.cat(adv_next_states, dim=1)
                #4/19改到这
                team_r = batch['total_rewards'][:, t:t+1] # .unsqueeze(-1) 在最后一个维度上扩1维，从形状 (batch_size,) 变为 (batch_size, 1)。之前按t截取缺一个降维，应该正好和这个升维抵消
                # dones = batch['dones'][self.adversary_ids[0]][:, t:t+1] #.unsqueeze(-1) 同上 ？？用第一个追逐者是否结束来确定整个qtot是否有效，不合适吧？
                dones = torch.stack(
                    [batch['dones'][aid][:, t] for aid in self.agent_ids],
                    dim=0
                ).any(dim=0).float().view(-1, 1)  # [batch_size, 1]
                q_tot = self.Mixing_net(q_stack, state_stack)  # q_tot -> [batch_size, 1]
                q_next_tot = self.Mixing_target_net(q_next_stack, next_state_st).detach()
                mix_target = team_r + gamma * (1 - dones) * q_next_tot
                # loss更新放在t循环外
                mix_loss = F.mse_loss(q_tot, mix_target)
                q_mixing_loss.append(mix_loss)
                # self.update_mixing(mix_loss)
                # self.qmix_logger.info(f"QMIX Loss: {mix_loss.item():.4f}")

        # ========== 全局Q、mixing、局部Q网络更新 ==========
        for agent_id, agent in self.agents.items():
            if len(q_global_loss[agent_id]) > 0:
                avg_loss = torch.stack(q_global_loss[agent_id]).mean()
                agent.update_critic(avg_loss)

        self.update_mixing(torch.stack(q_mixing_loss).mean())

        # ========== 2) ACTOR UPDATE (FACMAC style) ==========
        actor_h = {aid: self.agents[aid].actor.init_hidden().repeat(batch_size, 1)
                   for aid in self.agent_ids}

        actor_loss_list = {aid: [] for aid in self.agent_ids}


        # replay the same T steps to compute policy gradient
        for t in range(T-1):
            # mask = batch['lengths'] > t
            # if mask.sum() == 0:
            #     continue

            # build per‐agent obs & last_act
            obs_t = {aid: batch['obs'][aid][:, t:t+1].view(batch_size, -1) for aid in self.agent_ids}
            act = {aid: batch['actions'][aid][:, t:t+1].view(batch_size, -1) for aid in self.agent_ids}  # [aid]   [banch_size, act_dim]
            if t > 0:
                last_act = {aid: batch['actions'][aid][:, t-1:t].view(batch_size, -1) for aid in self.agent_ids}
            else:
                last_act = {aid: torch.zeros_like(act[aid]) for aid in self.agent_ids}
            # global obs & joint action
            global_obs = torch.cat([batch['obs'][aid][:, t:t+1].view(batch_size, -1) for aid in self.agent_ids], dim=1)
            adv_state = torch.cat([obs_t[adv] for adv in self.adversary_ids], dim=1)

            #原actor loss计算方式，除了自身都是采样动作
            # acts_for_id, logits_t = {}, {}
            # for aid in self.agent_ids:
            #     a, logit, h_new = self.agents[aid].action(obs_t[aid], last_act[aid], actor_h[aid], model_out=True)
            #     acts_for_id = {**act, aid: a}  # 创建新动作字典
            #     logits_t[aid] = logit
            #     actor_h[aid] = h_new
            #
            #     g_act = torch.cat([acts_for_id[aid] for aid in self.agent_ids], dim=1)  # [batch_size, act_dim * num_agents]
            #     q_global = self.agents[aid].critic_value(global_obs, g_act)  # [batch_size, 1]
            #
            #     if self.adversary_ids:
            #         local_qs = [
            #             self.agents[adv].local_critic_value(obs_t[adv], acts_for_id[adv], last_act[adv])
            #             for adv in self.adversary_ids
            #         ]  # local_qs -> [num_adversarys, batch_size]（[[batch_size],[batch_size],[],[],....num_adversarys个]）
            #         q_stack = torch.stack(local_qs, dim=1)  # q_stack -> [batch_size,num_adversarys] ([[num_adversarys],[num_adversarys],[],...batch_size个])
            #
            #         q_tot = self.Mixing_net(q_stack, adv_state).squeeze(-1)
            #         combined_q = alpha * q_global + (1/self.num_adversaries) * (1-alpha) * q_tot
            #
            #     else:
            #         combined_q = q_global
            #     actor_loss = -combined_q.mean() + 1e-3 * torch.pow(logit, 2).mean()
            #     actor_loss_list[aid].append(actor_loss)
            new_actions = {}
            logits_dict = {}
            for aid in self.agent_ids:
                a, logit, h_new = self.agents[aid].action(obs_t[aid], last_act[aid], actor_h[aid], model_out=True)
                new_actions[aid] = a
                logits_dict[aid] = logit
                actor_h[aid] = h_new

            # 计算每个智能体的actor loss
            for aid in self.agent_ids:
                # 梯度隔离逻辑 创建梯度隔离的动作字典
                detached_actions = {
                    other_aid: new_actions[other_aid].detach() if other_aid != aid
                    else new_actions[other_aid]
                    for other_aid in self.agent_ids
                }
                # 拼接全局动作
                g_act = torch.cat([detached_actions[other_aid] for other_aid in self.agent_ids], dim=1)
                # 计算全局Q值
                q_global = self.agents[aid].critic_value(global_obs, g_act)
                # q_tot计算保持梯度隔离
                if self.adversary_ids:
                    # 使用detach后的追逐者动作计算局部Q
                    local_qs = [
                        self.agents[adv].local_critic_value(
                            obs_t[adv],
                            detached_actions[adv],  # 使用隔离后的动作
                            last_act[adv]
                        )
                        for adv in self.adversary_ids
                    ]
                    q_stack = torch.stack(local_qs, dim=1)
                    q_tot = self.Mixing_net(q_stack, adv_state).squeeze(-1)
                    combined_q = alpha * q_global + (1 / self.num_adversaries) * (1 - alpha) * q_tot
                else:
                    combined_q = q_global

                # 计算最终loss
                actor_loss = -combined_q.mean() + 1e-3 * torch.pow(logits_dict[aid], 2).mean()
                actor_loss_list[aid].append(actor_loss)

        # ========== 3) ACTOR UPDATE ==========
        for agent_id, agent in self.agents.items():
            avg_actor_loss = torch.stack(actor_loss_list[agent_id]).mean()
            agent.update_actor(avg_actor_loss)


    def update_mixing(self, qmix_loss):
        """
        更新QMIX网络和个别追逐者Q网络
        参数:
            qmix_loss: QMIX网络的损失值
        """
        # 重置所有相关网络的梯度
        self.mixer_optimizer.zero_grad()
        for agent_id in self.adversary_ids:
            self.agents[agent_id].local_critic_optimizer.zero_grad()

        # 反向传播计算梯度
        qmix_loss.backward()

        # 应用梯度裁剪，使用不同的阈值
        # QMIX网络使用较大的阈值，因为需要处理多个智能体的信息
        torch.nn.utils.clip_grad_norm_(self.Mixing_net.parameters(), 0.5)

        # 追逐者的局部Q网络使用较小的阈值，因为只处理单个智能体的信息
        for agent_id in self.adversary_ids:
            torch.nn.utils.clip_grad_norm_(
                self.agents[agent_id].local_critic.parameters(),
                max_norm=0.5
            )

        # 更新网络参数
        self.mixer_optimizer.step()
        for agent_id in self.adversary_ids:
            self.agents[agent_id].local_critic_optimizer.step()

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
                soft_update(agent.local_critic, agent.target_local_critic)

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
