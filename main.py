import argparse
import os
import gym
import matplotlib.pyplot as plt
import numpy as np

from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2, simple_world_comm_v2

from MADDPG import MADDPG


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
    if env_name == 'simple_world_comm_v2':
        new_env = simple_world_comm_v2.parallel_env(num_good=num_good, num_adversaries=num_adversaries, num_obstacles=1,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 解析命令行参数
    parser.add_argument('env_name', type=str, default='simple_adversary_v2', help='name of the env',
                        choices=['simple_adversary_v2', 'simple_spread_v2', 'simple_tag_v2', 'simple_world_comm_v2'])
    parser.add_argument('--episode_num', type=int, default=20000,
                        help='total episode num during training procedure')
    parser.add_argument('--episode_length', type=int, default=25, help='steps per episode')
    parser.add_argument('--learn_interval', type=int, default=4,
                        help='episodes interval between learning time')
    parser.add_argument('--random_episodes', type=int, default=2000,
                        help='random episodes before the agent start to learn')
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=5000, help='capacity of episode replay buffer')
    parser.add_argument('--batch_size', type=int, default=32, help='batch-size of replay buffer (in episodes)')
    parser.add_argument('--actor_lr', type=float, default=0.01, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate of critic')
    args = parser.parse_args()

    # create folder to save result
    env_dir = os.path.join('./results', args.env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(result_dir)

    env, dim_info, num_good, num_adversaries = get_env(args.env_name, args.episode_length)
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr,
                    result_dir, num_good, num_adversaries)

    episode_count = 0  # Track episodes
    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}
    episode_total_rewards = np.zeros(args.episode_num)
    
    for episode in range(args.episode_num):
        obs = env.reset()  # Initialize environment
        
        # Initialize last actions with zeros
        last_action = {agent_id: np.zeros(env.action_space(agent_id).shape) 
                      for agent_id in env.agents}
        
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # Track episode rewards
        r_total_reward = 0
        
        # Run the episode
        while env.agents:
            # Select actions - either random during initial exploration or from policy
            if episode < args.random_episodes:
                action = {agent_id: env.action_space(agent_id).sample() 
                        for agent_id in env.agents}
            else:
                action = maddpg.select_action(obs, last_action)
            
            # 执行动作并获取环境反馈
            # 注意: 这里的total_reward是环境直接返回的追逐者团队总体奖励
            next_obs, reward, total_reward, done, info = env.step(action)
            
            # 将步骤添加到经验回放缓冲区
            maddpg.add(obs, action, reward, done, total_reward)
            
            # 更新每个智能体的累计奖励
            for agent_id, r in reward.items():
                agent_reward[agent_id] += r
            # 更新追逐者团队的累计总奖励
            r_total_reward += total_reward
            
            # 准备下一步
            obs = next_obs
            last_action = action

        
        # 结束当前episode，将完整episode存入缓冲区
        maddpg.end_episode()
        episode_count += 1
        
        # 在随机探索期之后，每隔learn_interval个episode进行一次学习
        if episode >= args.random_episodes and episode % args.learn_interval == 0:
            # maddpg.learn(args.batch_size, args.gamma)
            # maddpg.qmix_learn(args.batch_size, args.gamma)
            maddpg.maddpg_learn(args.batch_size, args.gamma)
            maddpg.update_target(args.tau)
        
        # 记录奖励
        for agent_id, r in agent_reward.items():
            episode_rewards[agent_id][episode] = r
        episode_total_rewards[episode] = r_total_reward
        
        # 每100个episode打印一次信息
        if (episode + 1) % 100 == 0:
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():
                message += f'{agent_id}: {r:>4f}; '
                sum_reward += r
            message += f'总奖励: {sum_reward}, 追逐者团队奖励: {r_total_reward}'
            print(message)
    
    # 保存模型和奖励数据
    maddpg.save(episode_rewards, episode_total_rewards)
    
    # 绘制奖励曲线
    def get_running_reward(arr: np.ndarray, window=100):
        """计算滑动平均奖励"""
        running_reward = np.zeros_like(arr)
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])
        return running_reward
    
    # 绘制奖励曲线
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    
    # 绘制每个智能体的奖励曲线
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id)
        ax.plot(x, get_running_reward(reward))
    
    # 绘制追逐者团队的总奖励曲线
    ax.plot(x, episode_total_rewards, label='Pursuers Team', linewidth=2)
    ax.plot(x, get_running_reward(episode_total_rewards), linewidth=2)
    
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maddpg+qmix solve {args.env_name}'
    ax.set_title(title)
    
    plt.savefig(os.path.join(result_dir, title))