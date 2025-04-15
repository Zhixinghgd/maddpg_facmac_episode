import numpy as np

from .._mpe_utils.core import Agent, Landmark, World
from .._mpe_utils.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, num_good=1, num_adversaries=3, num_obstacles=2):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = num_good
        num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = num_obstacles
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.35, 0.85, 0.35])
                if not agent.adversary
                else np.array([0.85, 0.35, 0.35])
            )
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def calculate_total_reward(self, world):
        collision_reward = 0
        proximity_reward = 0
        cooperation_reward = 0
        time_penalty = -0.1
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)

        # 碰撞奖励（稀疏奖励）
        for adv in adversaries:
            if adv.collide:
                for ag in agents:
                    if self.is_collision(ag, adv):
                        collision_reward += 5  # 调整为2

        # 密集奖励：追逐者与所有逃跑者的距离
        # 改进的 proximity_reward：按最近目标 + 分层惩罚
        for adv in adversaries:
            # 计算到所有逃跑者的距离，取最小值
            distances = [
                np.sqrt(np.sum(np.square(ag.state.p_pos - adv.state.p_pos)))
                for ag in agents
            ]
            min_distance = min(distances) if distances else 0

            # 分层奖励设计
            if min_distance < 3.0:  # 近距离高权重
                proximity_reward -= 0.03 * min_distance
            elif min_distance < 10.0:  # 中距离中等权重
                proximity_reward -= 0.01 * min_distance
            else:  # 远距离低权重
                proximity_reward -= 0.005 * min_distance

        # 协作奖励：多个追逐者包围同一目标
        for ag in agents:
            distances = [np.sqrt(np.sum(np.square(ag.state.p_pos - adv.state.p_pos))) for adv in adversaries]
            num_close_adv = sum(d < 1.0 for d in distances)
            if num_close_adv >= 2:
                cooperation_reward += 1 * num_close_adv

        total_reward = collision_reward + proximity_reward + cooperation_reward + time_penalty
        return total_reward
    # def calculate_total_reward(self, world):
    #     # 计算所有追逐者的碰撞奖励之和
    #     collision_reward = 0
    #     agents = self.good_agents(world)
    #     adversaries = self.adversaries(world)
    #     for adv in adversaries:
    #         if adv.collide:
    #             for ag in agents:
    #                 if self.is_collision(ag, adv):
    #                     collision_reward += 10  # 碰撞奖励
    #
    #     # 计算所有逃跑者到最近追逐者的距离之和
    #     distance_penalty = 0
    #     for ag in agents:
    #         min_distance = min(
    #             np.sqrt(np.sum(np.square(ag.state.p_pos - adv.state.p_pos)))
    #             for adv in adversaries
    #         )
    #         distance_penalty += min_distance
    #
    #     # 总奖励 = 碰撞奖励 - 0.1 * 距离惩罚
    #     total_reward = collision_reward - 0.1 * distance_penalty
    #     return total_reward

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        # shape = False
        shape = True  # 启用形状奖励
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(
                    np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
                )
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    # def adversary_reward(self, agent, world):
    #     # Adversaries are rewarded for collisions with agents
    #     rew = 0
    #     # shape = False
    #     shape = True  # 启用形状奖励
    #     agents = self.good_agents(world)
    #     adversaries = self.adversaries(world)
    #     if (
    #         shape
    #     ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
    #         for adv in adversaries:
    #             rew -= 0.1 * min(
    #                 np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
    #                 for a in agents
    #             )
    #     if agent.collide:
    #         for ag in agents:
    #             for adv in adversaries:
    #                 if self.is_collision(ag, adv):
    #                     rew += 10
    #     return rew
    def adversary_reward(self, agent, world):
        rew = 0
        shape = True  # 启用形状奖励
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)

        # 形状奖励：仅计算当前追捕者到逃跑者的最小距离
        if shape:
            min_distance = min(
                np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos)))
                for a in agents
            )
            rew -= 0.3 * min_distance  # 距离越近，惩罚越小（奖励越大）

        # 碰撞奖励（原始逻辑）
        if agent.collide:
            for ag in agents:
                if self.is_collision(ag, agent):
                    rew += 10  # 碰撞奖励 +10

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + other_pos
            + other_vel
        )
