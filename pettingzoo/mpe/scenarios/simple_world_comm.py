import numpy as np

from .._mpe_utils.core import Agent, Landmark, World
from .._mpe_utils.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(
        self,
        num_good_agents=2,
        num_adversaries=4,
        num_landmarks=1,
        num_food=2,
        num_forests=2,
    ):
        world = World()
        # set any world properties first
        world.dim_c = 4
        # world.damping = 1
        num_good_agents = num_good_agents
        num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = num_landmarks
        num_food = num_food
        num_forests = num_forests
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_index = i - 1 if i < num_adversaries else i - num_adversaries
            base_index = 0 if base_index < 0 else base_index
            base_name = "adversary" if agent.adversary else "agent"
            base_name = "leadadversary" if i == 0 else base_name
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.leader = True if i == 0 else False
            agent.silent = True if i > 0 else False
            agent.size = 0.075 if agent.adversary else 0.045
            agent.accel = 3.0 if agent.adversary else 4.0
            # agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        world.food = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.food):
            landmark.name = "food %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03
            landmark.boundary = False
        world.forests = [Landmark() for i in range(num_forests)]
        for i, landmark in enumerate(world.forests):
            landmark.name = "forest %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.3
            landmark.boundary = False
        world.landmarks += world.food
        world.landmarks += world.forests
        # world.landmarks += self.set_boundaries(world)
        # world boundaries now penalized with negative reward
        return world

    def set_boundaries(self, world):
        boundary_list = []
        landmark_size = 1
        edge = 1 + landmark_size
        num_landmarks = int(edge * 2 / landmark_size)
        for x_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([x_pos, -1 + i * landmark_size])
                boundary_list.append(l)

        for y_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([-1 + i * landmark_size, y_pos])
                boundary_list.append(l)

        for i, l in enumerate(boundary_list):
            l.name = "boundary %d" % i
            l.collide = True
            l.movable = False
            l.boundary = True
            l.color = np.array([0.75, 0.75, 0.75])
            l.size = landmark_size
            l.state.p_vel = np.zeros(world.dim_p)

        return boundary_list

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.45, 0.95, 0.45])
                if not agent.adversary
                else np.array([0.95, 0.45, 0.45])
            )
            agent.color -= (
                np.array([0.3, 0.3, 0.3]) if agent.leader else np.array([0, 0, 0])
            )
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for i, landmark in enumerate(world.food):
            landmark.color = np.array([0.15, 0.15, 0.65])
        for i, landmark in enumerate(world.forests):
            landmark.color = np.array([0.6, 0.9, 0.6])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.food):
            landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.forests):
            landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
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

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        # boundary_reward = -10 if self.outside_boundary(agent) else 0
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        return main_reward

    def outside_boundary(self, agent):
        if (
            agent.state.p_pos[0] > 1
            or agent.state.p_pos[0] < -1
            or agent.state.p_pos[1] > 1
            or agent.state.p_pos[1] < -1
        ):
            return True
        else:
            return False

    def agent_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        # shape = False
        shape = True  # 启用形状奖励
        adversaries = self.adversaries(world)
        if shape:
            for adv in adversaries:
                rew += 0.1 * np.sqrt(
                    np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
                )
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 5

        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)  # 1 + (x - 1) * (x - 1)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= 2 * bound(x)

        for food in world.food:
            if self.is_collision(agent, food):
                rew += 2
        rew -= 0.05 * min(
            np.sqrt(np.sum(np.square(food.state.p_pos - agent.state.p_pos)))
            for food in world.food
        )

        return rew

    def adversary_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:
            rew -= 0.1 * min(
                np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos)))
                for a in agents
            )
        # if agent.collide:
        #     for ag in agents:
        #         for adv in adversaries:
        #             if self.is_collision(ag, adv):
        #                 rew += 5
        if agent.collide:
            for ag in agents:
                if self.is_collision(ag, agent):
                    rew += 5  # 碰撞奖励 +10
        return rew

    def calculate_total_reward(self, world):
        # 计算所有追逐者的碰撞奖励之和
        collision_reward = 0
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        for adv in adversaries:
            if adv.collide:
                for ag in agents:
                    if self.is_collision(ag, adv):
                        collision_reward += 5  # 碰撞奖励

        # 计算所有逃跑者到最近追逐者的距离之和
        distance_penalty = 0
        for ag in agents:
            min_distance = min(
                np.sqrt(np.sum(np.square(ag.state.p_pos - adv.state.p_pos)))
                for adv in adversaries
            )
            distance_penalty += min_distance

        # 总奖励 = 碰撞奖励 - 0.1 * 距离惩罚
        total_reward = collision_reward - 0.1 * distance_penalty
        return total_reward

    def observation2(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        food_pos = []
        for entity in world.food:
            if not entity.boundary:
                food_pos.append(entity.state.p_pos - agent.state.p_pos)
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

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        in_forest = [np.array([-1]) for _ in range(len(world.forests))]
        inf = [False for _ in range(len(world.forests))]

        for i in range(len(world.forests)):
            if self.is_collision(agent, world.forests[i]):
                in_forest[i] = np.array([1])
                inf[i] = True

        food_pos = []
        for entity in world.food:
            if not entity.boundary:
                food_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)

            oth_f = [
                self.is_collision(other, world.forests[i])
                for i in range(len(world.forests))
            ]

            # without forest vis
            for i in range(len(world.forests)):
                if inf[i] and oth_f[i]:
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
                    if not other.adversary:
                        other_vel.append(other.state.p_vel)
                    break
            else:
                if ((not any(inf)) and (not any(oth_f))) or agent.leader:
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
                    if not other.adversary:
                        other_vel.append(other.state.p_vel)
                else:
                    other_pos.append([0, 0])
                    if not other.adversary:
                        other_vel.append([0, 0])

        # to tell the pred when the prey are in the forest
        prey_forest = []
        ga = self.good_agents(world)
        for a in ga:
            if any([self.is_collision(a, f) for f in world.forests]):
                prey_forest.append(np.array([1]))
            else:
                prey_forest.append(np.array([-1]))
        # to tell leader when pred are in forest
        prey_forest_lead = []
        for f in world.forests:
            if any([self.is_collision(a, f) for a in ga]):
                prey_forest_lead.append(np.array([1]))
            else:
                prey_forest_lead.append(np.array([-1]))

        comm = [world.agents[0].state.c]

        if agent.adversary and not agent.leader:
            return np.concatenate(
                [agent.state.p_vel]
                + [agent.state.p_pos]
                + entity_pos
                + other_pos
                + other_vel
                + in_forest
                + comm
            )
        if agent.leader:
            return np.concatenate(
                [agent.state.p_vel]
                + [agent.state.p_pos]
                + entity_pos
                + other_pos
                + other_vel
                + in_forest
                + comm
            )
        else:
            return np.concatenate(
                [agent.state.p_vel]
                + [agent.state.p_pos]
                + entity_pos
                + other_pos
                + in_forest
                + other_vel
            )
