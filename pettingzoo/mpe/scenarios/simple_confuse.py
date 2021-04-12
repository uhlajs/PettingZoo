import numpy as np
from .._mpe_utils.core import World, Agent, Landmark
from .._mpe_utils.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self, N=2):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = N + 1
        world.num_agents = num_agents
        num_adversaries = 1
        num_landmarks = num_agents - 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = '{}_{}'.format(base_name, base_index)
            agent.collide = False
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        world.agents[0].color = np.array([255, 87, 51]) / 255
        for i in range(1, world.num_agents):
            world.agents[i].color = np.array([0, 51, 204]) / 255
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
        # set goal landmark
        goal = np_random.choice(world.landmarks)
        goal.color = np.array([89, 204, 51]) / 255
        for agent in world.agents:
            agent.goal_a = goal
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def distance(self, first, second):
        return np.linalg.norm(first.state.p_pos - second.state.p_pos, ord=2)

    def agent_reward(self, agent, world):
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = True
        shaped_adv_reward = True
        pos_rew, adv_rew = 0, 0

        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        if shaped_adv_reward:  # distance-based adversary reward
            # adv_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
            adv_rew = sum([self.distance(a, a.goal_a) for a in adversary_agents])

        for a in adversary_agents:
            # if np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) < a.size + a.goal_a.size:
            if self.distance(a, a.goal_a) < a.size + a.goal_a.size:
                adv_rew -= 5

        # Calculate positive reward for agents
        good_agents = self.good_agents(world)
        if shaped_reward:  # distance-based agent reward
            pos_rew = -min(
                # [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
                [self.distance(a, a.goal_a) for a in good_agents])

        # if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]) \
        if min([self.distance(a, a.goal_a) for a in good_agents]) \
                < 2 * agent.goal_a.size:
            pos_rew += 10

        return pos_rew + adv_rew

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        shaped_reward = True
        adv_rew = 0

        if shaped_reward:  # distance-based reward
            # adv_rew = -np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            adv_rew = -self.distance(agent, agent.goal_a)

        # if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < agent.goal_a.size agent.size:
        if self.distance(agent, agent.goal_a) < agent.goal_a.size + agent.size:
            adv_rew += 5
        return adv_rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        if not agent.adversary:
            return np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos)
        else:
            return np.concatenate(entity_pos + other_pos)
