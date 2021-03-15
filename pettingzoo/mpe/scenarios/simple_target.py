import numpy as np
from .._mpe_utils.core import World, Agent, Landmark
from .._mpe_utils.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, N=3):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = N
        num_landmarks = N
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent_{}'.format(i)
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0, 51, 204]) / 255
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([89, 204, 51]) / 255
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def global_reward(self, world):
        """
        # Original reward
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        return rew
        """

        """
        # Touching bonus
        rew = 0
        for l in world.landmarks:
            dists = [d if (d := (np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) - (a.size + l.size))) > 0 else -10 
                     for a in world.agents]
            rew -= min(dists)
        return rew
        """
        landmark_bonus = 10

        def distance(agent, landmark):
            return np.linalg.norm(agent.state.p_pos - landmark.state.p_pos, ord=2)

        distance_matrix = np.array([[distance(a, landmark) for a in world.agents] for landmark in world.landmarks])
        touch_size = np.array([[a.size + landmark.size for a in world.agents] for landmark in world.landmarks])
        touch_matrix = distance_matrix < touch_size
        inactive_landmark = np.sum(touch_matrix, axis=1) == 0
        inactive_agent = np.sum(touch_matrix, axis=0) == 0
        reduce_distance_matrix = distance_matrix[inactive_landmark, :]
        reduce_distance_matrix = reduce_distance_matrix[:, inactive_agent]

        active_landmark_bonus = (len(inactive_landmark) - np.sum(inactive_landmark)) * landmark_bonus
        if reduce_distance_matrix.size == 0:
            distance_penalty = 0
        else:
            distance_penalty = -np.sum(np.min(reduce_distance_matrix, axis=1))

        # print(distance_matrix)
        # print(touch_size)
        # print(touch_matrix)
        # print(reduce_distance_matrix)

        # print(active_landmark_bonus)
        # print(distance_penalty)
        return active_landmark_bonus + distance_penalty

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
