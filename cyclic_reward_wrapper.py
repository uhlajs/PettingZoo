from supersuit.base_aec_wrapper import PettingzooWrap
from collections import deque


class queuesum:
    def __init__(self):
        self.queue = deque()
        self.sum = 0
        self.size = 0

    def add(self, item):
        self.queue.append(item)
        self.sum += item
        if len(self.queue) > self.size:
            self.sum -= self.queue[0]
            self.queue.popleft()
        return self.sum

    def resize(self, new_size):
        assert new_size >= self.size
        self.size = new_size

    def __str__(self):
        return f"{self.queue}"


class cyclically_expansive_learning(PettingzooWrap):
    def __init__(self, env, curriculum):
        '''
        The curriculum is a sorted list of tuples:
        (schedual_step, reward_steps_to_sum)
        '''
        assert curriculum == list(sorted(curriculum))
        self.curriculum = curriculum
        self.env_step = 0
        self.curriculum_step = 0
        super().__init__(env)

    def _check_wrapper_params(self):
        pass

    def _modify_spaces(self):
        pass

    def reset(self):
        super().reset()
        self.reward_queues = {agent: queuesum() for agent in self.agents}
        for qs in self.reward_queues.values():
            qs.resize(self.curriculum[0][1])
        self._cumulative_rewards = {a: 0 for a in self.agents}

    def step(self, action):
        agent = self.env.agent_selection
        super().step(action)
        if self.curriculum_step < len(self.curriculum)-1 and self.env_step >= self.curriculum[self.curriculum_step+1][0]:
            self.curriculum_step += 1
            num_cycles_keep = self.curriculum[self.curriculum_step][1]
            for qs in self.reward_queues.values():
                qs.resize(num_cycles_keep)

        self._cumulative_rewards = {a: self.reward_queues[a].add(r) for a, r in self.env.rewards.items()}
        self.env_step += 1

        #agent = self.env.agent_selection
        #_, rew, _, _ = self.env.last(observe=False)
        #print("Step: ", self.env_step)
        #print(f"  Agent: {agent} - reward: {self.env.rewards[agent]:1.5f}, cum rew: {self.env._cumulative_rewards[agent]:1.5f}, last: {rew:1.5f}")
        #print(f"  deque: {self.reward_queues[agent]}")
        #print("   rewards: ",[f"{r:1.5f}" for r in self.rewards.values()])
        #print("   cumulat: ",[f"{r:1.5f}" for r in self._cumulative_rewards.values()])
