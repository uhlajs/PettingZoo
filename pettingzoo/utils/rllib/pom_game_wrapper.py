from ray.rllib.env import MultiAgentEnv

class POMGameEnv(MultiAgentEnv):
    """An interface to the PettingZoo MARL environment library.
    See: https://github.com/PettingZoo-Team/PettingZoo

    Inherits from MultiAgentEnv and exposes a given AEC
    (actor-environment-cycle) game from the PettingZoo project via the
    MultiAgentEnv public API.

    It reduces the class of AEC games to Partially Observable Markov (POM) games by imposing the following
    important restrictions onto an AEC environment:

    1. Each agent is listed exactly ones in agent_order.
    2. The order of the agents in agent_order does not change over time.
    3. Agents act simultaneously (-> No hard-turn games like chess).
    4. Environments are positive sum games (-> Agents are expected to cooperate to maximize reward).
    5. All agents have the same action_spaces and observation_spaces.
    Note: If, within your aec game, agents do not have homogeneous action / observation spaces, apply SuperSuit wrappers
    to apply padding functionality: https://github.com/PettingZoo-Team/SuperSuit#built-in-multi-agent-only-functions
    6. By default: If all agents are done, the simulation signals termination and is restarted.

    Examples:
        >>> from pettingzoo.gamma import prison
        >>> env = POMGameEnv(env_config={'env_cls': simple_spread_v0})
        >>> obs = env.reset()
        >>> print(obs)

        {
            "0": [110, 119],
            "1": [105, 102],
            "2": [99, 95],
        }
        >>> obs, rewards, dones, infos = env.step(
            action_dict={
                "0": 1, "1": 0, "2": 2,
            })
        >>> print(rewards)
        {
            "0": 0,
            "1": 1,
            "2": 0,
        }
        >>> print(dones)
        {
            "0": False,    # agent 0 is still running
            "1": True,     # agent 1 is done
            "__all__": False,  # the env is not done
        }
        >>> print(infos)
        {
            "0": {},  # info for agent 0
            "1": {},  # info for agent 1
        }
    """

    def __init__(self, env_config, env_creator: object):
        """
        Parameters:
        -----------
        env_config: Dict, specifies:
                    - Optional game-dependent parameters.
                    - Optional SuperSuit wrappers.
                    - Optional flag 'se

                    - Example:
                        env_config = {
                            'game_args': None,      # Optional dict:
                                                    # Passed to environment, when constructed.

                            'wrappers':             # Optional List[Dict]:
                            [                       # SuperSuit wrappers, that the env is wrapped in.
                                {                   # Applied in order of appearence.
                                    'wrapper_function': normalize_obs,
                                    'named_params': {
                                        'env_min': 0,
                                        'env_max': 1
                                    },
                                },
                            ],

                            'set_all_done': False,  # Optional Bool, Default: True:
                                                    # If True: when single agent is done, rollout ends and env resets.
                        }

        env_creator: function, which is called to instantiate an env object.
        """
        self.config = env_config

        # Unpack all optional environment params
        if 'game_args' in self.config.keys() and isinstance(self.config['game_args'], dict):
            self.aec_env = env_creator(**self.config['game_args'])
        else:
            self.aec_env = env_creator()

        self._wrap_env()

        # If True (default): When single agent is done, rollout ends and env is reseted.
        if 'set_all_done' in self.config.keys() and not self.config['set_all_done']:
            self.set_all_done = False
        else:
            # Default behavior if not specified
            self.set_all_done = True

        # agent idx list
        self.agents = self.aec_env.agents

        # Get dictionaries of obs_spaces and act_spaces
        self.observation_spaces = self.aec_env.observation_spaces
        self.action_spaces = self.aec_env.action_spaces

        # Get first observation space, assuming all agents have equal space
        self.observation_space = self.observation_spaces[self.agents[0]]

        # Get first action space, assuming all agents have equal space
        self.action_space = self.action_spaces[self.agents[0]]

        self.rewards = {}
        self.dones = {}
        self.obs = {}
        self.infos = {}

        _ = self.reset()


    def _wrap_env(self):
        # Wraps env object in provided wrapper functions
        if 'wrappers' in self.config.keys():
            for wrapper_dict in self.config['wrappers']:

                if 'named_params' in wrapper_dict.keys():
                    self.aec_env = wrapper_dict['wrapper_function'](env=self.aec_env, **wrapper_dict['named_params'])
                else:
                    self.aec_env = wrapper_dict['wrapper_function'](env=self.aec_env)

    def _init_dicts(self):
        # initialize with zero
        self.rewards = dict(zip(self.agents,
                                [0 for _ in self.agents]))
        # initialize with False
        self.dones = dict(zip(self.agents,
                                [False for _ in self.agents]))
        self.dones['__all__'] = False

        # initialize with None info object
        self.infos = dict(zip(self.agents,
                                [{} for _ in self.agents]))

        # initialize empty observations
        self.obs = dict(zip(self.agents,
                            [None for _ in self.agents]))

    def reset(self):
        """
        Resets the env and returns observations from ready agents.

        Returns:
            obs (dict): New observations for each ready agent.
        """
        # 1. Reset environment; agent pointer points to first agent in agent_order.
        self.aec_env.reset(observe=False)

        # 2. Copy agents from environment
        self.agents = self.aec_env.agents

        # 3. Reset dictionaries
        self._init_dicts()

        # 4. Get initial observations
        for agent in self.agents:

            # For each agent get initial observations
            self.obs[agent] = self.aec_env.observe(agent)

        return self.obs

    def step(self, action_dict):
        """
        Executes input actions from RL agents and returns observations from
        environment agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """
        # iterate over self.agents
        for agent in self.agents:

            # Execute only for agents that have not been done in previous steps
            if agent in action_dict.keys():
                # Execute agent action in environment
                self.obs[agent] = self.aec_env.step(action_dict[agent], observe=True)
                # Get reward
                self.rewards[agent] = self.aec_env.rewards[agent]
                # Update done status
                self.dones[agent] = self.aec_env.dones[agent]

            # For agents with done = True, remove from dones, rewards and observations
            else:
                del self.dones[agent]
                del self.rewards[agent]
                del self.obs[agent]
                del self.infos[agent]

        # update self.agents
        self.agents = list(action_dict.keys())

        # Set __all__ on done, whenever all agents are individually done -> Used to signal episode termination.
        if self.set_all_done:
            if all(list(self.dones.values())[:-1]):
                self.dones['__all__'] = True

        # Update infos stepwise, here we do not copy the dictionary,
        # since RLlib expects {}, not None, if no infos are given.
        for agent in self.agents:
            if self.aec_env.infos[agent]:
                self.infos[agent] = self.aec_env.infos[agent]

        return self.obs, self.rewards, self.dones, self.infos

    def render(self, mode='human'):
        self.aec_env.render(mode=mode)

    def close(self):
        self.aec_env.close()

    def with_agent_groups(self, groups, obs_space=None, act_space=None):
        raise NotImplementedError