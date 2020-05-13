from ray.rllib.env import MultiAgentEnv
from pettingzoo.utils.pom_registry import get_game_class
from supersuit import   color_reduction, \
                        continuous_actions, \
                        down_scale, \
                        dtype, \
                        flatten, \
                        frame_stack, \
                        normalize_obs, \
                        reshape


class POMGameEnv(MultiAgentEnv):
    """An interface to the PettingZoo MARL environment library.

    This class inherents from MultiAgentEnv and exposes a given AEC
    (actor-environment-cycle) game from the PettingZoo project via the
    MultiAgentEnv public API.

    It reduces the class of AEC games to Partially Observable Markov (POM) games by imposing the following
    important restrictions onto an AEC environment:

    1. Each agent is listed exactly ones in agent_order.
    2. The order of the agents in agent_order does not change over time.
    3. Agents act simultaneously (-> No hard-turn games like chess).
    4. Environments are positive sum games (-> Agents are expected to cooperate to maximize reward).
    5. All agents have the same action_spaces and observation_spaces.
    Note: If, within your aec game, agents do not have homogeneous action / observation spaces, use the wrapper class
    from PettingZoo to apply padding functionality.
    6. By default: If all agents are done, the simulation signals termination and is restarted.
    ToDo: Check & update description and example, add env_config_dict_description

    Examples:
        >>> from pettingzoo.gamma import prison
        >>> env = POMGameEnv(env_config={'aec_env': 'simple_spread'})
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

    def __init__(self, env_config):

        # choose the game
        if 'aec_env' in env_config.keys():

            # Pass all optional environment params
            if 'game_args' in env_config.keys() and isinstance(env_config['game_args'], dict):
                self.aec_env = get_game_class(env_config['aec_env']).env(**env_config['game_args'])
            else:
                self.aec_env = get_game_class(env_config['aec_env']).env()
        else:
            raise ValueError('POM game not specified. Consult utils/pom_registry.py to see valid inputs.')

        # if flag is specified and False, then disable functionality in
        if 'set_all_done' in env_config.keys() and not env_config['set_all_done']:
            self.set_all_done = False
        else:
            # Default behavior if not specified
            self.set_all_done = True

        # instantiate
        if any(('color_reduction' in env_config.keys(), #{'full', 'R', 'G', 'B'}
                'continuous_actions' in env_config.keys(), #Union[Bool,[seed:int]]
                'down_scale' in env_config.keys(), #Union[Bool, Tuple[x_scale:int,y_scale:int]]   1,1
                'dtype' in env_config.keys(), #dtype, maybe as string 'Many graphical games return uint8 observations, while neural networks generally want float16 or float32.
                'flatten' in env_config.keys(), #Bool
                'frame_stack' in env_config.keys(), #Optional[num_frames:int]   4
                'normalize_obs' in env_config.keys(), #Union[Bool, Tuple[env_min:int,env_max:int]]   0,1
                'reshape' in env_config.keys(), #Tuple[int, int]
                'agent_indicator' in env_config.keys(),     #Bool
                'pad_action_space' in env_config.keys(),    #Bool
                'pad_observations' in env_config.keys(),    #Bool

                # To be used in Rllib, all params in env_config have to be pickable.
                # Hence lambdas are not supported for now.
                'action_lambda' in env_config.keys(),
                'observations_lambda' in env_config.keys()),
               ):

            self.wrap = True

            # ToDo: Change to using supersuit, soround with try catch block
            self.aec_env = wrapper(self.aec_env,
                                   color_reduction=env_config.get('color_reduction', None),
                                   down_scale=env_config.get('down_scale', None),
                                   reshape=env_config.get('reshape', None),
                                   range_scale=env_config.get('range_scale', None),
                                   new_dtype=env_config.get('new_dtype', None),
                                   continuous_actions=env_config.get('continuous_actions', False),
                                   frame_stacking=env_config.get('frame_stacking', 1)
                                   )

        else:
            self.wrap = False

        # agent idx list
        self.agents = self.aec_env.agents

        # ToDo: Inspect if this code block is still neeeded, probably not, since _modify_spaces is called when initializing a Wrapper instance.
        if self.wrap:
            # Get modified dictionaries from wrapper
            self.aec_env.modify_observation_space()
            self.aec_env.modify_action_space()

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

            # For agents with done = True, remove from dones, rewards and observationss
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