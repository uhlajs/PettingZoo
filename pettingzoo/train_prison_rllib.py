from pettingzoo.utils.pom_game_wrapper import POMGameEnv
from pettingzoo.gamma import prison
from copy import deepcopy
import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env
from typing import Dict, Tuple

'''For this script, you need:
1. Algorithm name and according module, e.g.: 'PPo' + agents.ppo as agent
2. Name of the aec game you want to train on, e.g.: 'prison'.
3. num_cpus
4. num_rollouts
'''

alg_name = 'PPO'
game_name = 'prison'
num_cpus= 2
num_rollouts = 2

# 1. Get's default training configuration and specifies the AECgame to load.
def get_default_config_with_aec(alg_name='PPO', game_name='prison'):
    agent_cls = get_agent_class(alg_name)
    config = deepcopy(agent_cls._default_config)

    def add_game_name(config, game_name) -> Dict:
        config['env_config'] = {'aec_env': game_name, 'run': alg_name}
        return config

    return add_game_name(config, game_name)


custom_config = get_default_config_with_aec(alg_name=alg_name,
                                            game_name=game_name)

# 2. Register env
register_env(game_name, lambda env_config: POMGameEnv(env_config))

# 3. Extracts action_spaces and observation_spaces from environment instance
def get_spaces(input_config) -> Tuple:
    test_env = POMGameEnv(input_config['env_config'])
    obs = test_env.observation_space
    act = test_env.action_space
    test_env.close()
    return obs, act

obs_space, act_space = get_spaces(input_config=custom_config)

# 4. Configuration for multiagent setup with policy sharing:
custom_config["multiagent"] = {
        "policies": {
            # the first tuple value is None -> uses default policy
            "av": (None, obs_space, act_space, {}),
        },
        "policy_mapping_fn": lambda agent_id: 'av'
        }

# 5. Initialize ray and trainer object
ray.init(num_cpus=num_cpus+1)

# 6. Adding range scale to demonstrate wrapper functionality
custom_config['env_config']['range_scale'] = (-300, 300)
custom_config['env_config']['set_all_done'] = True     # Default = True
custom_config['env_config']['game_args'] = None


# custom_config['normalize_actions'] = True  # Not working at the moment.
custom_config['log_level'] = 'DEBUG'
custom_config['num_workers'] = 1
custom_config['sample_batch_size'] = 10     # Fragment length, collected at once from each worker and for each agent!
custom_config['train_batch_size'] = 200     # Training batch size -> Fragments are concatenated up to this point.
custom_config['horizon'] = 300              # After n steps, force reset simulation
custom_config['no_done_at_end'] = False     # Default: False
# Info: If False, each agents trajectory is expected to have maximum one done=True in the last step of the trajectory.
# If no_done_at_end = True, dones are ignored.

trainer = get_agent_class(alg_name)(env=game_name, config=custom_config)

# 7. Train 10 iterations
for i in range(10):
    trainer.train()
    print(i)