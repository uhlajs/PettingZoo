from copy import deepcopy
import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env
from pettingzoo.utils.rllib.pom_game_wrapper import POMGameEnv
from pettingzoo.gamma import prison_v0
from supersuit import normalize_obs, dtype, color_reduction

from numpy import float32


'''For this script, you need:
1. Algorithm name and according module, e.g.: 'PPo' + agents.ppo as agent
2. Name of the aec game you want to train on, e.g.: 'prison'.
3. num_cpus
4. num_rollouts

Does require SuperSuit
'''

alg_name = 'PPO'
env_cls = prison_v0
num_cpus= 1
num_rollouts = 2

# 1. Get's default training configuration and specifies the POMgame to load.
config = deepcopy(get_agent_class(alg_name)._default_config)

# 2. Adding range scale to demonstrate wrapper functionality
config['env_config']['wrappers'] = [{'wrapper_function': dtype,
                                     'named_params': {'dtype': float32}},
                                    {'wrapper_function': color_reduction,
                                     'named_params': {'mode': 'R'}},

                                    ]
config['env_config']['game_args'] = None

#ToDo: Debugging, testing, documentation

# 3. Register env
register_env('prison', lambda config: POMGameEnv(env_config=config, env_creator=env_cls.env))

# 4. Extract space dimensions
test_env = POMGameEnv(env_config=config['env_config'], env_creator=env_cls.env)
obs_space = test_env.observation_space
act_space = test_env.action_space
test_env.aec_env.render()
test_env.close()

# 5. Configuration for multiagent setup with policy sharing:
config["multiagent"] = {
        "policies": {
            # the first tuple value is None -> uses default policy
            "av": (None, obs_space, act_space, {}),
        },
        "policy_mapping_fn": lambda agent_id: 'av'
        }

config['log_level'] = 'DEBUG'
config['num_workers'] = 1
config['sample_batch_size'] = 30     # Fragment length, collected at once from each worker and for each agent!
config['train_batch_size'] = 200     # Training batch size -> Fragments are concatenated up to this point.
config['horizon'] = 200              # After n steps, force reset simulation
config['no_done_at_end'] = False     # Default: False
# Info: If False, each agents trajectory is expected to have maximum one done=True in the last step of the trajectory.
# If no_done_at_end = True, environment is not resetted when dones[__all__]= True.


# 6. Initialize ray and trainer object
ray.init(num_cpus=num_cpus+1)
trainer = get_agent_class(alg_name)(env='prison', config=config)

# 7. Train once
trainer.train()

test_env.reset()