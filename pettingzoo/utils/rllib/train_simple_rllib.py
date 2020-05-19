from pettingzoo.utils.rllib.pom_game_wrapper import POMGameEnv
from copy import deepcopy
import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env
from pettingzoo.mpe import simple_spread_v0
from supersuit import continuous_actions

'''For this script, you need:
1. Algorithm name and according module, e.g.: 'PPo' + agents.ppo as agent
2. Name of the aec game you want to train on, e.g.: 'prison'.
3. num_cpus
4. num_rollouts

Does not use SuperSuit
'''

alg_name = 'PPO'
env_cls = simple_spread_v0
num_cpus= 1
num_rollouts = 2

# 1. Get's default training configuration and specifies the POMgame to load.
config = deepcopy(get_agent_class(alg_name)._default_config)

# 2. Register env
register_env('prison', lambda env_config: POMGameEnv(env_config=env_config, env_creator=env_cls.env))

# 3. Get space dimensions
test_env = POMGameEnv(env_config=config, env_creator=env_cls.env)
obs_space = test_env.observation_space
act_space = test_env.action_space
test_env.close()

# 4. Configuration for multiagent setup with policy sharing:
config["multiagent"] = {
        "policies": {
            # the first tuple value is None -> uses default policy
            "av": (None, obs_space, act_space, {}),
        },
        "policy_mapping_fn": lambda agent_id: 'av'
        }

# 5. Training configs
config['log_level'] = 'DEBUG'
config['num_workers'] = 4
config['sample_batch_size'] = 200     # Fragment length, once from each worker. Rollout is divided into fragments
config['train_batch_size'] = 4000     # Training batch size -> Fragments are concatenated up to this point.
config['horizon'] = 100              # After 100 steps, force reset simulation
config['no_done_at_end'] = False


# 6. Initialize ray and trainer object
ray.init(num_cpus=num_cpus+1)

# 7. Train once
trainer = get_agent_class(alg_name)(env='prison', config=config)
trainer.train()

# 8. Apply the trainer