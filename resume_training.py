import sys
from pettingzoo.sisl import unpruned_pursuit_v0, pursuit_v3
from supersuit import flatten_v0
import ray
from ray.tune.registry import register_env
import ray.rllib.rollout as rollout
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2


"""
nohup python resume_training.py ~/results_unpruned/unpruned/PPO/PPO_pruned_6492d_00000_0_2021-01-03_23-42-51/checkpoint_27700/checkpoint-27700 --run PPO --env unpruned --no-render --track-progress --episodes 60000 --out rollouts.pkl > cont_unpruned_PPO_0.out &

cp ../../results_unpruned/unpruned/PPO/PPO_pruned_6492d_00000_0_2021-01-03_23-42-51/progress.csv initial_progress.csv

"""

tf1, tf, tfv = try_import_tf()

class MLPModelV2(TFModelV2):
        def __init__(self, obs_space, action_space, num_outputs, model_config,
                     name="my_model"):
            super(MLPModelV2, self).__init__(obs_space, action_space, num_outputs, model_config,
                                             name)
            input_layer = tf.keras.layers.Input(
                obs_space.shape,
                dtype=obs_space.dtype)
            layer_1 = tf.keras.layers.Dense(
                400,
                activation="relu",
                kernel_initializer=normc_initializer(1.0))(input_layer)
            layer_2 = tf.keras.layers.Dense(
                300,
                activation="relu",
                kernel_initializer=normc_initializer(1.0))(layer_1)
            output = tf.keras.layers.Dense(
                num_outputs,
                activation=None,
                kernel_initializer=normc_initializer(0.01))(layer_2)
            value_out = tf.keras.layers.Dense(
                1,
                activation=None,
                name="value_out",
                kernel_initializer=normc_initializer(0.01))(layer_2)
            self.base_model = tf.keras.Model(input_layer, [output, value_out])
            self.register_variables(self.base_model.variables)

        def forward(self, input_dict, state, seq_lens):
            model_out, self._value_out = self.base_model(input_dict["obs"])
            return model_out, state

        def value_function(self):
            return  tf.reshape(self._value_out, [-1])


def make_env_creator(env_name, game_env):
    def env_creator(args):
        env = game_env.env()
        env = flatten_v0(env)
        return env
    return env_creator

def get_env(env_name):
    if env_name == 'unpruned':
        game_env = unpruned_pursuit_v0
    elif env_name == 'pruned':
        game_env = pursuit_v3
    elif env_name == 'curriculum':
        game_env = pursuit_v3
    else:
        raise TypeError("{} environment not supported!".format(game_env))
    return game_env
    

if __name__ == "__main__":

    env_name = 'unpruned'

    game_env = get_env(env_name) 
    env_creator = make_env_creator(env_name, game_env)
    register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("MLPModelV2", MLPModelV2)

    parser = rollout.create_parser()
    args = parser.parse_args()
    rollout.run(args, parser)


