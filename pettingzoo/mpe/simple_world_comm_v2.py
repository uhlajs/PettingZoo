import functools

from pettingzoo.utils.conversions import parallel_wrapper_fn

from ._mpe_utils.manual_control import manual
from ._mpe_utils.simple_env import SimpleDiscreateEnv, make_env
from .scenarios.simple_world_comm import Scenario


class raw_env(SimpleDiscreateEnv):
    def __init__(self, num_good=2, num_adversaries=4, num_obstacles=1, num_food=2, max_cycles=25, num_forests=2):
        scenario = Scenario()
        world = scenario.make_world(num_good, num_adversaries, num_obstacles, num_food, num_forests)
        super().__init__(scenario, world, max_cycles)
        self.metadata['name'] = "simple_world_comm_v2"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
manual_control = functools.partial(manual, parallel_env=parallel_env)
