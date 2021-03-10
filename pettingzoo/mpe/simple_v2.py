import functools

from pettingzoo.utils.conversions import parallel_wrapper_fn

from ._mpe_utils.manual_control import manual
from ._mpe_utils.simple_env import SimpleDiscreateEnv, make_env
from .scenarios.simple import Scenario


class raw_env(SimpleDiscreateEnv):
    def __init__(self, max_cycles=25):
        scenario = Scenario()
        world = scenario.make_world()
        super().__init__(scenario, world, max_cycles)
        self.metadata['name'] = "simple_v2"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
manual_control = functools.partial(manual, parallel_env=parallel_env)
