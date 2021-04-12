import functools

from pettingzoo.utils.conversions import parallel_wrapper_fn

from .._mpe_utils.manual_control import manual
from .._mpe_utils.simple_env import SimpleContinuousEnv, make_env
from ..scenarios.simple_confuse import Scenario


class raw_env(SimpleContinuousEnv):
    def __init__(self, N=2, max_cycles=25):
        scenario = Scenario()
        world = scenario.make_world(N)
        super().__init__(scenario, world, max_cycles)
        self.metadata['name'] = "simple_confuse_continuous_v0"


env = make_env(raw_env, continuous=True)
parallel_env = parallel_wrapper_fn(env)
manual_control = functools.partial(manual, parallel_env=parallel_env, continuous=True)
