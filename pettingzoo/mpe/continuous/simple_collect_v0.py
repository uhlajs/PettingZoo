import functools

from pettingzoo.utils.conversions import parallel_wrapper_fn

from .._mpe_utils.manual_control import manual
from .._mpe_utils.simple_env import SimpleContinuousEnv, make_env
from ..scenarios.simple_collect import Scenario


class raw_env(SimpleContinuousEnv):
    def __init__(self, N=3, local_ratio=0.5, max_cycles=25):
        assert 0. <= local_ratio <= 1., "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N)
        super().__init__(scenario, world, max_cycles, local_ratio)
        self.metadata['name'] = "simple_collect_continuous_v2"


env = make_env(raw_env, continuous=True)
parallel_env = parallel_wrapper_fn(env)
manual_control = functools.partial(manual, parallel_env=parallel_env, continuous=True)
