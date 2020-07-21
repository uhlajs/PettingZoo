from gym.spaces import Discrete, Box
import numpy as np
import warnings
import magent
from pettingzoo import AECEnv
import math
from pettingzoo.magent.render import Renderer
from pettingzoo.utils import agent_selector
from .magent_env import magent_parallel_env
from pettingzoo.utils._parallel_env import _parallel_env_wrapper
import random

def env(map_size=100, seed=None):
    return _parallel_env(gather_markov_env(map_size, seed))


def load_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size * 2, "map_height": map_size})
    cfg.set({"minimap_mode": True})
    cfg.set({"embedding_size": 10})

    agent = cfg.register_agent_type(
        "agent",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 1,
         'view_range': gw.CircleRange(6),
         'damage': 2, 'step_recover': 0.1,

         'step_reward': -1,
         })

    g0 = cfg.add_group(agent)

    return cfg


def generate_map(env, map_size, handles):
    """ generate a map, which consists of two squares of agents and vertical lines"""
    width = map_size * 2
    height = map_size
    margin = map_size * 0.1
    line_num = 9
    wall_width = 4
    gap = 2
    road_height = 2
    road_num = 4
    init_num = margin * height * 0.8

    def random_add(x1, x2, y1, y2, n):
        added = set()
        ct = 0
        while ct < n:
            x = random.randint(x1, x2)
            y = random.randint(y1, y2)

            next = (x, y)
            if next in added:
                continue
            added.add(next)
            ct += 1
        return list(added)

    # left
    pos = random_add(0, margin, 0, height, init_num)
    env.add_agents(handles[0], method="custom", pos=pos)

    # right
    # pos = random_add(width - margin, width, 0, height, init_num)
    # env.add_agents(handles[rightID], method="custom", pos=pos)

    # wall
    lines = set()
    low, high = margin * 2 + wall_width, width - margin * 2 - wall_width
    ct = 0
    while ct < line_num:
        next = random.randint(low, high)
        collide = False
        for j in range(-wall_width - gap, wall_width+gap + 1):
            if next+j in lines:
                collide = True
                break

        if collide:
            continue
        lines.add(next)
        ct += 1

    lines = list(lines)
    walls = []
    for item in lines:
        road_skip = set()
        for i in range(road_num):
            road_start = random.randint(1, height-1 - road_height)
            for j in range(road_height):
                road_skip.add(road_start + j)

        for i in range(height):
            if i in road_skip:
                continue
            for j in range(-wall_width//2, wall_width//2 + 1):
                walls.append((item+j, i))

    env.add_walls(method="custom", pos=walls)


class gather_markov_env(markov_env):
    def __init__(self, map_size, seed):
        env = magent.GridWorld(load_config(map_size=map_size))
        handles = env.get_handles()

        names = ["agent"]
        super().__init__(env, handles[:1], names, map_size, seed)

    def generate_map(self):
        generate_map(self.env, self.map_size, self.handles)
