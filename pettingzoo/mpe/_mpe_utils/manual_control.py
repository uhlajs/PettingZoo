import numpy as np
import pyglet


class Policy:
    """Individual agent policy."""

    def __init__(self):
        pass

    def action(self, obs):
        raise NotImplementedError()


class InteractivePolicy(Policy):
    """Interactive policy based on keyboard input."""

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.raw_env = env.aec_env.env.env
        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(self.raw_env.world.dim_c)]
        # register keyboard events with this environment's window
        self.raw_env.viewer.window.on_key_press = self.key_press
        self.raw_env.viewer.window.on_key_release = self.key_release

        self.active_agent = 0
        self.active_comm = 0

    def key_press(self, k, mod):
        # keyboard event callbacks
        if k == pyglet.window.key.LEFT:
            self.move[0] = True
        if k == pyglet.window.key.RIGHT:
            self.move[1] = True
        if k == pyglet.window.key.UP:
            self.move[2] = True
        if k == pyglet.window.key.DOWN:
            self.move[3] = True

    def key_release(self, k, mod):
        # Movement
        if k == pyglet.window.key.LEFT:
            self.move[0] = False
        if k == pyglet.window.key.RIGHT:
            self.move[1] = False
        if k == pyglet.window.key.UP:
            self.move[2] = False
        if k == pyglet.window.key.DOWN:
            self.move[3] = False

        # Communication
        if k == pyglet.window.key.F:
            self.active_agent = (self.active_agent + 1) % len(self.env.possible_agents)
        if k == pyglet.window.key.D:
            self.active_agent = (self.active_agent - 1) % len(self.env.possible_agents)
        if k == pyglet.window.key.A:
            self.active_comm = (self.active_comm + 1) % len(self.raw_env.world.dim_c)
        if k == pyglet.window.key.S:
            self.active_agent = (self.active_agent - 1) % len(self.raw_env.world.dim_c)


class DiscreateInteractivePolicy(InteractivePolicy):
    """Hard-coded policy which deals only with discreate movement, NOT communication."""
    def action(self, obs, a):
        # ignore observation and just act based on keyboard events
        u = 0
        if a == self.env.possible_agents[self.active_agent]:
            if self.move[0]:
                u = 1
            if self.move[1]:
                u = 2
            if self.move[2]:
                u = 4
            if self.move[3]:
                u = 3
        # TODO: Choose action for speaker based on choosen environment
        # It would be better to split this for each environment.
        return u


class ContinuousInteractivePolicy(InteractivePolicy):
    """Hard-coded policy which deals only with continuous movement, NOT communication."""
    def action(self, obs, a):
        # ignore observation and just act based on keyboard events
        u = np.array([0, 0])
        if a == self.env.possible_agents[self.active_agent]:
            if self.move[0]:
                u += np.array([-1, 0])
            if self.move[1]:
                u += np.array([1, 0])
            if self.move[2]:
                u += np.array([0, 1])
            if self.move[3]:
                u += np.array([0, -1])
        # TODO: Choose action for speaker based on choosen environment
        # It would be better to split this for each environment.
        return np.array(u)


def manual(parallel_env, continuous=False, **kwargs):
    env = parallel_env(**kwargs)
    noop = 0 if not continuous else np.array([0, 0])

    obs, dones = env.reset(), {a: False for a in env.possible_agents}
    try:
        # We need to render environment in order to create a env window
        env.render()
        # Create interactive policy and attach keyboard events
        policy = DiscreateInteractivePolicy(env) if not continuous else ContinuousInteractivePolicy(env)

        ep_ret = 0
        t = 0
        while not all(dones.values()):
            # Change sensitivity of control
            if t % 5 == 0:
                # actions = {a: env.action_spaces[a].sample() for a in env.possible_agents}
                actions = {a: policy.action(obs[a], a) for a in env.possible_agents}
            else:
                actions = {a: noop for a in env.possible_agents}
            obs, rewards, dones, _ = env.step(actions)
            print(rewards)
            ep_ret += sum(rewards.values())

            env.render()
            t += 1
        print(ep_ret)
    except Exception as e:
        raise e
    finally:
        env.close()

