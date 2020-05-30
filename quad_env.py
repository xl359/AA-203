import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from quad_sim.quadFiles.quad import Quadcopter
from quad_sim.utils.windModel import Wind


class QuadEnv(gym.Env):
    def __init__(self):
        # integrations time step size in sec
        self.t_step = 0.1
        self.current_time = 0
        self.quad = Quadcopter(self.current_time)
        self.wind = Wind('None')
        # lower and upper bounds for motor command
        self.action_lb = np.zeros(4)
        self.action_ub = 1000 * np.ones(4)
        # lower and upper bounds for quad position, terminate episode if quad goes outside of this box
        self.pos_lb = - 20 * np.ones(3)
        self.pos_ub = 20 * np.ones(3)
        self.Q = np.eye(13)
        # first element of quaternion should be one in stable hover
        self.Q[3, 3] = 0
        self.R = (1 / 500.0**2) * np.eye(4)
        return

    def step(self, action):
        # action should be a 1d np array
        # clip action
        action = np.maximum(self.action_lb, np.minimum(self.action_ub, action))
        self.quad.update(self.current_time, self.t_step, action, self.wind)
        self.current_time += self.t_step
        # state is a 1d np array
        state = self.quad.state
        # calculate rewards
        # for now, just try to hover
        kinematic_states = state[0:13]
        rewards = - (np.dot(kinematic_states, np.dot(self.Q, kinematic_states)) + np.dot(action, np.dot(self.R, action)))
        # terminate episode if quad goes outside of the box
        pos = state[0:3]
        done = False
        if any(pos < self.pos_lb) or any(pos > self.pos_ub):
            done = True
        return state, rewards, done, {}

    def reset(self):
        self.current_time = 0.0
        self.quad.reset(self.current_time)
        return

    def render(self, mode='human'):
        return

    def close(self):
        return


def test():
    env = QuadEnv()
    state, rewards, done, _ = env.step(env.quad.params['w_hover'] * np.ones(4))
    env.reset()
    print('it runs!')
    return

test()
