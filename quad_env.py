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
        self.quad = Quadcopter(self.t_step)
        self.wind = Wind('None')
        self.current_time = 0
        return

    def step(self, action):
        # action should be a 1d np array
        self.quad.update(self.current_time, self.t_step, action, self.wind)
        return

    def reset(self):
        return

