import logging



log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

## env wrapper for stable baselines
import itertools

import gym
import numpy as np


class EnvCityGym(gym.Env):
    """
    Env wrapper coming from the gym library. For a single agent controling multiple buildings
    """

    def __init__(self, env):
        self.env = env
        self.metadata = self.env.metadata

        # get the number of buildings
        self.num_buildings = len(env.action_space)
        from src.helpers import dump
        # define action and observation space
        self.action_space = gym.spaces.Box(low=np.array([-1] * self.num_buildings),
                                           high=np.array([1] * self.num_buildings), dtype=np.float32)

        # define the observation space

        self.obs_lows =  np.array([])
        self.obs_highs = []
        for obs_box in env.observation_space:
            self.obs_lows = np.concatenate((self.obs_lows, obs_box.low))
            self.obs_highs = np.concatenate((self.obs_highs, obs_box.high))

        #todo: CityLearn have multiple boxes (one for each building)
        #We need to join the boxes on a single one:
        #todo: Move this to a make function and leave this class with the esentials ??
        self.observation_space = gym.spaces.Box(low=np.array(self.obs_lows), high=np.array( self.obs_highs),
                                                dtype=np.float32)


        # TO THINK : normalize the observation space

    def reset(self):
        obs = self.env.reset()

        observation = self.get_observation(obs)

        return observation

    def get_observation(self, obs):
        obs_list =  np.array([])
        for obs_box in obs:
            obs_list = np.concatenate((obs_list, obs_box))

        return obs_list

    def step(self, action):
        """
        we apply the same action for all the buildings
        """
        # reprocessing action
        action = [[act] for act in action]

        # we do a step in the environment
        obs, reward, done, info = self.env.step(action)

        observation = self.get_observation(obs)

        return observation, sum(reward), done, info

    def render(self, mode='human'):
        return self.env.render(mode)
