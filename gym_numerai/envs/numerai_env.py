#!/usr/bin/env python

"""
Numerai environment.

Each episode is a round
"""

# Core Library
import logging.config
import math
import random
from typing import Any, Dict, List, Tuple

# Third party
import cfg_load
import gym
import numpy as np
import pandas as pd
import pkg_resources
from gym import spaces

path = "config.yaml"  # always use slash in packages
filepath = pkg_resources.resource_filename("gym_numerai", path)
config = cfg_load.load(filepath)
logging.config.dictConfig(config["LOGGING"])

# Global funcs


### Environment
class numeraiEnv(gym.Env):
    """
    Define a simple numerai environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self) -> None:
        self.__version__ = "0.1.0"
        logging.info(f"numeraiEnv - Version {self.__version__}")
        # download the latest training dataset (takes around 30s)
        self.training_data = pd.read_csv("https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz")
        self.training_data['era'] = self.training_data['era'].str.replace('era', '').astype(int)
        self.training_data = self.training_data.drop('id', 1)
        self.training_data = self.training_data.drop('data_type',1)
        self.targets = self.training_data['target']
        self.training_data = self.training_data.drop('target',1)
        self.curr_step = 0

        # Define what the agent can do -- Make a prediction
        # Target Column
        self.action_space = spaces.Box(np.array([0]),np.array([1]))

        # Observation is the current id features
        low = np.array([0]*311)  # remaining_tries
        high = np.array([1]*311)  # remaining_tries
        high[0] = 200
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory: List[Any] = []

    def step(self, action: float) -> Tuple[List[int], float, bool, Dict[Any, Any]]:
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : float

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob : List[int]
                an environment-specific object representing your observation of
                the environment.
            reward : float
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over : bool
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info : Dict
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        if(self.curr_step == len(self.training_data)):
            raise RuntimeError("Episode is done")
        self._take_action(action)
        reward = self._get_reward()
        self.curr_step += 1
        ob = self._get_state()
        return ob, reward, self.curr_step == len(self.training_data), {}

    def _take_action(self, action: float) -> None:
        self.action_episode_memory[self.curr_episode].append(action)
        self.target = action

    def _get_reward(self) -> float:
        """Reward is given for a sold numerai."""
        mse = (self.targets[self.curr_step]-self.target)*(self.targets[self.curr_step]-self.target)
        return 1.0 - mse

    def reset(self) -> List[int]:
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation: List[int]
            The initial observation of the space.
        """
        self.curr_step = 0
        self.curr_episode += 1
        self.action_episode_memory.append([])
        return self._get_state()

    def _render(self, mode: str = "human", close: bool = False) -> None:
        return None

    def _get_state(self) -> List[int]:
        """Get the observation."""
        ob = self.training_data.iloc[self.curr_step].to_numpy()
        return ob

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed
