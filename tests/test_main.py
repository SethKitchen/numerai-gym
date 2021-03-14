#!/usr/bin/env python

# Core Library
import unittest

# Third party
import gym

# First party
import gym_numerai  # noqa


class Environments(unittest.TestCase):
    def test_env(self):
        env = gym.make("numerai-v0")
        env.seed(0)
        env.reset()
        env.step(0)
