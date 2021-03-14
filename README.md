This repository contains a PIP package which is an OpenAI environment for
numerai stock trading learning.


## Installation

Install the [OpenAI gym](https://gym.openai.com/docs/).

Then install this package via

```
pip install -e .
```

```
pip install -i https://test.pypi.org/simple/ gym-numerai
```

## Usage

```
import gym
import gym_numerai

env = gym.make('numerai-v0')
```

See https://github.com/matthiasplappert/keras-rl/tree/master/examples for some
examples.


## The Environment

Pulls latest numerai training data and sends each row as a state. Take an action from 0-1
and that will be compared against the actual target and returns (1-MSE) as reward. Episode ends when all training data has been run through.
