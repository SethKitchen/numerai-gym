This repository contains a PIP package which is an OpenAI environment for
simulating an enironment in which numerais get sold.


## Installation

Install the [OpenAI gym](https://gym.openai.com/docs/).

Then install this package via

```
pip install -e .
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

Imagine you are selling numerais. One at a time. And the numerais get bad pretty
quickly. Let's say in 3 days. The probability that I will sell the numerai
is given by

$$p(x) = (1+e)/(1. + e^(x+1))$$

where x-1 is my profit. This x-1 is my reward. If I don't sell the
numerai, the agent gets a reward of -1 (the price of the numerai).
