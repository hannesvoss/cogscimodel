from abc import ABC, abstractmethod

import numpy as np

from helper import Helper


class Model(ABC):
    def __init__(self, environment, alpha, gamma, temperature):
        self._state_values = {'g0': 0.0, 'g1': 0.0, 'p0': 0.0, 'p1': 0.0, 'p2': 0.0, 'p3': 0.0}
        self._environment = environment
        self._alpha = alpha
        self._gamma = gamma
        self._temperature = temperature

    @abstractmethod
    def update(self, first_choice, second_choice, reward):
        pass

    def _calculate_delta(self, stage, value_first_choice, value_second_choice, reward):
        if stage == 1:
            _delta = self._gamma * (value_second_choice - value_first_choice)
        elif stage == 2:
            _delta = self._gamma * (reward - value_second_choice)
        else:
            raise ValueError("Invalid stage")
        return _delta

    def choose_action(self, observations):
        # get the state values for the observations
        obs_values = [self._state_values[token] for token in observations]
        # apply softmax to the values (if temperature is small it will be close to argmax)
        obs_values_distribution = Helper.softmax(obs_values, self._temperature)
        # take an action according to the distribution
        return np.random.choice(observations, p=obs_values_distribution)

    @property
    def environment(self):
        return self._environment
