# abstract model class
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
    def update(self, first_stage_choice, second_stage_choice, reward):
        pass

    def _calculate_delta(self, stage, value_first_stage_choice, value_second_stage_choice, reward):
        if stage == 1:
            _delta = self._gamma * (value_second_stage_choice - value_first_stage_choice)
        elif stage == 2:
            _delta = self._gamma * (reward - value_second_stage_choice)
        else:
            raise ValueError("Invalid stage")
        return _delta

    def choose_action(self, observations):
        obs_values = [self._state_values[token] for token in observations]
        obs_values_distribution = Helper.softmax(obs_values, self._temperature)
        return np.random.choice(observations, p=obs_values_distribution)

    @property
    def environment(self):
        return self._environment
