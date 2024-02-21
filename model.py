# abstract model class
from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, environment, alpha, gamma, temperature):
        self._environment = environment
        self._alpha = alpha
        self._gamma = gamma
        self._temperature = temperature

    @abstractmethod
    def update(self, first_stage_choice, second_stage_choice, reward):
        pass

    @abstractmethod
    def choose_action(self, observations):
        pass

    @property
    def environment(self):
        return self._environment
