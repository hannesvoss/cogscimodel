import numpy as np

from model import Model


class TDLR(Model):
    def __init__(self, environment, alpha, gamma, temperature):
        super().__init__(environment, alpha, gamma, temperature)
        self._state_values = {'g0': 0.0, 'g1': 0.0, 'p0': 0.0, 'p1': 0.0, 'p2': 0.0, 'p3': 0.0}  # {'g1': 0.0, 'g2': 0.0, 'p1': 0.0, 'p2': 0.0, 'b1': 0.0, 'b2': 0.0}

    def _calculate_delta(self, stage, value_first_stage_choice, value_second_stage_choice, reward):
        if stage == 1:
            _delta = self._gamma * (value_second_stage_choice - value_first_stage_choice)
        elif stage == 2:
            _delta = self._gamma * (reward - value_second_stage_choice)
        else:
            raise ValueError("Invalid stage")
        return _delta

    def _softmax(self, x, temperature):  # could be unstable if temperature is too low (1e-3)
        x = np.array(x)
        return np.exp(x / temperature) / np.exp(x / temperature).sum()

    def choose_action(self, observations):
        obs_values = [self._state_values[token] for token in observations]
        obs_values_distribution = self._softmax(obs_values, self._temperature)
        return np.random.choice(observations, p=obs_values_distribution)

    def update(self, first_stage_choice, second_stage_choice, reward):
        delta1 = self._calculate_delta(
            1, self._state_values[first_stage_choice], self._state_values[second_stage_choice], reward
        )
        self._state_values[first_stage_choice] += self._alpha * delta1

        delta2 = self._calculate_delta(
            2, self._state_values[first_stage_choice], self._state_values[second_stage_choice], reward
        )
        self._state_values[second_stage_choice] += self._alpha * delta2

    @property
    def environment(self):
        return self._environment
