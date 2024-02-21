from model import Model


class TDLR(Model):
    def __init__(self, environment, alpha, gamma, temperature):
        super().__init__(environment, alpha, gamma, temperature)

    def update(self, first_stage_choice, second_stage_choice, reward):
        delta1 = self._calculate_delta(
            1, self._state_values[first_stage_choice], self._state_values[second_stage_choice], reward
        )
        self._state_values[first_stage_choice] += self._alpha * delta1

        delta2 = self._calculate_delta(
            2, self._state_values[first_stage_choice], self._state_values[second_stage_choice], reward
        )
        self._state_values[second_stage_choice] += self._alpha * delta2
