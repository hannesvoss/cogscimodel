from model import Model


class TDLRModelBased(Model):
    def __init__(self, environment, alpha, gamma, temperature):
        super().__init__(environment, alpha, gamma, temperature)
        self._transition_probabilities = {'g0-g': 0.0, 'g1-g': 0.0, 'g0-p': 0.0, 'g1-p': 0.0}
        # self._reward_probabilities = {'p1': 0.25, 'p2': 0.6, 'b1': 0.7, 'b2': 0.4}
        # self._episode_data = []
        # self._transition_probabilities_list = []

    def update(self, first_stage_choice, second_stage_choice, reward):
        delta1 = self._calculate_delta(
            1, self._state_values[first_stage_choice], self._state_values[second_stage_choice], reward
        )
        self._state_values[first_stage_choice] += self._transition_probabilities[f"{first_stage_choice}-{second_stage_choice[:-1]}"] * self._alpha * delta1

        delta2 = self._calculate_delta(
            2, self._state_values[first_stage_choice], self._state_values[second_stage_choice], reward
        )
        self._state_values[second_stage_choice] += self._alpha * delta2

    def update_transition_probabilities(self, transition_count, first_stage_count):
        # update probabilities for each transition but only if the count is not 0
        if first_stage_count['g0'] > 0:
            self._transition_probabilities['g0-g'] = transition_count['g0-g'] / first_stage_count['g0']
            self._transition_probabilities['g0-p'] = transition_count['g0-p'] / first_stage_count['g0']
        if first_stage_count['g1'] > 0:
            self._transition_probabilities['g1-g'] = transition_count['g1-g'] / first_stage_count['g1']
            self._transition_probabilities['g1-p'] = transition_count['g1-p'] / first_stage_count['g1']
