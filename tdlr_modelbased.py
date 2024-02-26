from model import Model


class TDLRModelBased(Model):
    """
    This class is a subclass of the Model class and implements a model-based version of the TDLR model.
    """
    def __init__(self, environment, alpha, gamma, temperature):
        super().__init__(environment, alpha, gamma, temperature)
        # initialize transition probabilities to 0
        self._transition_probabilities = {'g0-p': 0.0, 'g1-p': 0.0, 'g0-b': 0.0, 'g1-b': 0.0}

    def update(self, first_choice, second_choice, reward):
        """
        This method updates the state values for the first and second choices.

        :param first_choice: The first choice made by the agent
        :param second_choice: The second choice made by the agent
        :param reward: The reward received by the agent
        :return: None
        """
        delta = self._calculate_delta(
            1, self._state_values[first_choice], self._state_values[second_choice], reward
        )
        self._state_values[first_choice] += self._transition_probabilities[f"{first_choice}-{second_choice[:-1]}"] * self._alpha * delta

        delta = self._calculate_delta(
            2, self._state_values[first_choice], self._state_values[second_choice], reward
        )
        self._state_values[second_choice] += self._alpha * delta

    def update_transition_probabilities(self, transition_count, first_stage_count):
        """
        This method updates the transition probabilities based on the counts of the transitions.

        :param transition_count: The counts of the transitions
        :param first_stage_count: The counts of the first stage choices
        :return: None
        """
        # update probabilities for each transition but only if the count is not 0
        if first_stage_count['g0'] > 0:
            self._transition_probabilities['g0-p'] = transition_count['g0-p'] / first_stage_count['g0']
            self._transition_probabilities['g0-b'] = transition_count['g0-b'] / first_stage_count['g0']
        if first_stage_count['g1'] > 0:
            self._transition_probabilities['g1-p'] = transition_count['g1-p'] / first_stage_count['g1']
            self._transition_probabilities['g1-b'] = transition_count['g1-b'] / first_stage_count['g1']
