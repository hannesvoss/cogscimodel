from model import Model


class TDLR(Model):
    """
    This class is a subclass of the Model class and implements a model-free version of the TDLR model.
    """
    def __init__(self, environment, alpha, gamma, temperature):
        super().__init__(environment, alpha, gamma, temperature)

    def update(self, first_choice, second_choice, reward):
        """
        This method updates the state values for the first and second choices.

        :param first_choice: The first choice made by the agent
        :param second_choice: The second choice made by the agent
        :param reward: The reward received by the agent
        :return: None
        """
        delta = self._calculate_delta(1, self._state_values[first_choice], self._state_values[second_choice], reward)
        self._state_values[first_choice] += self._alpha * delta

        delta = self._calculate_delta(2, self._state_values[first_choice], self._state_values[second_choice], reward)
        self._state_values[second_choice] += self._alpha * delta
