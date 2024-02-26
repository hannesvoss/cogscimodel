from dataset import Dataset
from helper import Helper


class TwoStep:
    """
    This class is the environment for the two-step task.

    The environment has two steps. In the first step, the agent can choose between two actions: g0 and g1.
    In the second step, the agent can choose between four actions: p0, p1, p2, and p3.

    The environment has a dataset with the observations for each trial.

    The environment has a method to reset the environment and a method to choose an action.

    The environment has a method to update the environment based on the action chosen by the agent.

    The environment has a method to check if there are more trials.

    The environment has a method to reset the trials.
    """
    VALID_STEP_ONE_ACTIONS = {"g0", "g1"}
    VALID_STEP_TWO_ACTIONS = {"p0", "p1", "p2", "p3"}

    def __init__(self, filename):
        self._dataset = Dataset(filename)
        self._trial_num = 0
        self._step = 0
        self._lastObservation = []

    def reset(self):
        """
        This method resets the environment.

        :return: The first observation, the reward, and a boolean indicating if the trial is done.
        """
        self._step = 0
        self._lastObservation = []
        reward, done = None, False
        return Helper.format_obs(step=0, obs=self._dataset.trials[self._trial_num]["stepOne_Param"]), reward, done

    def reset_trials(self):
        self._trial_num = 0

    # choose an action, potentially be rewarded and end the trial
    def step(self, action):
        obs, reward, done = None, None, False
        if self._step == 0:
            assert action in self.VALID_STEP_ONE_ACTIONS
            if action == "g0":
                obs = self._dataset.trials[self._trial_num]["stepOneTwo_Param"]
            elif action == "g1":
                obs = self._dataset.trials[self._trial_num]["stepTwoTwo_Param"]
            else:
                raise Exception(f"Invalid action supplied for step one: {action}")
            self._lastObservation = Helper.format_obs(step=1, obs=obs)
        elif self._step == 1:
            assert action in self.VALID_STEP_TWO_ACTIONS
            assert action in self._lastObservation
            action = int(action[1])  # extract token index
            reward = self._dataset.trials[self._trial_num]["rewards_Param"][action]
            done = True
            self._trial_num += 1
        self._step += 1
        return Helper.format_obs(step=self._step, obs=obs), reward, done

    def has_more_trials(self):
        return True if self._trial_num < len(self._dataset.trials) else False
