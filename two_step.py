from dataset import Dataset
from helper import Helper


class TwoStep:
    VALID_STEP_ONE_ACTIONS = {"g0", "g1"}
    VALID_STEP_TWO_ACTIONS = {"p0", "p1", "p2", "p3"}

    def __init__(self, filename):
        self._dataset = Dataset(filename)
        self._trial_num = 0
        self._step = 0
        self._lastObservation = []

    # new trial, return the tokens observed at step one
    def reset(self):
        self._step = 0
        self._lastObservation = []
        reward, done = None, False
        return self.format_obs(self._dataset.trials[self._trial_num]["stepOne_Param"]), reward, done

    # prepend color to token index (g for first step, p for second step)
    def format_obs(self, obs, step=None):
        return Helper.format_obs_static(step if step else self._step, obs)

    # choose an action, potentially be rewarded and end the trial
    def step(self, action):
        obs = None
        reward = None
        done = False
        if self._step == 0:
            assert action in self.VALID_STEP_ONE_ACTIONS
            if action == "g0":
                obs = self._dataset.trials[self._trial_num]["stepOneTwo_Param"]
            elif action == "g1":
                obs = self._dataset.trials[self._trial_num]["stepTwoTwo_Param"]
            else:
                raise Exception(f"Invalid action supplied for step one: {action}")
            self._lastObservation = self.format_obs(obs, step=1)
        elif self._step == 1:
            assert action in self.VALID_STEP_TWO_ACTIONS
            assert action in self._lastObservation
            action = int(action[1])  # extract token index
            reward = self._dataset.trials[self._trial_num]["rewards_Param"][action]
            done = True
            self._trial_num += 1
        self._step += 1
        obs = self.format_obs(obs)
        return obs, reward, done

    def has_more_trials(self):
        return True if self._trial_num < len(self._dataset.trials) else False
