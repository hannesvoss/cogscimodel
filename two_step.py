import io
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv


def parse_string_to_list(string):
    return eval(string.replace('t', 'T').replace('f', 'F'))  # this is very hacky


class TwoStep:
    VALID_STEP_ONE_ACTIONS = {"g0", "g1"}
    VALID_STEP_TWO_ACTIONS = {"p0", "p1", "p2", "p3"}

    @staticmethod
    def format_obs_static(step, obs):
        if obs is None:
            return None
        STEP_COLOR = ["g", "p"]
        if obs is list:
            return list(map(lambda o: f"{STEP_COLOR[step]}{o}", obs))
        else:
            return f"{STEP_COLOR[step]}{obs}"

    def __init__(self, csv_filename):
        self._trials = []
        self._trial_num = 0
        self._step = 0
        self._lastObservation = []
        with open(csv_filename) as csv_file:
            reader = csv.DictReader(csv_file)
            for line in reader:
                self._trials.append({k: parse_string_to_list(line[k]) for k in
                                     ('stepOne_Param', 'stepOneTwo_Param', 'stepTwoTwo_Param', 'rewards_Param')})

    # new trial, return the tokens observed at step one
    def reset(self):
        self._step = 0
        self._lastObservation = []
        reward, done = None, False
        return self.format_obs(self._trials[self._trial_num]["stepOne_Param"]), reward, done

    # prepend color to token index (g for first step, p for second step)
    def format_obs(self, obs, step=None):
        return self.format_obs_static(step if step else self._step, obs)

    # choose an action, potentially be rewarded and end the trial
    def step(self, action):
        obs = None
        reward = None
        done = False
        if self._step == 0:
            assert action in self.VALID_STEP_ONE_ACTIONS
            if action == "g0":
                obs = self._trials[self._trial_num]["stepOneTwo_Param"]
            elif action == "g1":
                obs = self._trials[self._trial_num]["stepTwoTwo_Param"]
            else:
                raise Exception(f"Invalid action supplied for step one: {action}")
            self._lastObservation = self.format_obs(obs, step=1)
        elif self._step == 1:
            assert action in self.VALID_STEP_TWO_ACTIONS
            assert action in self._lastObservation
            action = int(action[1])  # extract token index
            reward = self._trials[self._trial_num]["rewards_Param"][action]
            done = True
            self._trial_num += 1
        self._step += 1
        obs = self.format_obs(obs)
        return obs, reward, done

    def has_more_trials(self):
        return True if self._trial_num < len(self._trials) else False
