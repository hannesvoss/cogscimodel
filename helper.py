import csv

import numpy as np


class Helper:
    @staticmethod
    def format_obs(step, obs):
        # prepend color to token index (g for first step, p for second step)
        if obs is None:
            return None
        step_color = ["g", "p"]
        if isinstance(obs, list):
            return list(map(lambda o: f"{step_color[step]}{o}", obs))
        else:
            return f"{step_color[step]}{obs}"

    @staticmethod
    def parse_string_to_list(string):
        return eval(string.replace('t', 'T').replace('f', 'F'))  # this is very hacky

    # is this the best way to evaluate the fit? maybe there's a better way
    # Think we need to compute the log likelihood and maximize it (see below)
    @staticmethod
    def compare_trials(trials_subject1, trials_subject2):
        assert (len(trials_subject1) == len(trials_subject2))

        num_choices = 0
        choices_matched = 0
        for trial in range(len(trials_subject1)):
            for k in trials_subject1[trial].keys():
                if trials_subject1[trial][k] == trials_subject2[trial][k]:
                    choices_matched += 1
                num_choices += 1

        return choices_matched / num_choices

    @staticmethod
    def get_user_choices_from_csv(filename):  # this function feels way more verbose than it needs to be
        trials = []
        with open(filename) as csv_file:
            reader = csv.DictReader(csv_file)
            for line in reader:

                trial_choices = {k: Helper.parse_string_to_list(line[k]) for k in ('stepOneChoice', 'stepTwoChoice')}
                for step_string, choice in trial_choices.items():
                    if 'One' in step_string:
                        trial_choices[step_string] = Helper.format_obs(0, choice)
                    elif 'Two' in step_string:
                        trial_choices[step_string] = Helper.format_obs(1, choice)
                trials.append(trial_choices)
        return trials

    @staticmethod
    def softmax(x, temperature):  # could be unstable if temperature is too low (1e-3)
        x = np.array(x)
        return np.exp(x / temperature) / np.exp(x / temperature).sum()

    @staticmethod
    def is_first_choice(choice):
        # checks if the choice is a valid first choice
        return choice in ['g0', 'g1']

    @staticmethod
    def is_second_choice(choice):
        # checks if the choice is a valid second choice
        return choice in ['p0', 'p1', 'p2', 'p3']

    @staticmethod
    def is_pink_choice(choice):
        # checks if the choice is a valid pink choice
        return choice in ['p0', 'p1']

    @staticmethod
    def is_blue_choice(choice):
        # checks if the choice is a valid blue choice
        return choice in ['p2', 'p3']
