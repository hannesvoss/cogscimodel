import csv


class Helper:
    @staticmethod
    def format_obs_static(step, obs):
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
                        trial_choices[step_string] = Helper.format_obs_static(0, choice)
                    elif 'Two' in step_string:
                        trial_choices[step_string] = Helper.format_obs_static(1, choice)
                trials.append(trial_choices)
        return trials
