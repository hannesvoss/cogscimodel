import numpy as np
import pandas as pd

from helper import Helper
from tdlr import TDLR
from tdlr_modelbased import TDLRModelBased
from two_step import TwoStep

if __name__ == '__main__':
    # -------------- Load Data ----------------
    csv_filenames = [
        "./data/experiment_data_Fatemeh.csv",
        "./data/experiment_data_Lane.csv",
        "./data/experiment_data_hannes.csv",
        "./data/experiment_data_sonja.csv",
        "./data/experiment_data_timothy_ho.csv",
        "./data/experiment_katarzyna_data.csv",
    ]
    for csv_filename in csv_filenames:
        df = pd.read_csv(csv_filename)
        print(csv_filename, (df["reward"] == True).sum())

    # -------------- TDLR Learning Simulation ----------------
    sonja = TwoStep(filename="./data/experiment_data_sonja.csv")
    tdlr = TDLR(environment=sonja, alpha=0.8, gamma=0.9, temperature=1)
    tdlr_trials = []  # Initial values for each choice at both stages

    first_stage_choice, second_stage_choice = None, None
    while tdlr.environment.has_more_trials():
        obs, reward, done = tdlr.environment.reset()
        tdlr_trials.append({})
        while not done:
            action = tdlr.choose_action(obs)
            if action[0] == 'g':
                first_stage_choice = action
                tdlr_trials[-1]["stepOneChoice"] = action
            elif action[0] == 'p':
                second_stage_choice = action
                tdlr_trials[-1]["stepTwoChoice"] = action
            obs, reward, done = tdlr.environment.step(action)
        tdlr.update(first_stage_choice, second_stage_choice, reward)

    # -------------- TDLR Model Based Learning Simulation ----------------
    sonja = TwoStep(filename="./data/experiment_data_sonja.csv")
    tdlr_modelbased = TDLRModelBased(environment=sonja, alpha=0.8, gamma=0.9, temperature=1)
    tdlr_trials = []  # Initial values for each choice at both stages

    episodes = 10

    first_stage_choice, second_stage_choice = None, None
    for _ in range(episodes):
        tdlr_modelbased.environment.reset_trials()
        transition_count = {'g0-g': 0, 'g1-g': 0, 'g0-p': 0, 'g1-p': 0}
        first_stage_count = {'g0': 0, 'g1': 0}
        while tdlr_modelbased.environment.has_more_trials():
            obs, reward, done = tdlr_modelbased.environment.reset()
            tdlr_trials.append({})
            while not done:
                action = tdlr_modelbased.choose_action(obs)
                if action[0] == 'g':
                    first_stage_choice = action
                    tdlr_trials[-1]["stepOneChoice"] = action
                elif action[0] == 'p':
                    second_stage_choice = action
                    tdlr_trials[-1]["stepTwoChoice"] = action
                obs, reward, done = tdlr_modelbased.environment.step(action)
                first_stage_count[first_stage_choice] += 1
            transition_count[f"{first_stage_choice}-g"] += 1 if second_stage_choice[:-1] == 'g' else 0
            transition_count[f"{first_stage_choice}-p"] += 1 if second_stage_choice[:-1] == 'p' else 0
            tdlr_modelbased.update(first_stage_choice, second_stage_choice, reward)
        tdlr_modelbased.update_transition_probabilities(transition_count, first_stage_count)

    # -------------- Parameter Fitting ----------------

    def compute_likelihood(model_probs, observed_choices):
        log_likelihood = 0.0
        for first_stage_choice, second_stage_choice in observed_choices:
            # Get the model probabilities for the observed choices
            # here I am struggling a bit with how exactly the probabilities have to be computed as we need the prob
            # for the first step and the for the second (but given the first??)
            prob_first_stage = model_probs[first_stage_choice]
            prob_second_stage = prob_first_stage[second_stage_choice]

            # Add the log probability to the overall log-likelihood
            log_likelihood += np.log(prob_second_stage)
        return log_likelihood


    def run_parameter_fitting_tdlr():
        # compare log_likelihood for different parameter values
        alphas = np.arange(0, 1, 0.05)
        gammas = np.arange(0, 1, 0.05)
        temperatures = [0.0025, 0.05, 0.1, 1, 10]

        episodes = 10

        models = [
            TDLR,
            TDLRModelBased,
        ]
        for modeltype in models:
            print(f"--" * 20)
            print(f"Running parameter fitting for {modeltype.__name__}")
            for csv_filename in csv_filenames:
                user_trials = Helper.get_user_choices_from_csv(csv_filename)
                print(f"--" * 20)
                print(f"Fitting parameters for {csv_filename}")

                max = 0
                config = None
                scores = []
                for alpha in alphas:
                    for gamma in gammas:
                        for temperature in temperatures:
                            model = modeltype(
                                environment=TwoStep(csv_filename),
                                alpha=alpha,
                                gamma=gamma,
                                temperature=temperature
                            )
                            trials = []

                            first_stage_choice, second_stage_choice = None, None
                            if modeltype == TDLR:
                                while model.environment.has_more_trials():
                                    obs, reward, done = model.environment.reset()
                                    trials.append({})
                                    while not done:
                                        action = model.choose_action(obs)
                                        if action[0] == 'g':
                                            first_stage_choice = action
                                            trials[-1]["stepOneChoice"] = action
                                        elif action[0] == 'p':
                                            second_stage_choice = action
                                            trials[-1]["stepTwoChoice"] = action
                                        obs, reward, done = model.environment.step(action)
                                    model.update(first_stage_choice, second_stage_choice, reward)
                            elif modeltype == TDLRModelBased:
                                for _ in range(episodes):
                                    model.environment.reset_trials()
                                    transition_count = {'g0-g': 0, 'g1-g': 0, 'g0-p': 0, 'g1-p': 0}
                                    first_stage_count = {'g0': 0, 'g1': 0}
                                    while model.environment.has_more_trials():
                                        obs, reward, done = model.environment.reset()
                                        trials.append({})
                                        while not done:
                                            action = model.choose_action(obs)
                                            if action[0] == 'g':
                                                first_stage_choice = action
                                                trials[-1]["stepOneChoice"] = action
                                            elif action[0] == 'p':
                                                second_stage_choice = action
                                                trials[-1]["stepTwoChoice"] = action
                                            obs, reward, done = model.environment.step(action)
                                            first_stage_count[first_stage_choice] += 1

                                        transition_count[f"{first_stage_choice}-g"] += 1 if second_stage_choice[:-1] == 'g' else 0
                                        transition_count[f"{first_stage_choice}-p"] += 1 if second_stage_choice[:-1] == 'p' else 0
                                        model.update(first_stage_choice, second_stage_choice, reward)
                                    model.update_transition_probabilities(transition_count, first_stage_count)

                            score = Helper.compare_trials(user_trials, trials) if modeltype == TDLR else Helper.compare_trials(user_trials * episodes, trials)
                            if score > max:
                                max = score
                                config = f"Alpha: {alpha}, Gamma: {gamma}, Temperature: {temperature}"
                                print(max, config)
                            # scores.append(score)
                            # print(compare_trials(tim_trials, trials)


    run_parameter_fitting_tdlr()
