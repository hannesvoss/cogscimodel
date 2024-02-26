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
        "./data/experiment_data_Hannes.csv",
        "./data/experiment_data_Sonja.csv",
        "./data/experiment_data_Timothy.csv",
        "./data/experiment_data_Katarzyna.csv",
    ]
    for csv_filename in csv_filenames:
        df = pd.read_csv(csv_filename)
        print(csv_filename, (df["reward"] == True).sum())

    # -------------- TDLR Learning Simulation ----------------
    # defining parameters
    # alpha = 0.8
    # gamma = 0.9
    # temperature = 1
    # csv_filename = "./MCN Group Project/experiment_data_timothy_ho.csv"
    def simulate_tdlr(filename, alpha, gamma, temperature):
        """
        Runs temporal difference learning on observations with a given set of parameters.

        Args:
            filename (String): CSV file containing observations.
            alpha (float): The learning rate.
            gamma (float): The discount factor.
            temperature (float): The softmax temperature.

        Returns:
            list[dict]: The choices the model made at each trial of the two-step task.
            list[dict]: The state values at each trial of the two-step task.
        """

        # Create a TDLR model with the environment and the learning parameters
        tdlr = TDLR(
            environment=TwoStep(filename),
            alpha=alpha,
            gamma=gamma,
            temperature=temperature,
        )
        trials = []
        state_values = []

        # Initial values for each choice at both stages
        first_choice, second_choice = None, None

        # iterate over all 200 trials and update the model after each trial
        while tdlr.environment.has_more_trials():
            obs, reward, done = tdlr.environment.reset()
            trials.append({})
            while not done:
                state_values.append(tdlr._state_values.copy())
                action = tdlr.choose_action(obs)
                if Helper.is_first_choice(action):
                    first_choice = action
                    trials[-1]['stepOneChoice'] = first_choice
                else:
                    second_choice = action
                    trials[-1]['stepTwoChoice'] = second_choice
                obs, reward, done = tdlr.environment.step(action)
            tdlr.update(first_choice, second_choice, reward)
        return trials, state_values

    # -------------- TDLR Model Based Learning Simulation ----------------
    def simulate_tdlr_modelbased(filename, alpha, gamma, temperature):
        # Create a TDLR model with the environment and the learning parameters
        tdlr_modelbased = TDLRModelBased(
            environment=TwoStep(filename),
            alpha=alpha,
            gamma=gamma,
            temperature=temperature,
        )

        trials = []  # maybe use this to calculate the transition probabilities?
        state_values = []

        first_choice, second_choice = None, None
        transition_count = {'g0-p': 0, 'g1-p': 0, 'g0-b': 0, 'g1-b': 0}
        first_stage_count = {'g0': 0, 'g1': 0}

        while tdlr_modelbased.environment.has_more_trials():
            obs, reward, done = tdlr_modelbased.environment.reset()
            trials.append({})
            while not done:
                state_values.append(tdlr_modelbased._state_values.copy())
                action = tdlr_modelbased.choose_action(obs)
                if Helper.is_first_choice(action):
                    first_choice = action
                    trials[-1]['stepOneChoice'] = first_choice
                    first_stage_count[first_choice] += 1
                elif Helper.is_second_choice(action):
                    second_choice = action
                    trials[-1]['stepTwoChoice'] = second_choice
                obs, reward, done = tdlr_modelbased.environment.step(action)

            transition_count[f"{first_choice}-p"] += 1 if Helper.is_pink_choice(second_choice) else 0
            transition_count[f"{first_choice}-b"] += 1 if Helper.is_blue_choice(second_choice) else 0

            tdlr_modelbased.update_transition_probabilities(transition_count, first_stage_count)

            tdlr_modelbased.update(first_choice, second_choice, reward)
        return trials, transition_count, state_values

    # trials, state_values = simulate_tdlr(csv_filenames[2], 0.8, 0.9, 1)
    trials, tr_count, state_vals = simulate_tdlr_modelbased(csv_filenames[2], 0.8, 0.9, 1)
    print(tr_count)

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

    def parameter_search_tdlr_modelbased(filename, params):
        results = []
        for alpha in params['alpha']:
            for gamma in params['gamma']:
                for temperature in params['temperature']:
                    trials, n_transitions, state_vals = simulate_tdlr_modelbased(filename, alpha, gamma, temperature)
                    results.append({'alpha': alpha, 'gamma': gamma, 'temperature': temperature, 'trials': trials})
        return results


    for csv_filename in csv_filenames:
        user_trials = Helper.get_user_choices_from_csv(csv_filename)
        print(f"Fitting parameters for {csv_filename}")

        max = 0
        config = None
        scores = []

        alphas = np.arange(0, 1, 0.05)
        gammas = np.arange(0, 1, 0.05)
        temperatures = [0.0025, 0.05, 0.1, 1, 10]

        params = {'alpha': alphas, 'gamma': gammas, 'temperature': temperatures}

        parameter_search_results = parameter_search_tdlr_modelbased(csv_filename, params)
        # calculate mean of "probs" key in parameter_search_results
        # parameter_search_results.map(lambda x: x["probs"]).mean()

        for result in parameter_search_results:
            score = Helper.compare_trials(user_trials, result['trials'])
            if score > max:
                max = score
                config = f'Alpha: {result["alpha"]}, Gamma: {result["gamma"]}, Temperature: {result["temperature"]}'
                print(max, config)


    def run_parameter_fitting_tdlr():
        # compare log_likelihood for different parameter values
        alphas = np.arange(0, 1, 0.05)
        gammas = np.arange(0, 1, 0.05)
        temperatures = [0.0025, 0.05, 0.1, 1, 10]

        models = [
            # TDLR,
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

                            first_choice, second_choice = None, None
                            if modeltype == TDLR:
                                while model.environment.has_more_trials():
                                    obs, reward, done = model.environment.reset()
                                    trials.append({})
                                    while not done:
                                        action = model.choose_action(obs)
                                        if action[0] == 'g':
                                            first_choice = action
                                            trials[-1]["stepOneChoice"] = first_choice
                                        elif action[0] == 'p':
                                            second_choice = action
                                            trials[-1]["stepTwoChoice"] = second_choice
                                        obs, reward, done = model.environment.step(action)
                                    model.update(first_choice, second_choice, reward)
                            elif modeltype == TDLRModelBased:
                                model.environment.reset_trials()
                                transition_count = {'g0-p': 0, 'g1-p': 0, 'g0-b': 0, 'g1-b': 0}
                                first_stage_count = {'g0': 0, 'g1': 0}
                                while model.environment.has_more_trials():
                                    obs, reward, done = model.environment.reset()
                                    trials.append({})
                                    while not done:
                                        action = model.choose_action(obs)
                                        if Helper.is_first_choice(action):
                                            first_choice = action
                                            trials[-1]["stepOneChoice"] = action
                                            first_stage_count[first_choice] += 1
                                        elif Helper.is_second_choice(action):
                                            second_choice = action
                                            trials[-1]["stepTwoChoice"] = action
                                        obs, reward, done = model.environment.step(action)

                                    transition_count[f"{first_choice}-p"] += 1 if Helper.is_pink_choice(second_choice) else 0
                                    transition_count[f"{first_choice}-b"] += 1 if Helper.is_blue_choice(second_choice) else 0
                                    model.update(first_choice, second_choice, reward)
                                model.update_transition_probabilities(transition_count, first_stage_count)

                            score = Helper.compare_trials(user_trials, trials) if modeltype == TDLR else Helper.compare_trials(user_trials, trials)
                            if score > max:
                                max = score
                                config = f"Alpha: {alpha}, Gamma: {gamma}, Temperature: {temperature}"
                                print(max, config)
                            # scores.append(score)
                            # print(compare_trials(tim_trials, trials)

    run_parameter_fitting_tdlr()
