import math
from itertools import chain

import numpy as np

from dependencies.utils import get_bet_return, get_bet_return_vectorized_optimized, softmax


def get_index_to_scenario():
    """Get a dictionary with a index as key and a scenario as value."""
    n_max = 7
    n = range(n_max)
    results_values = []
    for j in n:  # j for the away team
        for i in n:  # i for the home team
            value = f"{i} : {j}"
            results_values.append(value)

    index_to_scenario = dict(zip(list(range(n_max * n_max + 1)), results_values))
    return index_to_scenario


INDEX_TO_SCENARIO = get_index_to_scenario()

# @profile
# def generate_bet_return(df_prob, df_bet, num_simulations, allocation_array):
#     num_trials = 1

#     financial_return_list = []

#     df_bet["solution"] = allocation_array

#     for _ in range(num_simulations):

#         financial_return = 0
        
#         for game_id, game_data in df_bet.groupby("GameId", sort=False):
            
#             probabilities = list(chain(*df_prob[game_id].values))

#             probabilities /= sum(probabilities)

#             # Generate a single random sample from fixed probabilities dataframe,
#             # multinomial distribution as a proxy
#             random_values = np.random.multinomial(num_trials, probabilities)

#             # Get the position index of the generated random value
#             position = list(random_values).index(1)

#             # Map the position to the actual match result
#             scenario = INDEX_TO_SCENARIO.get(position)

#             #################################################################
#             # sampled_result_split = sampled_result.split(" : ")
#             # i = int(sampled_result_split[0])
#             # j = int(sampled_result_split[1])
#             # df_log.iloc[j, i] = df_log.iloc[j, i] + 1
#             #################################################################

#             # Calculate the financial return
#             financial_return += get_bet_return_vectorized_optimized(
#                 df=game_data, allocation_array=game_data.solution, scenario=scenario
#             )

#             #print(f"sampled_result: {scenario} ---- financial_return: {financial_return}")

#         financial_return_list.append(financial_return)

#     return np.array(financial_return_list)


def generate_bet_return(df_prob, df_bet, num_simulations, allocation_array):
    num_trials = 1
    financial_return_list = []

    df_bet["solution"] = allocation_array

    # Precompute probability sums and flatten the probabilities arrays
    precomputed_probs = {
        game_id: np.array(list(chain(*df_prob[game_id].values))) / sum(chain(*df_prob[game_id].values))
        for game_id in df_bet['GameId'].unique()
    }

    for _ in range(num_simulations):
        financial_return = 0
        
        for game_id, game_data in df_bet.groupby("GameId", sort=False):
            probabilities = precomputed_probs[game_id]

            # Generate a single random sample using multinomial distribution
            random_values = np.random.multinomial(num_trials, probabilities)

            # Get the position index of the generated random value
            position = np.argmax(random_values)  # Faster than list.index(1)

            # Map the position to the actual match result
            scenario = INDEX_TO_SCENARIO.get(position)

            # Calculate the financial return
            financial_return += get_bet_return_vectorized_optimized(
                df=game_data, allocation_array=game_data.solution, scenario=scenario
            )

        financial_return_list.append(financial_return)

    return np.array(financial_return_list)


def compute_objective_via_simulation(
    x,  # allocation array
    df_prob,
    df_bet,
    num_simulations,
):
    x = softmax(x)

    bet_returns = generate_bet_return(
        df_prob=df_prob,
        df_bet=df_bet,
        num_simulations=num_simulations,
        allocation_array=x,
    )

    #print(f"mean: {np.mean(bet_returns)}")
    #print(f"std: {np.std(bet_returns)}")

    output = np.mean(bet_returns) / np.std(bet_returns)

    if math.isnan(output):
        output = 0

    # print(f"output: {output}")

    return -output
