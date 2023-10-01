import pandas as pd
import numpy as np
from itertools import chain
import math

from scipy.optimize import minimize
from utils import get_favorable_scenarios

def get_index_to_scenario():
    """Get a dictionary with a index as key and a scenario as value."""
    n_max = 7
    n = range(n_max)
    results_values = []
    for j in n: # j for the away team
        for i in n: # i for the home team
            value = f"{i} : {j}"
            results_values.append(value)

    index_to_scenario = dict(zip(list(range(n_max*n_max + 1)), results_values))
    return index_to_scenario


INDEX_TO_SCENARIO = get_index_to_scenario()


def generate_bet_return(
    df_prob, 
    df_bet,
    num_simulations,
    allocation_array
):
    num_trials = 1
    probabilities = list(chain(*df_prob.values))
    financial_return_list = []
    print(f"Running generate_bet_return")

    for _ in range(num_simulations):
        
        # Generate a single random sample from the fixed probabilities dataframe,
        # multinomial distribution as a proxy
        random_values = np.random.multinomial(num_trials, probabilities)

        #print(random_values)

        # Get the position index of the generated random value
        position = list(random_values).index(1)

        # Map the position to the actual match result
        sampled_result = INDEX_TO_SCENARIO.get(position)
        #print(f"position: {position}; sampled_result: {sampled_result}")

        #################################################################
        #sampled_result_split = sampled_result.split(" : ")
        #i = int(sampled_result_split[0])
        #j = int(sampled_result_split[1])
        #df_log.iloc[j, i] = df_log.iloc[j, i] + 1
        #################################################################
                
        df_bet['sample_flag'] = df_bet['BetMap'].apply(get_favorable_scenarios).apply(lambda x: sampled_result in x)

        # Calculate the financial return
        financial_return = sum(df_bet['Odd'] * df_bet['sample_flag'] * allocation_array)
        #print(f"financial_return: {financial_return}")
                    
        print(f"sampled_result: {sampled_result} ---------- financial_return: {financial_return}")

        financial_return_list.append(financial_return)
    
    return np.array(financial_return_list)


def compute_objective_via_simulation(
    x, # allocation array
    df_prob,
    df_bet,
    num_simulations,
):

    bet_returns = generate_bet_return(df_prob=df_prob,
                                      df_bet=df_bet,
                                      num_simulations=num_simulations,
                                      allocation_array=x)

    print(f"mean: {np.mean(bet_returns)}")
    print(f"std: {np.std(bet_returns)}")

    output = np.mean(bet_returns) / np.std(bet_returns)

    if math.isnan(output):
        output = 0

    print(f"output: {output}")

    return -output


def minimize_simulation(df_prob, df_bet, num_simulations):

    # Set restriction that sum of allocation percentages should sum up to 1
    def constraint1(x):
        return sum(x) - 1
    
    con1 = ({'type': 'eq', 'fun': constraint1})

    # Set restriction that all allocation percentages are between 0 and 1
    n_opps = len(df_bet)
    bnds = ((0, 1),) * n_opps

    # Set initial guess
    x0 = np.zeros(n_opps) + 0.5

    args = (df_prob, df_bet, num_simulations)

    res = minimize(fun=compute_objective_via_simulation,
                   x0=x0,
                   args=args,
                   constraints=con1,
                   bounds=bnds,
                   method='SLSQP', 
                   tol=0.01,
                   options={'maxiter': 100, 'disp': True, 'return_all':True})
    
    if res.success:
        return res.x
    else: 
        raise ValueError(res.message)
