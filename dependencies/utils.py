import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def get_index_to_scenario_for_betmap():
    """
    Get a dictionary with a index as key and a scenario as value, for 
    7x7 soccer scenarios. In the iteration, we fix first the home team
    index, so we get 0x0, 0x1, 0x2, ..., 0x6, 1x0, 1x1, 1x2, ..., 6x6.
    """
    n_max = 7
    n = range(n_max)
    results_values = []
    for i in n:  # i for the home team
        for j in n:  # j for the away team
            value = f"{i} : {j}"
            results_values.append(value)

    index_to_scenario = dict(zip(list(range(n_max*n_max + 1)), results_values))
    
    return index_to_scenario


def find_positions(input_list, target_element) -> list:
    """Get target element indexes of given input list."""
    # Using a list comprehension to find positions
    positions = [index for index, element in enumerate(input_list) if element == target_element]
    
    return positions


def get_values_by_keys(dictionary, keys_to_lookup) -> list:
    """Get list of values of dictionary by given keys"""
    return [dictionary.get(key) for key in keys_to_lookup]


def get_scenarios(x: list)-> list:
    """Get list of 7x7 soccer scenarios by given list of dummies"""
    
    INDEX_TO_SCENARIO_BET_MAP = get_index_to_scenario_for_betmap()
    
    positions = find_positions(x, target_element=1)
    scenarios = get_values_by_keys(INDEX_TO_SCENARIO_BET_MAP, positions)

    return scenarios


def get_bet_return(df: pd.DataFrame, allocation_array: list, scenario: str) -> float:
    """Get financial return of the bet by given allocation and scenario"""
    check_scenario = lambda x: scenario in x
    # Check if scenario is inside the BetMap
    df['flag'] = df['BetMap'].apply(get_scenarios).apply(check_scenario)
    
    logger.info(f"Bets won:\n{df[df.flag][['Market', 'Bet', 'Scenario', 'Odd', 'flag']]}")
    
    logger.info(f"Allocation won:\n{pd.Series(allocation_array)[df.flag.to_list()]}")
    
    # Calculate the financial return
    return sum(df['Odd'] * df['flag'] * allocation_array)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
