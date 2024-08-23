import numpy as np
import pandas as pd
import logging
import glob
import os

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


INDEX_TO_SCENARIO_BET_MAP = get_index_to_scenario_for_betmap()


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
        
    positions = find_positions(x, target_element=1)
    scenarios = get_values_by_keys(INDEX_TO_SCENARIO_BET_MAP, positions)

    return scenarios


def get_bet_return(df: pd.DataFrame, allocation_array: list, scenario: str) -> float:
    """Get financial return of the bet by given allocation and scenario"""
    check_scenario = lambda x: scenario in x
    # Check if scenario is inside the BetMap
    df['flag'] = df['BetMap'].apply(get_scenarios).apply(check_scenario)
    
    # logger.info(f"Bets won:\n{df[df.flag][['Market', 'Bet', 'Scenario', 'Odd', 'flag']]}")
    
    # logger.info(f"Allocation won:\n{pd.Series(allocation_array)[df.flag.to_list()]}")
    
    # Calculate the financial return
    return sum(df['Odd'] * df['flag'] * allocation_array)


def get_scenarios_vectorized_optimized(betmap_matrix: np.ndarray) -> np.ndarray:
    """
    Optimized version to retrieve the scenarios from the BetMap matrix.
    
    betmap_matrix: NumPy 2D array where each row corresponds to a game, 
                   and each column corresponds to a specific score scenario.
    
    Returns a 2D boolean array where each row indicates which scenarios are active.
    """
    # Generate the scenario strings (for a 7x7 grid)
    scenario_strings = np.array([f"{i} : {j}" for i in range(7) for j in range(7)])
    
    # Directly create a 2D boolean array where each element is True if the scenario is active
    active_scenarios = betmap_matrix == 1
    
    return active_scenarios, scenario_strings


def get_bet_return_vectorized_optimized(df: pd.DataFrame, allocation_array: list, scenario: str) -> float:
    """
    Get financial return of the bet by given allocation and scenario in a fully optimized vectorized manner.
    
    df: Pandas DataFrame that contains BetMap, Odd, etc.
    allocation_array: List of allocations corresponding to each bet.
    scenario: The specific scenario we are interested in checking.
    """
    # Convert BetMap to a NumPy array
    betmap_matrix = np.vstack(df['BetMap'].values)

    # Get a 2D boolean array for active scenarios and the scenario strings
    active_scenarios, scenario_strings = get_scenarios_vectorized_optimized(betmap_matrix)
    
    # Find the index of the target scenario in the scenario_strings array
    scenario_index = np.where(scenario_strings == scenario)[0][0]
    
    # Vectorized check: Get the flag for each row where the scenario is active
    scenario_flags = active_scenarios[:, scenario_index]
    
    # Convert the allocation array to a NumPy array
    allocation_array = np.array(allocation_array)
    
    # Calculate the financial return using vectorized operations
    # financial_return = np.sum(df['Odd'].values * scenario_flags * allocation_array)
    financial_return = np.dot(df['Odd'].values * scenario_flags, allocation_array)

    
    return financial_return


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def sparsemax(x):
    """
    Compute sparsemax values for each set of scores in x.
    """
    # Sort x in descending order
    x_sorted = np.sort(x)[::-1]
    
    # Compute the cumulative sum of sorted values
    cum_sum = np.cumsum(x_sorted)
    
    # Compute the threshold for sparsemax
    k = np.arange(1, len(x) + 1)
    threshold = (cum_sum - k) / k
    
    # Find the index where x_sorted is greater than the threshold
    tau = np.maximum(x_sorted - threshold, 0)
    
    # Reconstruct the sparsemax output
    sparsemax_values = np.zeros_like(x)
    sparsemax_values[np.argsort(x)[::-1]] = tau
    
    return sparsemax_values


def save_df_as_parquet(df, filename, directory='EDA'):
    """
    Saves a DataFrame to a specified Parquet file within a given directory.

    Args:
    df (pandas.DataFrame): The DataFrame to save.
    filename (str): The base filename to use, without an extension.
    directory (str): The directory in which to save the files.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Construct the full file path
    file_path = os.path.join(directory, f"{filename}.parquet")
    
    # Save the DataFrame
    df.to_parquet(file_path, index=False)


def read_all_parquet(directory):
    # List all Parquet files in the directory
    files = glob.glob(f'{directory}/*.parquet')
    
    # Read each file into a DataFrame and append to a list
    dfs = [pd.read_parquet(file) for file in files]
    
    # Concatenate all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df
