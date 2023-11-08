import numpy as np
import pandas as pd
import math

from analytical_return import (
    softmax,
    compute_objective_via_analytical,
)
from data import (
    load_metadata_artefacts,
    load_odds,
    join_metadata,
    apply_final_treatment,
)
from GameProbs import GameProbs
from Optimizer import Optimizer
from joblib import Parallel, delayed
from typing import Tuple, List, Any
from time import time
from filter import filter_by_linear_combination
from dependencies.utils import get_bet_return
from dependencies.config import load_config

config = load_config("config/config.yml")
metadata, gameid_to_outcome = load_metadata_artefacts(config.metadata_path)
odds = load_odds(config.odds_path)
odds = join_metadata(odds, metadata)

odds = odds[odds.Datetime.apply(str)<="2019-04-28"]

def process_group(group: Tuple[str, pd.DataFrame]) -> List[List[Any]]:
    
    is_valid_solution = True

    group_name, group_data = group
   
    games_ids = group_data['GameId'].unique()
    
    # Initialize dict to store dataframes of favorable bet opportunities
    odds_dict = {}
    # Initialize dict to store 7x7 matrices/dataframes of real probabilities 
    df_probs_dict = {}

    if len(games_ids) > 1:

        for game_id in games_ids:
            df = GameProbs(game_id).build_dataframe()
            
            odds_sample = group_data[(group_data.GameId==game_id)]
            odds_sample = apply_final_treatment(df_odds=odds_sample, df_real_prob=df)
            odds_sample = filter_by_linear_combination(odds_sample)

            odds_dict[game_id] = odds_sample
            df_probs_dict[game_id] = df

        odds_dt = pd.concat(odds_dict.values())
        print(f"odds_dt.shape: {odds_dt.shape}")

        if len(odds_dt) <= config.max_vector_length:

            print(f"Date: {group_name}")

            odds_favorable = np.array(odds_dt['Odd'])
            real_prob_favorable = np.array(odds_dt['real_prob'])
            event_favorable = list(odds_dt['BetMap'].values)
            games_ids = np.array(odds_dt['GameId'])
                
            #try:
            print("Execution of minimization task...")    
            solution, time_limit_flag = Optimizer().run_optimization(
                fun=compute_objective_via_analytical,
                x0=np.zeros(len(odds_favorable)),
                args=(odds_favorable, real_prob_favorable, event_favorable, games_ids, df_probs_dict)
            )               
            print("Finalization of minimization task...")

            #except ValueError:
                #continue
            
            if any(math.isnan(x) for x in solution):
                is_valid_solution = False

            odds_dt['solution'] = softmax(solution)

            track_record = []

            for game_id, game_data in odds_dt.groupby('GameId', sort=False):
                scenario = gameid_to_outcome[game_id]
                financial_return = get_bet_return(df=game_data,
                                                  allocation_array=game_data.solution,
                                                  scenario=scenario)

                print(f"game_id: {game_id}; financial_return: {financial_return}")

                track_record.append([str(game_id),
                                     financial_return,
                                     len(game_data),
                                     group_name,
                                     time_limit_flag,
                                     is_valid_solution])

            return track_record


def run_strategy():
    
    start_time = time()
    
    grouped = odds.groupby('Datetime')
    
    # Use all available CPU cores for parallel execution
    num_jobs = -1  
    # Parallelize the group processing
    results = Parallel(n_jobs=num_jobs)(delayed(process_group)(group) for group in grouped)

    data = [x for x in results if x is not None]
    flat_list = [item for sublist in data for item in sublist]
    pd.DataFrame(flat_list).to_csv(config.strategy_result_path, index=False)

    elapsed_time = time() - start_time
    print("Final Elapsed: %.3f sec" % elapsed_time)


if __name__ == "__main__":
    run_strategy()
