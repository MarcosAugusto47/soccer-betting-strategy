import os
import argparse
import numpy as np
import pandas as pd
import math
import datetime

import torch
from BoTorchOptimizer import BoTorchOptimizer

from data import (
    load_metadata_artefacts,
    load_odds,
    join_metadata,
    apply_final_treatment,
)
from artifacts import (
    save_csv_artifact,
    save_plot_strategy,
    build_plot_df_wrapper,
)
from GameProbs import GameProbs
from Optimizer import Optimizer
from joblib import Parallel, delayed
from typing import Tuple, List, Any
from time import time
from filter import filter_by_linear_combination
from dependencies.utils import get_bet_return, softmax
from dependencies.config import load_config

config = load_config("config/config.yml")
metadata, gameid_to_outcome = load_metadata_artefacts(config.metadata_path)
odds = load_odds(config.odds_path)
odds = join_metadata(odds, metadata)

odds = odds.sort_values(["Datetime", "GameId"], ascending=True)

#odds = odds[(odds.Datetime.apply(str)<"2020-01-01")]
odds = odds[(odds.Datetime.apply(str)>"2022-01-01") & (odds.Datetime.apply(str)<"2024-01-01")]

def process_group(group: Tuple[str, pd.DataFrame], args) -> List[List[Any]]:
    
    is_valid_solution = True

    _, group_data = group
   
    games_ids = group_data['GameId'].unique()
    
    # Initialize dict to store dataframes of favorable bet opportunities
    odds_dict = {}
    # Initialize dict to store 7x7 matrices/dataframes of real probabilities 
    df_probs_dict = {}

    if len(games_ids) > args.min_games:

        for game_id in games_ids:
            df = GameProbs(game_id).build_dataframe()
            
            odds_sample = group_data[(group_data.GameId==game_id)]
            odds_sample = apply_final_treatment(df_odds=odds_sample, df_real_prob=df)
            if not args.do_baseline:   
                odds_sample = filter_by_linear_combination(odds_sample)
            else:
                odds_sample = odds_sample.sample(1)
            odds_dict[game_id] = odds_sample
            df_probs_dict[game_id] = df

        odds_dt = pd.concat(odds_dict.values())

        if len(odds_dt) <= config.max_vector_length:
            iteration_date = odds_dt.Datetime.apply(str).unique()[0]
            print(f"Date: {iteration_date}")

            odds_favorable = torch.tensor(np.array(odds_dt['Odd']))
            real_prob_favorable = torch.tensor(np.array(odds_dt['real_prob']))
            event_favorable = list(odds_dt['BetMap'].values)
            games_ids = np.array(odds_dt['GameId'])

            #try:
            print("Execution of minimization task...")

            time_limit_flag = None

            optimizer_instance = BoTorchOptimizer(
                public_odd=odds_favorable,
                real_probabilities=real_prob_favorable,
                event=event_favorable,
                games_ids=games_ids,
                df_probs_dict=df_probs_dict
            )
            
            solution = optimizer_instance.run_optimization()

            print("Finalization of minimization task...")

            #except ValueError:
                #continue
            
            if any(math.isnan(x) for x in solution):
                is_valid_solution = False
            odds_dt['solution'] = softmax(solution)
            print(softmax(solution))
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
                                     time_limit_flag,
                                     is_valid_solution,
                                     iteration_date])

            return track_record


def run_strategy(args):
    
    start_time = time()

    grouped = odds.groupby(args.aggregator)
    
    # Use all available CPU cores for parallel execution
    num_jobs = 1 
    # Parallelize the group processing
    results = Parallel(n_jobs=num_jobs)(delayed(process_group)(group, args) for group in grouped)

    data = [x for x in results if x is not None]
    df_flat = pd.DataFrame([item for sublist in data for item in sublist])

    # Create artefacts folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    artefacts_folder = f"artefacts/botorch_aggregator{args.aggregator}_min_games{args.min_games}_do_baseline{args.do_baseline}_{timestamp}"
    os.makedirs(artefacts_folder)
    args.artefacts_folder = artefacts_folder

    save_csv_artifact(artefacts_folder, "result", df_flat)
    df_plot = build_plot_df_wrapper(args)
    save_csv_artifact(artefacts_folder, "result_plot", df_plot)
    save_plot_strategy(args, df_plot)
    
    elapsed_time = time() - start_time
    print("Final Elapsed: %.3f sec" % elapsed_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregator",
        type=str,
        help="aggregate by GameId or by Datetime"
    )
    parser.add_argument(
        "--min_games",
        type=int,
        default=0,
        help="threshold of minimum number of games to enter the optimization task"
    )
    parser.add_argument(
        "--do_baseline",
        action='store_true',
        help="flag to apply baseline logic or not, not specifying the argument return the opposite of the action"
    )
    args = parser.parse_args()
    print(args)
    run_strategy(args)
