import argparse
import datetime
import math
import os
from time import time
from typing import Any, List, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sparsemax import Sparsemax

from artifacts import (
    build_plot_df_wrapper_baseline_kreiner,
    save_csv_artifact,
    save_plot_strategy,
)
#from BoTorchOptimizer import BoTorchOptimizer
from data import (
    apply_final_treatment,
    join_metadata,
    load_metadata_artefacts,
    load_odds,
)
from dependencies.config import load_config
from dependencies.utils import get_bet_return, save_df_as_parquet, softmax
from filter import filter_by_linear_combination
from GameProbs2 import GameProbs, PreprocessKreiner
from loguru import logger

config = load_config("config/config.yml")


def setup(args):

    metadata, gameid_to_outcome = load_metadata_artefacts(config.metadata_path)
    odds = load_odds(config.odds_path, args.bookmakers)
    odds = join_metadata(odds, metadata)
    
    replacements = {
        'h': 'home',
        'a': 'away',
        'r': 'draw',
    }

    odds["Bet"] = odds["Bet"].replace(replacements)
    
    odds = odds[(odds.Market=="h2h")]

    odds = odds.sort_values(["Datetime", "GameId"], ascending=True)

    #odds = odds[(odds.Datetime.apply(str)>"2019-08-01")&(odds.Datetime.apply(str)<"2019-09-01")]
    #odds = odds[(odds.Datetime.apply(str)>"2022-01-01")&(odds.Datetime.apply(str)<"2023-01-01")]
    odds = odds[(odds.Datetime.apply(str)>="2023-01-01")&(odds.Datetime.apply(str)<"2024-01-01")]
    
    # odds = odds[
    #     (odds.Datetime.apply(str) > "2019-06-01")
    #     & (odds.Datetime.apply(str) < "2019-07-01")
    # ]

    # odds = odds[odds]

    return odds, gameid_to_outcome


def process_group(
    group: Tuple[str, pd.DataFrame], gameid_to_outcome, args
) -> List[List[Any]]:
    is_valid_solution = True

    date, group_data = group

    games_ids = group_data["GameId"].unique()

    # Initialize dict to store dataframes of favorable bet opportunities
    odds_dict = {}
    # Initialize dict to store 7x7 matrices/dataframes of real probabilities
    df_probs_dict = {}

    if len(games_ids) > args.min_games:
        for game_id in games_ids:
            logger.info(f"GameId: {game_id}")
            preprocess_method = PreprocessKreiner(game_id)
            df = GameProbs(preprocess_method).build_dataframe()
            odds_sample = group_data[(group_data.GameId == game_id)]
            odds_sample = apply_final_treatment(df_odds=odds_sample, df_real_prob=df)
            if not args.do_baseline:
                odds_sample = filter_by_linear_combination(odds_sample, n=args.bets_per_game, weight=args.weight)
            #else:
            #    odds_sample = odds_sample.sample(1)
            odds_dict[game_id] = odds_sample
            df_probs_dict[game_id] = df

        odds_dt = pd.concat(odds_dict.values())


        # Here, we are getting the top bets as proxy for the implementation of the Kreiner paper
        # odds_dt = odds_dt.sample(30)
        
        if len(odds_dt) <= config.max_vector_length and len(odds_dt) > 1:
        #if len(odds_dt) <= config.max_vector_length :
            iteration_date = odds_dt.Datetime.apply(str).unique()[0]
            logger.info(f"Date: {iteration_date}")

            odds_favorable = torch.tensor(np.array(odds_dt["Odd"]))
            real_prob_favorable = torch.tensor(np.array(odds_dt["real_prob"]))
            event_favorable = list(odds_dt["BetMap"].values)
            games_ids = np.array(odds_dt["GameId"])
            time_limit_flag = None

            if not args.do_baseline:
                # try:
                logger.info("Execution of minimization task...")


                # optimizer_instance = BoTorchOptimizer(
                #     n_iterations=args.n_iterations,
                #     public_odd=odds_favorable,
                #     real_probabilities=real_prob_favorable,
                #     event=event_favorable,
                #     games_ids=games_ids,
                #     df_probs_dict=df_probs_dict,
                # )

                allocation = 1
                solution = [allocation for _ in range(len(odds_favorable))]
                odds_dt["solution"] = solution

                logger.info("Finalization of minimization task...")

                if any(math.isnan(x) for x in solution):
                    is_valid_solution = False

            else:
                odds_dt["solution"] = 1

            save_df_as_parquet(odds_dt, str(date))

            track_record = []
            
            financial_return_aggregated = 0

            for game_id, game_data in odds_dt.groupby("GameId", sort=False):
                scenario = gameid_to_outcome[game_id]
                financial_return = get_bet_return(
                    df=game_data, allocation_array=game_data.solution, scenario=scenario
                )
                financial_return_aggregated += financial_return
                logger.info(
                    f"game_id: {game_id}; financial_return: {np.round(financial_return, 3)}"
                )

                track_record.append(
                    [
                        str(game_id),
                        financial_return,
                        len(game_data),
                        odds_dt.n_favorable_bets.values[0],
                        time_limit_flag,
                        is_valid_solution,
                        iteration_date,
                    ]
                )
            

            if financial_return_aggregated < 1:
                logger.warning(f"Negative return for the day {iteration_date}")
            
            else:
                logger.info(f"Positive return for the day {iteration_date}")

            return track_record


def run_strategy(args):
    start_time = time()

    logger.info("Starting the strategy...")
    logger.info(f"Aggregator: {args.aggregator}")
    logger.info(f"Minimum number of games: {args.min_games}")
    logger.info(f"Bookmakers: {args.bookmakers}")
    logger.info(f"Number of bets per game: {args.bets_per_game}")
    logger.info(f"Weight: {args.weight}")
    logger.info(f"Do baseline: {args.do_baseline}")
    logger.info(f"Number of iterations: {args.n_iterations}")
    logger.info(f"Probability mapping: {args.probability_mapping}")
    logger.info(f"Number of jobs: {args.n_jobs}")
    logger.info(f"Save experiment: {args.save_experiment}")

    odds, gameid_to_outcome = setup(args)

    grouped = odds.groupby(args.aggregator)
    # Parallelize the group processing
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_group)(group, gameid_to_outcome, args) for group in grouped
    )

    data = [x for x in results if x is not None]
    df_flat = pd.DataFrame([item for sublist in data for item in sublist])

    # Start an MLflow experiment
    with mlflow.start_run():
        # Log parameters (e.g., settings of the optimizer)
        mlflow.log_param("aggregator", args.aggregator)
        mlflow.log_param("min_games", args.min_games)
        mlflow.log_param("bookmakers", args.bookmakers)
        mlflow.log_param("bets_per_game", args.bets_per_game)
        mlflow.log_param("weight", args.weight)
        mlflow.log_param("probability_mapping", args.probability_mapping)
        mlflow.log_param("do_baseline", args.do_baseline)
        mlflow.log_param("n_iterations", args.n_iterations)
        mlflow.log_param("start_date", odds.Datetime.min())
        mlflow.log_param("end_date", odds.Datetime.max())


        if args.save_experiment:
            # Create artefacts folder
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            artefacts_folder = f"artefacts/{timestamp}"
            os.makedirs(artefacts_folder)
            save_csv_artifact(artefacts_folder, "result", df_flat)
            df_plot = build_plot_df_wrapper_baseline_kreiner(artefacts_folder,  args.aggregator, args.do_baseline)
            save_csv_artifact(artefacts_folder, "result_plot", df_plot)
            save_plot_strategy(artefacts_folder, df_plot)
        
        mlflow.log_param("timestamp", timestamp)
        mlflow.log_artifact(f"{artefacts_folder}/result_plot.csv")
        mlflow.log_artifact(f"{artefacts_folder}/plot.PNG")
        mlflow.log_metric("wealth", np.round(df_plot["stake"].values[-1], 3))

        # End the MLflow run
        mlflow.end_run()

    elapsed_time = time() - start_time
    logger.info("Final Elapsed: %.3f sec" % elapsed_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bookmakers',
        nargs='+',
        default=None,
        help='A list of strings',
    )
    parser.add_argument(
        "--aggregator", type=str, help="aggregate by GameId or by Datetime"
    )
    parser.add_argument(
        "--min_games",
        type=int,
        default=1,
        help="threshold of minimum number of games to enter the optimization task",
    )
    parser.add_argument(
        "--bets_per_game", type=int, default=5, help="number of bets per game"
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=0.5,
        help="weight of the linear combination filter",
    )
    parser.add_argument(
        "--probability_mapping",
        type=str,
        default="softmax",
        help="probability mapping function to use",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=100,
        help="number of iterations to run the optimization task",
    )
    parser.add_argument(
        "--do_baseline",
        action="store_true",
        help="flag to apply baseline logic or not, not specifying the argument return the opposite of the action",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="number of jobs to run in parallel",
    )
    parser.add_argument(
        "--save_experiment",
        action="store_true",
        help="flag to save the experiment artefacts, not specifying the argument return the opposite of the action",
    )
    args = parser.parse_args()
    print(args)
    run_strategy(args)
