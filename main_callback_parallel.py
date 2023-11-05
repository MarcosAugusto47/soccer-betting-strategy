import numpy as np
import pandas as pd
import logging

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
from utils import get_bet_return
from config import games_ids as GAMES_IDS
from joblib import Parallel, delayed
from typing import Tuple
from time import time

start_time = time()

#logging.basicConfig(level=logging.INFO, filename="my_log.log", filemode="w")
#logger = logging.getLogger(__name__)

#logger.info("RUN")

metadata, gameid_to_outcome = load_metadata_artefacts("data/metadata-with-date.parquet")
odds = load_odds("data/odds.parquet")
odds = join_metadata(odds, metadata)

#track_record_list = []

count = 0

odds = odds[odds.Datetime.apply(str)<"2020-12-31"]
#odds = odds[odds.Datetime.apply(str)<"2019-05-18"]
#odds = odds[odds.Datetime.apply(str)=="2019-05-11"]


def filter_better_odds(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Filter the n better rows by a rule. We use n highest absolute distances
    between sporstbook odd and real odd. This procedure will reduce the
    allocation array length.
    """
    df['odd_dist'] = df['Odd'] - 1/df['real_prob']
    return df.sort_values(['odd_dist'], ascending=False).head(n)


def process_group(group: Tuple[str, pd.DataFrame]):
    
    group_name, group_data = group
    
    games_ids = group_data['GameId'].unique()
    
    # Initialize dict to store dataframes of favorable bet opportunities
    odds_dict = {}

    # Initialze dict to store 7x7 matrices/dataframes of real probabilities 
    df_probs_dict = {}

    if len(games_ids) > 1:

        for game_id in games_ids:

            odds_sample = group_data[(group_data.GameId==game_id)]

            df = GameProbs(game_id).build_dataframe()

            odds_sample = apply_final_treatment(df_odds=odds_sample, df_real_prob=df)

            odds_sample = filter_better_odds(odds_sample, 5)
            
            #if len(odds_sample) > 50: # > 50
            #    break
            
            #logger.info(f"game_id: {game_id}, odds_sample.shape: {odds_sample.shape}")

            odds_dict[game_id] = odds_sample
            df_probs_dict[game_id] = df

        odds_dt = pd.concat(odds_dict.values())

        #if len(odds_dt) <= 80 and len(odds_dt) >20: # > 50
        if len(odds_dt) <= 80: # > 50
            
            odds_favorable = np.array(odds_dt['Odd'])
            real_prob_favorable = np.array(odds_dt['real_prob'])
            event_favorable = list(odds_dt['BetMap'].values)
            games_ids = np.array(odds_dt['GameId'])
                
            #try:
            x0 = np.zeros(len(odds_favorable))
            args = (odds_favorable, real_prob_favorable, event_favorable, games_ids, df_probs_dict)

            #logger.info("Execution of minimization task...")    
            print("Execution of minimization task...")    
            solution, time_limit_flag = Optimizer().optimize(fun=compute_objective_via_analytical,
                                                                x0=x0,
                                                                args=args)
                
            #logger.info("Finalization of minimization task...")
            print("Finalization of minimization task...")

            #except ValueError:
                #logger.info("ValueError for minimization task...")
                #continue

            odds_dt['solution'] = softmax(solution)

            track_record_list = []

            print(f"Date: {group_name}")

            for game_id, game_data in odds_dt.groupby('GameId', sort=False):
                
                print(f"game_id: {game_id}")

                scenario = gameid_to_outcome[game_id]

                financial_return = get_bet_return(df=game_data,
                                                    allocation_array=game_data.solution,
                                                    scenario=scenario)

                #track_record = {}
                #track_record['game_id'] = str(game_id)
                #track_record['return'] = financial_return
                #track_record['n_bets'] = len(game_data)
                #track_record['datetime'] = group_name
                #track_record['time_limit_flag'] = time_limit_flag
                track_record_list.append([str(game_id), financial_return, len(game_data), group_name, time_limit_flag])

            return track_record_list


if __name__ == "__main__":
    num_jobs = -1  # Use all available CPU cores for parallel execution
    grouped = odds.groupby('Datetime')

    # Parallelize the group processing
    results = Parallel(n_jobs=num_jobs)(delayed(process_group)(group) for group in grouped)

    data = [x for x in results if x is not None]

    flat_list = [item for sublist in data for item in sublist]

    pd.DataFrame(flat_list).to_csv("experiments/parallel_test.csv", index=False)

    elapsed_time = time() - start_time
    print("Elapsed: %.3f sec" % elapsed_time)    
