import numpy as np
import pandas as pd
import logging

from analytical_return import (
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
from time import time
from dependencies.utils import softmax

start_time = time()

logging.basicConfig(level=logging.INFO, filename="my_log.log", filemode="w")
logger = logging.getLogger(__name__)

logger.info("RUN")

metadata, gameid_to_outcome = load_metadata_artefacts("data/metadata-with-date.parquet")
odds = load_odds("data/odds.parquet")
odds = join_metadata(odds, metadata)

track_record_list = []

count = 0

odds = odds[odds.Datetime.apply(str)<"2019-05-18"]


def filter_better_odds(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Filter the n better rows by a rule. We use n highest absolute distances
    between sporstbook odd and real odd. This procedure will reduce the
    allocation array length.
    """
    df['odd_dist'] = df['Odd'] - 1/df['real_prob']
    return df.sort_values(['odd_dist'], ascending=False).head(n)


for group_name, group_data in odds.groupby('Datetime'):

    count+=1
    print(f"count: {count}")

    #if count > 9:
    #    break

    games_ids = group_data['GameId'].unique()
    
    logger.info(f"Datetime: {group_name} with {len(games_ids)} games")

    # Initialize dict to store dataframes of favorable bet opportunities
    odds_dict = {}

    # Initialze dict to store 7x7 matrices/dataframes of real probabilities 
    df_probs_dict = {}

    if len(games_ids) <=1 :
        continue

    for game_id in games_ids:

        odds_sample = odds[(odds.GameId==game_id)]

        df = GameProbs(game_id).build_dataframe()

        odds_sample = apply_final_treatment(df_odds=odds_sample, df_real_prob=df)

        odds_sample = filter_better_odds(odds_sample, 5)
        
        #if len(odds_sample) > 50: # > 50
        #    break
        
        logger.info(f"game_id: {game_id}, odds_sample.shape: {odds_sample.shape}")

        odds_dict[game_id] = odds_sample
        df_probs_dict[game_id] = df

    odds_dt = pd.concat(odds_dict.values())

    #if len(odds_dt) > 80 or len(odds_dt) <=20: # > 50
    if len(odds_dt) > 80: # > 50
        continue

    #if len(odds_dt) > 400: # > 50
    #    continue
    
    odds_favorable = np.array(odds_dt['Odd'])
    real_prob_favorable = np.array(odds_dt['real_prob'])
    event_favorable = list(odds_dt['BetMap'].values)
    games_ids = np.array(odds_dt['GameId'])
        
    try:
        x0 = np.zeros(len(odds_favorable))
        args = (odds_favorable, real_prob_favorable, event_favorable, games_ids, df_probs_dict)

        logger.info("Execution of minimization task...")    
        solution, time_limit_flag = Optimizer().optimize(fun=compute_objective_via_analytical,
                                                         x0=x0,
                                                         args=args)
        
        logger.info("Finalization of minimization task...")

    except ValueError:
        logger.info("ValueError for minimization task...")
        continue
    
    solution = softmax(solution)

    if len(set([round(num, 3) for num in solution]))==1:
        logger.warning("No distinct values in solution")

    odds_dt['solution'] = solution

    for game_id, game_data in odds_dt.groupby('GameId', sort=False):
        scenario = gameid_to_outcome[game_id]

        financial_return = get_bet_return(df=game_data,
                                          allocation_array=game_data.solution,
                                          scenario=scenario)

        logger.info(f"game_id: {game_id}; scenario: {scenario}; financial_return: {financial_return}")
        logger.info(f"solution:\n{[round(num, 3) for num in solution]}")

        track_record = {}
        track_record['game_id'] = str(game_id)
        track_record['return'] = financial_return
        track_record['n_bets'] = len(game_data)
        track_record['datetime'] = group_name
        track_record['time_limit_flag'] = time_limit_flag
        track_record_list.append(track_record)

    logger.info('-' * 100)
    logger.info('-' * 100)

logger.info("END")
pd.DataFrame(track_record_list).to_csv("experiments/main_callback.csv", index=False)
elapsed_time = time() - start_time
print("Final Elapsed: %.3f sec" % elapsed_time)
