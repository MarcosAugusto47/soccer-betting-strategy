import numpy as np
import pandas as pd
import logging

from analytical_return import (
    minimize_analytical,
)
from data import (
    load_metadata_artefacts,
    load_odds,
    join_metadata,
    build_empty_dataframe,
    apply_final_treatment,
)
from GameProbs import GameProbs
from utils import get_bet_return
from config import games_ids as GAMES_IDS
from dependencies.utils import softmax

logging.basicConfig(level=logging.INFO, filename="my_log.log", filemode="w")
logger = logging.getLogger(__name__)

logger.info("RUN")

metadata, gameid_to_outcome = load_metadata_artefacts("data/metadata-with-date.parquet")
odds = load_odds("data/odds.parquet")
odds = join_metadata(odds, metadata)

#odds = odds[odds.Datetime.apply(str) >= "2019-10-12"]

track_record_list = []

count = 0

for group_name, group_data in odds.groupby(['Datetime']):

    count+=1
    print(f"count: {count}")

    #if count > 200:
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
        
        #if len(odds_sample) > 50: # > 50
        #    break
        
        logger.info(f"game_id: {game_id}, odds_sample.shape: {odds_sample.shape}")

        odds_dict[game_id] = odds_sample
        df_probs_dict[game_id] = df

    odds_dt = pd.concat(odds_dict.values())

    #if len(odds_dt) > 80 or len(odds_dt) <=20: # > 50
    #    continue

    if len(odds_dt) > 80: # > 50
        continue
    
    logger.info(f"odds_dt.shape: {odds_dt.shape}")
    logger.info(f"len(df_probs_dict): {len(df_probs_dict)}")

    odds_favorable = np.array(odds_dt['Odd'])
    real_prob_favorable = np.array(odds_dt['real_prob'])
    event_favorable = list(odds_dt['BetMap'].values)
    games_ids = np.array(odds_dt['GameId'])
        
    try:
        solution = minimize_analytical(public_odd=odds_favorable,
                                       real_probabilities=real_prob_favorable,
                                       event=event_favorable,
                                       games_ids=games_ids,
                                       df_probs_dict=df_probs_dict)
    
    
    except ValueError:
        continue
    
    solution = softmax(solution)

    logger.info(f"len(solution): {len(solution)}")

    odds_dt['solution'] = solution

    for game_id, game_data in odds_dt.groupby(['GameId'], sort=False):
        scenario = gameid_to_outcome[game_id]
        logger.info(f"game_data.shape: {game_data.shape}")
        logger.info(f"len(game_data.solution): {len(game_data.solution)}")
        

        financial_return = get_bet_return(df=game_data, allocation_array=game_data.solution, scenario=scenario)

        logger.info(f"game_id: {game_id}; scenario: {scenario}; financial_return: {financial_return}")
        logger.info(f"solution:\n{[round(num, 3) for num in solution]}")
        logger.info('-' * 100)
        logger.info('-' * 100)

        track_record = {}
        track_record['game_id'] = str(game_id)
        track_record['return'] = financial_return
        track_record['n_bets'] = len(game_data)
        track_record['datetime'] = group_name
        track_record_list.append(track_record)

logger.info("END")
pd.DataFrame(track_record_list).to_csv("track_record_count_200_powell_01_onlydates_with_bets_between80-20_all_time_serie.csv", index=False)