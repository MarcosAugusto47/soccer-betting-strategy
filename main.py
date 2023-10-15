import numpy as np
import pandas as pd
import logging

from analytical_return import (
    minimize_analytical,
    softmax,
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

logging.basicConfig(level=logging.INFO, filename="my_log.log", filemode="w")
logger = logging.getLogger(__name__)

logger.info("RUN")

metadata, gameid_to_outcome = load_metadata_artefacts("data/metadata-with-date.parquet")
odds = load_odds("data/odds.parquet")
odds = join_metadata(odds, metadata)

track_record_list = []

count = 0

for group_name, group_data in odds.groupby(['Datetime']):

    count+=1
    print(f"count: {count}")

    if count > 2:
        break

    games_ids = group_data['GameId'].unique()
    
    logger.info(f"Datetime: {group_name} with {len(games_ids)} games")

    list_dfs = []

    for game_id in games_ids:

        odds_sample = odds[(odds.GameId==game_id)]

        df = GameProbs(game_id).build_dataframe()

        odds_sample = apply_final_treatment(df_odds=odds_sample, df_real_prob=df)
        
        if len(odds_sample) > 50: # > 50
            break
        
        logger.info(f"game_id: {game_id}")
        logger.info(f"odds_sample.shape: {odds_sample.shape}")

        list_dfs.append(odds_sample)

    odds_dt = pd.concat(list_dfs)
    
    logger.info(f"odds_dt.shape: {odds_dt.shape}")

    odds_favorable = np.array(odds_dt['Odd'])
    real_prob_favorable = np.array(odds_dt['real_prob'])
    event_favorable = list(odds_dt['BetMap'].values)
        
    try:
        solution = minimize_analytical(public_odd=odds_favorable,
                                       real_probabilities=real_prob_favorable,
                                       event=event_favorable,
                                       df_prob=df)
    
    
    except ValueError:
        continue
    
    solution = softmax(solution)

    odds_dt['solution'] = solution

    for game_id, game_data in odds_dt.groupby(['GameId'], sort=False):
        scenario = gameid_to_outcome[game_id]
        financial_return = get_bet_return(df=game_data, allocation_array=game_data.solution, scenario=scenario)

        logger.info(f"game_id: {game_id}; scenario: {scenario}; financial_return: {financial_return}")
        logger.info(f"solution:\n{[round(num, 3) for num in solution]}")
        logger.info('-' * 100)
        logger.info('-' * 100)

        track_record = {}
        track_record['game_id'] = str(game_id)
        track_record['return'] = financial_return
        track_record['Datetime'] = group_name
        track_record_list.append(track_record)

logger.info("END")
pd.DataFrame(track_record_list).to_csv("track_record.csv", index=False)