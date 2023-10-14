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

metadata, gameid_to_outcome = load_metadata_artefacts("data/metadata.parquet")
odds = load_odds("data/odds.parquet")

track_record_list = []

count = 0

for game_id in GAMES_IDS[:70]:
    
    count+=1
    print(f"count: {count}")

    odds_sample = odds[(odds.GameId==game_id)]
    odds_sample = join_metadata(odds_sample, metadata)

    df = GameProbs(game_id).build_dataframe()

    odds_sample = apply_final_treatment(df_odds=odds_sample, df_real_prob=df)

    if len(odds_sample) > 80: # > 50
        continue

    odds_favorable = np.array(odds_sample['Odd'])
    real_prob_favorable = np.array(odds_sample['real_prob'])
    event_favorable = list(odds_sample['BetMap'].values)
    
    try:
        solution = minimize_analytical(public_odd=odds_favorable,
                                       real_probabilities=real_prob_favorable,
                                       event=event_favorable,
                                       df_prob=df)
    
    
    except ValueError:
        continue
       
    solution = softmax(solution)
    scenario = gameid_to_outcome[game_id]
    financial_return = get_bet_return(df=odds_sample, allocation_array=solution, scenario=scenario)

    logger.info(f"game_id: {game_id}; scenario: {scenario}; financial_return: {financial_return}")
    logger.info(f"solution:\n{[round(num, 3) for num in solution]}")
    logger.info('-' * 100)
    logger.info('-' * 100)

    track_record = {}
    track_record['game_id'] = str(game_id)
    track_record['return'] = financial_return
    track_record_list.append(track_record)

logger.info("END")
pd.DataFrame(track_record_list).to_csv("track_record.csv", index=False)