import numpy as np
import pandas as pd

from analytical_return import (
    minimize_analytical,
    softmax,
)
from data import (
    load_metadata_artefacts,
    load_odds,
    join_metadata,
    apply_final_treatment,
)
from GameProbs import GameProbs
from strategies import filter_better_odds
from utils import get_bet_return
from config import games_ids as GAMES_IDS

#GAME_ID = "2744178" # Chapecoense x Flamengo

metadata, gameid_to_outcome = load_metadata_artefacts("data/metadata.parquet")
odds = load_odds("data/odds.parquet")

track_record_list = []
count=0

for game_id in GAMES_IDS[:100]:
    
    count += 1
    print(f"count: {count}")
    
    odds_sample = odds[(odds.GameId==game_id)]
    odds_sample = join_metadata(odds_sample, metadata)

    df = GameProbs(game_id).build_dataframe()

    odds_sample = apply_final_treatment(df_odds=odds_sample, df_real_prob=df)
    
    #print(f"Shape: {odds_sample.shape}")
    #odds_sample = filter_better_odds(odds_sample, n=20)
    #print(f"Shape: {odds_sample.shape}")

    if len(odds_sample) > 70: # > 50
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
    

    print('---------------------------------------------')
    print(f"game_id: {game_id}")
    
    solution = softmax(solution)
    rounded_solution = [round(num, 3) for num in solution]
    print(f"solution:\n{rounded_solution}")

    scenario = gameid_to_outcome[game_id]
    print(f"scenario: {scenario}")

    financial_return = get_bet_return(df=odds_sample, allocation_array=solution, scenario=scenario)
    print(f"financial_return: {financial_return}")
    print('---------------------------------------------')


    track_record = {}
    track_record['count'] = count
    track_record['game_id'] = str(game_id)
    track_record['return'] = financial_return
    track_record_list.append(track_record)

pd.DataFrame(track_record_list).to_csv("track_record.csv", index=False)