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
    build_empty_dataframe,
    apply_final_treatment,
)
from GameProbs import GameProbs
from utils import get_bet_return

GAME_ID = "2744178" # Chapecoense x Flamengo
game_id_list = [
'5924770',
 '2939252',
 '5255267',
 '5255635',
 '2921572',
 '5255194',
 '4396235',
 '5255296',
 '5924901',
 '2911344',
 '4698306',
 '4396233',
 '4698377',
 '4698321',
 '2992793',
 '5255530',
 '2939248',
 '5924787',
 '2934363',
 '4698368',
 '5254905',
 '5925281',
 '5924801',
 '5254935',
 '5925223'] # Chapecoense x Flamengo

metadata, gameid_to_outcome = load_metadata_artefacts("data/metadata.parquet")
odds = load_odds("data/odds.parquet")

track_record_list = []

for game_id in game_id_list:
    
    print(f"game_id: {game_id}")
    
    odds_sample = odds[(odds.GameId==game_id)]
    odds_sample = join_metadata(odds_sample, metadata)

    df = GameProbs(game_id).build_dataframe()

    odds_sample = apply_final_treatment(df_odds=odds_sample, df_real_prob=df)

    if len(odds_sample) > 50:
        continue

    odds_favorable = np.array(odds_sample['Odd'])
    real_prob_favorable = np.array(odds_sample['real_prob'])
    event_favorable = list(odds_sample['BetMap'].values)
    solution = minimize_analytical(public_odd=odds_favorable,
                                real_probabilities=real_prob_favorable,
                                event=event_favorable,
                                df_prob=df)
    solution = softmax(solution)
    print(f"solution:\n{solution}")

    scenario = gameid_to_outcome[game_id]
    print(f"scenario: {scenario}")

    financial_return = get_bet_return(df=odds_sample, allocation_array=solution, scenario=scenario)
    print(f"financial_return: {financial_return}")
    
    track_record = {}
    track_record['game_id'] = str(game_id)
    track_record['return'] = financial_return
    track_record_list.append(track_record)

pd.DataFrame(track_record_list).to_csv("track_record.csv", index=False)