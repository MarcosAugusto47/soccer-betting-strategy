import pandas as pd
import numpy as np

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

GAME_ID = "5254998" # Chapecoense x Flamengo

metadata, gameid_to_outcome = load_metadata_artefacts("data/metadata.parquet")
odds = load_odds("data/odds.parquet")

sportsbook_list = ['1xBet',
                   'Megapari Sport',
                   'NetBet',
                   'Betobet',
                   '18Bet',
                   'Mr Green Sport',
                   'Parimatch',
                   'Bet365']
odds_sample = odds[(odds.Sportsbook.isin(sportsbook_list))&(odds.GameId==GAME_ID)]
odds_sample = join_metadata(odds_sample, metadata)

my_game = GameProbs(GAME_ID) 
df = my_game.build_dataframe()

odds_sample = apply_final_treatment(df_odds=odds_sample, df_real_prob=df)

odds_sample = odds_sample[odds_sample.Market.isin(['exact', 'both_score'])]

odds_favorable = np.array(odds_sample['Odd'])
real_prob_favorable = np.array(odds_sample['real_prob'])
event_favorable = list(odds_sample['BetMap'].values)
solution = minimize_analytical(public_odd=odds_favorable,
                               real_probabilities=real_prob_favorable,
                               event=event_favorable,
                               df_prob=df)
solution = softmax(solution)
print(f"solution:\n{solution}")
