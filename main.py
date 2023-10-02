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
from utils import get_bet_return

GAME_ID = "2744178" # Chapecoense x Flamengo

metadata, gameid_to_outcome = load_metadata_artefacts("data/metadata.parquet")
odds = load_odds("data/odds.parquet")

odds_sample = odds[(odds.GameId==GAME_ID)]
odds_sample = join_metadata(odds_sample, metadata)

df = GameProbs(GAME_ID).build_dataframe()

odds_sample = apply_final_treatment(df_odds=odds_sample, df_real_prob=df)

odds_favorable = np.array(odds_sample['Odd'])
real_prob_favorable = np.array(odds_sample['real_prob'])
event_favorable = list(odds_sample['BetMap'].values)
solution = minimize_analytical(public_odd=odds_favorable,
                               real_probabilities=real_prob_favorable,
                               event=event_favorable,
                               df_prob=df)
solution = softmax(solution)
print(f"solution:\n{solution}")

scenario = gameid_to_outcome[GAME_ID]
print(f"scenario: {scenario}")

financial_return = get_bet_return(df=odds_sample, allocation_array=solution, scenario=scenario)
print(f"financial_return: {financial_return}")
