import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import chain

from monte_carlo_return import (
    generate_bet_return,
    compute_objective_via_simulation,
    minimize_simulation,
)
from analytical_return import (
    expectation,
    second_moment,
    variance,
    compute_objective_via_analytical,
    minimize_analytical,
    softmax,
)
from data import (
    load_map,
    treat_odds,
    build_empty_dataframe,
    get_both_score_bet_df,
    get_exact_bet_df,
    add_real_prob,
    apply_final_treatment,
)
from GameProbs import GameProbs

metadata = pd.read_parquet("data/metadata.parquet")
odds = pd.read_parquet("data/odds.parquet")
odds = treat_odds(odds)

GAME_ID = "5254998"
my_game = GameProbs(GAME_ID) # Chapecoense x Flamengo
df = my_game.build_dataframe()

# Example of match that has 46 options of exact result bet 
sportsbook_list = ['1xBet', 'Megapari Sport', 'NetBet', 'Betobet', '18Bet', 'Mr Green Sport', 'Parimatch', 'Bet365']
odds_sample = odds[(odds.Sportsbook.isin(sportsbook_list))&(odds.GameId==GAME_ID)]

# Join metadata info to the odds dataframe
odds_sample = pd.merge(odds_sample, metadata[['GameId', 'Home', 'Away']], on=['GameId'], how='left')
odds_sample = odds_sample[odds_sample.Market.isin(['exact', 'both_score'])].reset_index(drop=True)
odds_sample = apply_final_treatment(df_odds=odds_sample, df_real_prob=df)
odds_sample_favorable = odds_sample[odds_sample.bet_flag].copy(deep=True)

odds_favorable = np.array(odds_sample_favorable['Odd'])
real_prob_favorable = np.array(odds_sample_favorable['real_prob'])
scenario_favorable = np.array(odds_sample_favorable['Scenario'])
solution = minimize_analytical(public_odd=odds_favorable,
                               real_probabilities=real_prob_favorable,
                               scenario=scenario_favorable)
print(sum(solution))
solution = softmax(solution)
print(sum(solution))
