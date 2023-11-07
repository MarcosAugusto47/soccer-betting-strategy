import numpy as np
import pandas as pd
import logging

from data import (
    load_metadata_artefacts,
    load_odds,
    join_metadata,
    apply_final_treatment,
)
from GameProbs import GameProbs
from Optimizer import Optimizer
from utils import (
    get_index_to_scenario_for_betmap,
    find_positions,
    get_values_by_keys,
    get_scenarios,
    get_bet_return,
)
from config import games_ids as GAMES_IDS
from joblib import Parallel, delayed
from typing import Tuple
from time import time

metadata, gameid_to_outcome = load_metadata_artefacts("data/metadata-with-date.parquet")
odds = load_odds("data/odds.parquet")
odds = join_metadata(odds, metadata)

#odds = odds[(odds.Datetime.apply(str)<"2022-01-01")]
#odds = odds[(odds.Datetime.apply(str)>"2021-01-01")&(odds.Datetime.apply(str)<"2021-02-01")]
#odds = odds[(odds.Datetime.apply(str)>"2021-02-01")&(odds.Datetime.apply(str)<"2021-05-29")]
#odds = odds[(odds.Datetime.apply(str)>"2020-11-01")&(odds.Datetime.apply(str)<"2021-01-01")]
odds = odds[(odds.Datetime.apply(str)>"2019-05-01")&(odds.Datetime.apply(str)<"2019-06-01")]


def filter_better_odds(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Filter the n better rows by a rule. We use n highest absolute distances
    between sporstbook odd and real odd. This procedure will reduce the
    allocation array length.
    """
    df['odd_dist'] = np.round((df['Odd'] - 1/df['real_prob']) / df['Odd'], 1)
    #df['real_prob'] = np.round(df['real_prob'], 2)
    #df['score'] = df['odd_dist'] * df['real_prob']
    w = 0.5
    df['score'] = w*df['odd_dist'] + (1-w)*(df['real_prob'])
    df = df.drop_duplicates(subset=['Market', 'Scenario', 'Bet'])
    return df.sort_values(['score'], ascending=False).head(n)


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
                       
            #logger.info(f"game_id: {game_id}, odds_sample.shape: {odds_sample.shape}")
            #print(f"game_id: {game_id}, odds_sample.shape: {odds_sample.shape}")

            odds_dict[game_id] = odds_sample
            df_probs_dict[game_id] = df

        odds_dt = pd.concat(odds_dict.values())

        print(f"odds_dt.shape: {odds_dt.shape}")

        if len(odds_dt) <= 80: # > 50
            
            print(f"Date: {group_name}")

            for game_id, game_data in odds_dt.groupby('GameId', sort=False):
                
                #print(f"game_id: {game_id}")

                scenario = gameid_to_outcome[game_id]

                check_scenario = lambda x: scenario in x
                
                # Check if scenario is inside the BetMap
                game_data['flag'] = game_data['BetMap'].apply(get_scenarios).apply(check_scenario)

                print(f"sum(game_data['flag']): {sum(game_data['flag'])}")


if __name__ == "__main__":

    start_time = time()
    
    # Use all available CPU cores for parallel execution
    num_jobs = 1

    grouped = odds.groupby('Datetime')

    # Parallelize the group processing
    Parallel(n_jobs=num_jobs)(delayed(process_group)(group) for group in grouped)

    elapsed_time = time() - start_time
    print("Final Elapsed: %.3f sec" % elapsed_time)
