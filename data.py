import pandas as pd
import re
import json
import numpy as np

from itertools import chain


def load_metadata_artefacts(file_path: str):
    metadata = pd.read_parquet(file_path)
    metadata = metadata[~metadata.Home_score.isna()]
    metadata['Home_score'] = metadata.Home_score.apply(int)
    metadata['Away_score'] = metadata.Away_score.apply(int)
    metadata['Outcome'] = metadata.apply(lambda x: f"{str(x['Home_score'])} : { str(x['Away_score'])}", axis=1)

    gameid_to_outcome = metadata.set_index(['GameId'])['Outcome'].to_dict()

    return metadata, gameid_to_outcome


def load_map(file_path):

    # Open the JSON file for reading
    with open(file_path, 'r') as json_file:
        return json.load(json_file)


def load_odds(file_path):
    df = pd.read_parquet(file_path)
    df = df[df.League=="Brasileirão Série A"].drop("League", axis=1)
    df['Odd'] = df['Odd'].astype(float)
    df['public_prob'] = 1/df['Odd']
    return df


def build_empty_dataframe(nrow, ncol, value):

    # Create a dataframe with 7 rows and 7 columns filled with np.nan
    data = np.full((nrow, ncol), value)
    df = pd.DataFrame(data)
    df.columns = [str(number) for number in range(ncol)]
    df.index = [str(number) for number in range(nrow)]

    return df


def get_both_score_bet_df():
    """ Get map of bets for both_score market."""
    map_bet = load_map('data/probs.json')
    market = 'both_score'
    both_score_normalized = pd.json_normalize(map_bet[market])
    both_score_normalized = pd.melt(both_score_normalized, var_name='Bet', value_name='BetMap')
    both_score_normalized['Market'] = market
    return both_score_normalized


def get_exact_bet_df():
    """ Get map of bets for exact market."""
    map_bet = load_map('data/probs.json')
    market = 'exact'
    exact_normalized = pd.json_normalize(map_bet[market])
    exact_normalized.columns = [re.sub( '.o', '', col) for col in exact_normalized.columns]
    exact_normalized = pd.melt(exact_normalized, var_name='Scenario', value_name='BetMap')
    exact_normalized['Market'] = market
    return exact_normalized


def add_real_prob(df_bet, df_prob):
    """Add real probability of scenario."""
    for k in range(len(df_bet)):
    
        scenario = df_bet.loc[k, "Scenario"]
    
        # Split the Scenario string in two pieces
        scenario_list = scenario.split(' : ')
    
        try:
            i = scenario_list[0]  # home team
            j = scenario_list[1]  # away team
            df_bet.loc[k, "real_prob"] = df_prob.loc[j, i]

        except KeyError:
            pass
    
    return df_bet

class DataFrameFromBetMap:
    def __init__(self):
        map_bet = load_map('data/probs.json')
        self.map_bet = map_bet
    
    def get_both_score(self):
        """ Get map of bets for both_score market."""
        market = 'both_score'
        both_score_normalized = pd.json_normalize(self.map_bet[market])
        both_score_normalized = pd.melt(both_score_normalized, var_name='Bet', value_name='BetMap')
        both_score_normalized['Market'] = market
        self.both_score_normalized = both_score_normalized
        return self.both_score_normalized

    def get_exact(self):
        """ Get map of bets for exact market."""
        market = 'exact'
        exact_normalized = pd.json_normalize(self.map_bet[market])
        exact_normalized.columns = [re.sub( '.o', '', col) for col in exact_normalized.columns]
        exact_normalized = pd.melt(exact_normalized, var_name='Scenario', value_name='BetMap')
        exact_normalized['Market'] = market
        self.exact_normalized = exact_normalized
        return self.exact_normalized


def apply_final_treatment(
    df_odds: pd.DataFrame,
    df_real_prob: pd.DataFrame,
) -> pd.DataFrame:
    """
    Applies final treatment to bets and odds dataframe

    Args:
        df_odds: dataframe with public odds
        df_real_prob: 7x7 dataframe with the real probabilities given by the statistical model
    """

    # Get instance of DataFrameFromBetMap
    map_bet = DataFrameFromBetMap()

    df_exact = df_odds[df_odds.Market=='exact']
    
    # Add column of map of bets as list
    exact_normalized = map_bet.get_exact()
    df_exact = pd.merge(df_exact, exact_normalized, on=['Scenario', 'Market'], how='left')
    

    df_both_score = df_odds[df_odds.Market=='both_score']
    
    # Add column of map of bets as list
    both_score_normalized = map_bet.get_both_score()
    df_both_score = pd.merge(df_both_score, both_score_normalized, on=['Bet', 'Market'], how='left')
    
    df_new = pd.concat([df_exact, df_both_score], ignore_index=True)

    df_new = df_new.dropna(subset=['BetMap']).reset_index(drop=True)

    #df_new = df_new[df_new.Market=='exact']

    # odds_sample = odds_sample[odds_sample.Scenario.apply(lambda x: "7" not in x)].reset_index(drop=True)


    def add_real_prob2(x, df_real_prob):
        
        bet_matrix = np.transpose(np.array(x).reshape(7,7))
        matrix_mult = bet_matrix * df_real_prob.to_numpy()
        return sum(list(chain(*matrix_mult)))


    #df_new = add_real_prob(df_new, df_real_prob)
    df_new['real_prob'] = df_new.BetMap.apply(lambda x: add_real_prob2(x, df_real_prob))

    # Flag if public odd > predicted odd
    df_new['bet_flag'] =  df_new['public_prob'] < df_new['real_prob']

    return df_new