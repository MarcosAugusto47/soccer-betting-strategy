import pandas as pd
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
    #df = df[df.Market!="spread"]
    df['Odd'] = df['Odd'].astype(float)
    df['public_prob'] = 1/df['Odd']

    return df


def join_metadata(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    
    df = pd.merge(left=df,
                  right=metadata[['GameId', 'Home', 'Away']],
                  on=['GameId'],
                  how='left') 
    
    return df


def build_empty_dataframe(nrow, ncol, value):

    # Create a dataframe with 7 rows and 7 columns filled with np.nan
    data = np.full((nrow, ncol), value)
    df = pd.DataFrame(data)
    df.columns = [str(number) for number in range(ncol)]
    df.index = [str(number) for number in range(nrow)]

    return df


# def get_both_score_bet_df():
#     """ Get map of bets for both_score market."""
#     map_bet = load_map('data/probs.json')
#     market = 'both_score'
#     both_score_normalized = pd.json_normalize(map_bet[market])
#     both_score_normalized = pd.melt(both_score_normalized, var_name='Bet', value_name='BetMap')
#     both_score_normalized['Market'] = market
#     return both_score_normalized


# def get_exact_bet_df():
#     """ Get map of bets for exact market."""
#     map_bet = load_map('data/probs.json')
#     market = 'exact'
#     exact_normalized = pd.json_normalize(map_bet[market])
#     exact_normalized.columns = [re.sub( '.o', '', col) for col in exact_normalized.columns]
#     exact_normalized = pd.melt(exact_normalized, var_name='Scenario', value_name='BetMap')
#     exact_normalized['Market'] = market
#     return exact_normalized


# def add_real_prob(df_bet, df_prob):
#     """Add real probability of scenario."""
#     for k in range(len(df_bet)):

#         scenario = df_bet.loc[k, "Scenario"]

#         # Split the Scenario string in two pieces
#         scenario_list = scenario.split(' : ')

#         try:
#             i = scenario_list[0]  # home team
#             j = scenario_list[1]  # away team
#             df_bet.loc[k, "real_prob"] = df_prob.loc[j, i]

#         except KeyError:
#             pass

#     return df_bet

class DataFrameFromBetMap:
    def __init__(self):
        map_bet = load_map('data/probs.json')
        self.map_bet = map_bet
    
    def get_flat(self):
        """ Get map of bets for all markets as a flat table."""
        df = pd.DataFrame()
        
        for market in ['h2h', 'both_score', 'spread', 'over/under', 'exact']:
            df = pd.concat(objs=[df, self._get_flat_table(market)])
        
        df.replace(np.nan, '', inplace=True)

        return df

    def _get_flat_table(self, market):
        """ Get map of bets for a given market."""
        df_normalized = pd.json_normalize(self.map_bet[market], sep=',')              
        df_normalized = pd.melt(df_normalized, var_name='BetTemp', value_name='BetMap')
        
        bet_info_splitted = df_normalized['BetTemp'].str.split(pat=',', n=1, expand=True)
        
        try:
            df_normalized[['Scenario', 'Bet']] = bet_info_splitted
        
        except ValueError:
            df_normalized['Bet'] = bet_info_splitted
        
        df_normalized['Market'] = market
        df_normalized.drop(['BetTemp'], axis=1, inplace=True)
        
        return df_normalized


def apply_final_treatment(
    df_odds: pd.DataFrame,
    df_real_prob: pd.DataFrame,
) -> pd.DataFrame:
    """
    Applies final treatment to bets and odds dataframe

    Args:
        df_odds: dataframe with public odds
        df_real_prob: 7x7 dataframe with the real probabilities given by the
        statistical model
    
    Returns:
        Final treated pandas dataframe with only bet opportunities that
        the public odd is greater than the real odd.
    """

    # Get flat table from instance of DataFrameFromBetMap
    map_bet = DataFrameFromBetMap().get_flat()

    # Add column of list of map of bets
    df_odds = pd.merge(df_odds, map_bet, on=['Market', 'Bet', 'Scenario'], how='left')

    df_odds = df_odds.dropna(subset=['BetMap']).reset_index(drop=True)


    def add_real_prob(x: list, df_real_prob: pd.DataFrame) -> float:
        """
        Add the real probability of a bet event via the multiplicaiton
        of a list with lenght of 7x7 filled with 0's and 1's and the matrix of real
        probabilities 
        """
        bet_matrix = np.transpose(np.array(x).reshape(7,7))
        matrix_mult = bet_matrix * df_real_prob.to_numpy()

        return sum(list(chain(*matrix_mult)))

    # Compute the real probabilitity of the event associated to the bet
    df_odds['real_prob'] = df_odds.BetMap.apply(lambda x: add_real_prob(x, df_real_prob))

    # Flag if public odd > predicted odd
    df_odds['bet_flag'] =  df_odds['public_prob'] < df_odds['real_prob']

    df_odds = df_odds[df_odds.bet_flag]

    return df_odds
