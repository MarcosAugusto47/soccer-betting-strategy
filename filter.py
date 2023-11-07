import pandas as pd
import numpy as np

def filter_by_linear_combination(
        df: pd.DataFrame,
        n: int = 5,
        weight: float = 0.5
) -> pd.DataFrame:
    """
    Drops duplicates, to assure more bets diversity. Filters the n better rows
    by a linear combination of two components: (1) normalized distance from
    public odd to real odd, (2) real probability associated to the bet/row
    Finally, order it and select the first n rows.
    """
    df.drop_duplicates(subset=['Market', 'Scenario', 'Bet'], inplace=True)

    df['odd_dist'] = np.round((df['Odd'] - 1/df['real_prob']) / df['Odd'], 1)
    df['score'] = weight*df['odd_dist'] + (1-weight)*(df['real_prob'])
    
    return df.sort_values(['score'], ascending=False).head(n)
