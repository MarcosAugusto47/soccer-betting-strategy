import pandas as pd


def filter_better_odds(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Filter the n better rows by a rule. We use n highest absolute distances
    between sporstbook odd and real odd. This procedure will reduce the
    allocation array length.
    """
    df['odd_dist'] = df['Odd'] - 1/df['real_prob']
    return df.sort_values(['odd_dist'], ascending=False).head(n)