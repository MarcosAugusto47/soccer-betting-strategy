import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger


def save_csv_artifact(folder, path, data):

    csv_file_path = f"{folder}/{path}.csv"
    data.to_csv(csv_file_path, index=False)
    logger.info(f"CSV Artifact saved to: {csv_file_path}")


def process_results(aggregator, do_baseline, track_record):

    if not do_baseline:
        count_col = track_record.groupby(aggregator, sort=False).count().reset_index()['GameId']
        is_valid_solution = track_record.groupby(aggregator, sort=False).any().reset_index()['is_valid_solution']
        n_favorable_bets = track_record.groupby("Datetime").first().reset_index()['n_favorable_bets']
        gamma = track_record.groupby(aggregator, sort=False).first().reset_index()['gamma']
        track_record = track_record.groupby(aggregator, sort=False)[['return', 'n_bets']].sum().reset_index()
        track_record['count'] = count_col
        track_record['is_valid_solution'] = is_valid_solution
        track_record['n_favorable_bets'] = n_favorable_bets
        track_record['gamma'] = gamma
        track_record = track_record[track_record.is_valid_solution]
    
    return track_record


def compute_stake_baseline_kreiner(df, budget=100):
    stake = [budget]

    for index, r in enumerate(df['return']):
        budget = budget - df['n_bets'][index] + r
        stake.append(budget)
    
    return stake


def compute_variable_stake(df, stake=1):
    stake = [stake]
    current_stake = stake[0]

    for i, k in zip(df['return'], df['gamma']):
        preserved_stake = current_stake * (1-k)
        bet_stake = current_stake*k
        current_stake = preserved_stake + bet_stake*i

        stake.append(current_stake)
    
    return stake


def build_plot_df(stake, do_baseline, df):

    if do_baseline:
        df['Datetime'] = df.apply(lambda x: f"{x['Datetime']}_{x['GameId']}", axis=1)

    plot_df = pd.DataFrame(
        {'date': ['0'] + list(df.Datetime),
        'return': [0] + list(df['return']),
        #'count': [0] + list(df['count']),
        #'n_bets': [0] + list(df.n_bets),
        'stake': stake,
        'n_favorable_bets': [0] + list(df.n_favorable_bets)}
    )

    return plot_df


def save_plot_strategy(path, df):
    """ Create a line plot."""
    plt.figure(figsize=(20, 6))
    plt.plot(df.date, df.stake, linestyle='-')
    plt.title('Cumulative Profits Over Time (%)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profits')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{path}/plot.PNG")


def build_plot_df_wrapper_baseline_kreiner(path, aggregator, do_baseline):
    track_record = pd.read_csv(f"{path}/result.csv")
    track_record.columns = ['GameId', 'return',	'n_bets', 'n_favorable_bets', 'time_limit_flag', 'is_valid_solution', 'Datetime']
    df = process_results(aggregator, do_baseline, track_record)
    stake = compute_stake_baseline_kreiner(df)
    return build_plot_df(stake, do_baseline, df)


def build_plot_df_wrapper(path, aggregator, do_baseline):
    track_record = pd.read_csv(f"{path}/result.csv")
    track_record.columns = ['GameId', 'return',	'n_bets', 'n_favorable_bets', 'gamma', 'time_limit_flag', 'is_valid_solution', 'Datetime']
    df = process_results(aggregator, do_baseline, track_record)
    stake = compute_variable_stake(df)
    return build_plot_df(stake, do_baseline, df)
