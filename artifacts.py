import pandas as pd
import matplotlib.pyplot as plt


def save_csv_artifact(folder, path, data):

    csv_file_path = f"{folder}/{path}.csv"
    data.to_csv(csv_file_path, index=False)
    print(f"CSV Artifact saved to: {csv_file_path}")


def process_results(aggregator, do_baseline, track_record):

    if not do_baseline:
        count_col = track_record.groupby(aggregator, sort=False).count().reset_index()['game_id']
        is_valid_solution = track_record.groupby(aggregator, sort=False).any().reset_index()['is_valid_solution']
        track_record = track_record.groupby(aggregator, sort=False)[['return', 'n_bets']].sum().reset_index()
        track_record['count'] = count_col
        track_record['is_valid_solution'] = is_valid_solution
        track_record = track_record[track_record.is_valid_solution]
    
    return track_record


def compute_stake(df):
    stake = [1]
    current_stake = 1
    percentage = 0.10
    for i in df['return']:

        preserved_stake = current_stake * (1-percentage)
        bet_stake = current_stake*percentage
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
        'stake': stake}
    )

    return plot_df


def save_plot_strategy(args, df):
    """ Create a line plot."""
    plt.figure(figsize=(20, 6))
    plt.plot(df.date, df.stake, linestyle='-')
    plt.title('Cumulative Profits Over Time (%)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profits')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.artefacts_folder}/plot.PNG")


def build_plot_df_wrapper(args):
    track_record = pd.read_csv(f"{args.artefacts_folder}/result.csv")
    track_record.columns = ['GameId', 'return',	'n_bets', 'time_limit_flag', 'is_valid_solution', 'Datetime']
    df = process_results(args.aggregator, args.do_baseline, track_record)
    stake = compute_stake(df)
    return build_plot_df(stake, args.do_baseline, df)
