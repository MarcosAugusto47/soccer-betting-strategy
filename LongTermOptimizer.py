import numpy as np

from monte_carlo_return import generate_bet_return
from dependencies.utils import softmax


def generate_single_long_term_return(
    df_prob, df_bet, num_simulations, allocation_array
) -> float:

    bet_returns = generate_bet_return(
        df_prob=df_prob,
        df_bet=df_bet,
        num_simulations=num_simulations,
        allocation_array=allocation_array,
    )

    long_term_return = np.prod(np.array(bet_returns))
    #print(f"long_term_return: {np.round(long_term_return, 2)}")

    return long_term_return


def estimate_long_term_return_mean(
    x,  # allocation array
    df_prob,
    df_bet,
    num_simulations,
):
    
    x = softmax(x)

    observations = [
        generate_single_long_term_return(df_prob, df_bet, num_simulations, x)
        for _ in range(100)
    ]
    mean_long_term_return = np.mean(observations)

    return -mean_long_term_return


def estimate_long_term_return_sharpe_ratio(
    x,  # allocation array
    df_prob,
    df_bet,
    num_simulations,
):
    
    x = softmax(x)

    observations = [
        generate_single_long_term_return(df_prob, df_bet, num_simulations, x)
        for _ in range(100)
    ]
    mean_long_term_return = np.mean(observations)

    sigma = np.std(observations)

    return -mean_long_term_return / sigma


def estimate_long_term_return_prob(
    x,  # allocation array
    df_prob,
    df_bet,
    num_simulations,
):
    
    x = softmax(x)

    observations = [
        generate_single_long_term_return(df_prob, df_bet, num_simulations, x)
        for _ in range(100)
    ]

    probability_win = np.sum(np.array(observations) > 1) / len(observations)

    return -probability_win
