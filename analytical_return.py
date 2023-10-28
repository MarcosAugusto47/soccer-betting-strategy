import numpy as np
import pandas as pd
import logging

from scipy.optimize import minimize
from itertools import chain
from typing import Dict

#logger = logging.getLogger(__name__)

def expectation(allocation, public_odd, real_probabilities):
    return sum(public_odd * allocation * real_probabilities)


def second_moment(allocation: list,
                  public_odd: np.array,
                  real_probabilities: np.array,
                  event: np.array,
                  games_ids: np.array,
                  df_probs_dict: Dict[str, pd.DataFrame],
) -> float:
    """
    Computes the second moment of the random variable.

    Note:
        Check mathematical definition solved by hand at docs/return-analytical-definition.png

    Args:
        allocation: allocation of percentages of bet distribution
        public_odd: array of odds from Sports books
        real_probabilities: array of real probabilities of bet opportunities
        event: array of 7x7 lenght arrays that maps the scenarios
        df_prob: 7x7 pandas dataframe of real probablities, one for each scenario
    
    Returns:
        float: final computation
    """
    
    term1 = sum((public_odd * allocation)**2 * real_probabilities)

    term2_list = []

    n = len(public_odd)

    for i in range(n):
        for j in range(n):
            if i != j:
                
                theta_i = public_odd[i] * allocation[i]
                theta_j = public_odd[j] * allocation[j]
                theta_ij = theta_i * theta_j
                
                prob_ij = 0
                
                # Get event i 7x7 matrix
                event_i = np.transpose(np.array(event[i]).reshape(7,7))
                # Get event i 7x7 matrix
                event_j = np.transpose(np.array(event[j]).reshape(7,7))
                
                # for events from a single game, we get the intersection. So
                # {A} intersection with {B} is, in fact, the soccer scenarios
                # common in both events
                if games_ids[i] == games_ids[j]:
                
                    # Get events intersection, both matrices are filled with 0's or 1's
                    event_intersection_matrix = event_i * event_j
                    # Get events inserction mapped real probabilities
                    event_intersection_matrix_prob = event_intersection_matrix * df_probs_dict[games_ids[i]].to_numpy()
                    # Compute the final events intersection probability
                    prob_ij = sum(list(chain(*event_intersection_matrix_prob)))
                
                # for events from different games, we treat them indepently. So
                # P({A} intersection with {B}) = P({A}) * P({B})
                else:
                    # Get probablities matrix of event i of associated game
                    event_i_prob = event_i * df_probs_dict[games_ids[i]].to_numpy()
                    # Get probabilities matrix of event j of associated game, different from game above
                    event_j_prob = event_j * df_probs_dict[games_ids[j]].to_numpy()
                    # Compute probability of event i
                    prob_i = sum(list(chain(*event_i_prob)))
                    # Compute probability of event j
                    prob_j = sum(list(chain(*event_j_prob)))
                    # Compute of the intersection as the product
                    prob_ij = prob_i * prob_j

                term2_list.append(theta_ij*prob_ij)
    
    term2 = sum(term2_list)
            
    return term1 + term2


def variance(second_moment, expectation):
    return second_moment - (expectation)**2


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def compute_objective_via_analytical(
    x: np.ndarray,
    public_odd: np.ndarray,
    real_probabilities: np.ndarray,
    event: np.ndarray,
    games_ids: np.ndarray,
    df_probs_dict: Dict[str, pd.DataFrame],
) -> np.float64:

    x = softmax(x)

    my_expectation = expectation(allocation=x,
                                 public_odd=public_odd,
                                 real_probabilities=real_probabilities)
    #print(f"my_expectation: {my_expectation}")
    #logger.info(f"my_expectation: {my_expectation}")

    my_second_moment = second_moment(allocation=x,
                                     public_odd=public_odd,
                                     real_probabilities=real_probabilities,
                                     event=event,
                                     games_ids=games_ids,
                                     df_probs_dict=df_probs_dict)

    my_sigma = np.sqrt(variance(my_second_moment, my_expectation))
    #print(f"my_sigma: {my_sigma}")
    #logger.info(f"my_sigma: {my_sigma}")

    # if math.isnan(my_sigma) or my_sigma < 10:
    #     my_sigma = 10

    output = my_expectation / my_sigma

    return -output


def minimize_analytical(public_odd, real_probabilities, event, games_ids, df_probs_dict):
    
    # Set initial guess
    x0 = np.zeros(len(public_odd))
   
    args = (public_odd, real_probabilities, event, games_ids, df_probs_dict)
    
    res = minimize(fun=compute_objective_via_analytical,
                   x0=x0,
                   args=args,
                   tol=0.1,
                   #method='Powell',
                   #options={'maxiter': 3, 'disp': True, 'return_all': True}
                   )
    
    if res.success:
        return res.x
    else: 
        raise ValueError(res.message)