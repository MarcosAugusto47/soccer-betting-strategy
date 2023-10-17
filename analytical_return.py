import numpy as np
import pandas as pd

from scipy.optimize import minimize
from itertools import chain


def expectation(allocation, public_odd, real_probabilities):
    return sum(public_odd * allocation * real_probabilities)


def second_moment(allocation: list,
                  public_odd: np.array,
                  real_probabilities: np.array,
                  event: np.array,
                  df_prob: pd.DataFrame,
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
                # Get events intersections, both matrices are filled with 0's or 1's
                event_intersection_matrix = event_i * event_j
                # Get event insercetions mapped real probabilities
                event_intersection_matrix_prob = event_intersection_matrix * df_prob.to_numpy()
                # Compute the final events intersection probability
                prob_ij = sum(list(chain(*event_intersection_matrix_prob)))

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
    df_prob: pd.DataFrame
) -> np.float64:

    x = softmax(x)

    my_expectation = expectation(allocation=x,
                                 public_odd=public_odd,
                                 real_probabilities=real_probabilities)
    #print(f"my_expectation: {my_expectation}")

    my_second_moment = second_moment(allocation=x,
                                     public_odd=public_odd,
                                     real_probabilities=real_probabilities,
                                     event=event,
                                     df_prob=df_prob)

    my_sigma = np.sqrt(variance(my_second_moment, my_expectation))
    #print(f"my_sigma: {my_sigma}")

    # if math.isnan(my_sigma) or my_sigma < 10:
    #     my_sigma = 10

    output = my_expectation / my_sigma

    return -output


def minimize_analytical(public_odd, real_probabilities, event, df_prob):
    
    # Set initial guess
    x0 = np.zeros(len(public_odd))
   
    args = (public_odd, real_probabilities, event, df_prob)
    
    res = minimize(fun=compute_objective_via_analytical,
                   x0=x0,
                   args=args,
                   tol=0.001,
                   method='Powell',
                   options={'maxiter': 70, 'disp': True, 'return_all': True})
    
    if res.success:
        return res.x
    else: 
        raise ValueError(res.message)
