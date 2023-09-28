import numpy as np
from scipy.optimize import minimize


def expectation(allocation, public_odd, real_probabilities):
    return sum(public_odd * allocation * real_probabilities)


def second_moment(allocation, public_odd, real_probabilities, scenario):
    
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
                is_scenario_equal = scenario[i] == scenario[j]
                
                if is_scenario_equal:
                    prob_ij = real_probabilities[i]

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
    scenario: np.ndarray
) -> np.float64:

    x = softmax(x)

    my_expectation = expectation(allocation=x,
                                 public_odd=public_odd,
                                 real_probabilities=real_probabilities)
    print(f"my_expectation: {my_expectation}")

    my_second_moment = second_moment(allocation=x,
                                     public_odd=public_odd,
                                     real_probabilities=real_probabilities,
                                     scenario=scenario)

    my_sigma = np.sqrt(variance(my_second_moment, my_expectation))
    print(f"my_sigma: {my_sigma}")

    # if math.isnan(my_sigma) or my_sigma < 10:
    #     my_sigma = 10
    print(f"my_sigma: {my_sigma}")

    output = my_expectation / my_sigma

    return -output


def minimize_analytical(public_odd, real_probabilities, scenario):
    
    # Set initial guess
    x0 = np.zeros(len(public_odd))
    
    args = (public_odd, real_probabilities, scenario)
    
    res = minimize(fun=compute_objective_via_analytical,
                   x0=x0,
                   args=args,
                   tol=0.00001,
                   options={'maxiter': 1000, 'disp': True, 'return_all': True})
    
    if res.success:
        return res.x
    else: 
        raise ValueError(res.message)