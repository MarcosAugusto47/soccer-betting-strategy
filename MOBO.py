import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from itertools import chain
from typing import Dict
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
from gpytorch.constraints import GreaterThan
from botorch.utils.multi_objective.pareto import is_non_dominated

from botorch.models.transforms.input import Normalize

# Ensure all tensors are created with double precision
torch.set_default_dtype(torch.float64)

class MOBO:

    def __init__(self, public_odd, real_probabilities, event, games_ids, df_probs_dict):
        self.n_iterations = 50
        self.public_odd = public_odd
        self.real_probabilities = real_probabilities
        self.event = event
        self.games_ids = games_ids
        self.df_probs_dict = df_probs_dict
        self.n = len(public_odd)
        self.bounds = torch.tensor([[0.001]*self.n, [0.1]*self.n])


    def expectation(self, allocation, public_odd, real_probabilities):
    
        # Perform element-wise multiplication and sum across the second dimension (columns)
        result = torch.sum(allocation * public_odd * real_probabilities, dim=1)
        
        return result

    def second_moment(
            self,
            allocation: torch.Tensor,
            public_odd: torch.Tensor,
            real_probabilities: torch.Tensor,
            event: np.array,
            games_ids: np.array,
            df_probs_dict: Dict[str, pd.DataFrame],
    ) -> torch.Tensor:
        """
        Computes the second moment of the random variable.

        Note:
            Check mathematical definition solved by hand at docs/return-analytical-definition.png

        Args:
            allocation: allocation of percentages of bet distribution, should be n1 X n where n1 > 1
            public_odd: tensor of odds from Sports books
            real_probabilities: tensor of real probabilities of bet opportunities
            event: array of 7x7 lenght arrays that maps the scenarios
            df_prob: 7x7 pandas dataframe of real probablities, one for each scenario
        
        Returns:
            Tensor: final computation with nxn length
        """
        
        term1 = torch.sum((public_odd * allocation)**2 * real_probabilities, dim=1)
        #term1 = torch.sum((public_odd * allocation)**2 * real_probabilities)
        #term1 = sum((public_odd * allocation)**2 * real_probabilities)

        term2_list = []

        n = len(public_odd)

        for k in allocation:
            
            term2_sublist = [] # n x n size

            for i in range(n):
                for j in range(n):
                    if i != j:
                        
                        theta_i = public_odd[i] * k[i]
                        theta_j = public_odd[j] * k[j]
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

                        term2_sublist.append(theta_ij*prob_ij)

            term2_list.append(torch.stack(term2_sublist))

        term2_list = torch.stack(term2_list)
        term2 = torch.sum(term2_list, dim=1)
        
        return term1 + term2

    def variance(self, second_moment, expectation):
        return second_moment - (expectation)**2

    def compute_objective_via_analytical(
        self,
        x: np.ndarray,
        public_odd: np.ndarray,
        real_probabilities: np.ndarray,
        event: np.ndarray,
        games_ids: np.ndarray,
        df_probs_dict: Dict[str, pd.DataFrame],
    ) -> np.float64:

        x = F.softmax(x, dim=-1)

        my_expectation = self.expectation(allocation=x,
                                    public_odd=public_odd,
                                    real_probabilities=real_probabilities)
        #print(f"my_expectation: {my_expectation}")
        #logger.info(f"my_expectation: {my_expectation}")

        my_second_moment = self.second_moment(allocation=x,
                                        public_odd=public_odd,
                                        real_probabilities=real_probabilities,
                                        event=event,
                                        games_ids=games_ids,
                                        df_probs_dict=df_probs_dict)

        my_sigma = np.sqrt(self.variance(my_second_moment, my_expectation))
        #print(f"my_sigma: {my_sigma}")
        #logger.info(f"my_sigma: {my_sigma}")

        # if math.isnan(my_sigma) or my_sigma < 10:
        #     my_sigma = 10

        output = my_expectation / my_sigma

        return output


    def objective_1(self, X):
   
        output = self.compute_objective_via_analytical(
            x=X,
            public_odd=self.public_odd,
            real_probabilities=self.real_probabilities,
            event=self.event,
            games_ids=self.games_ids,
            df_probs_dict=self.df_probs_dict
        )

        return output
    
    def compute_hhi(self, x):
        x = F.softmax(x, dim=-1)
        output = torch.sum(np.square(x), dim=1)
        return -output
    
    def objective_2(self, X):
        output = self.compute_hhi(x=X)
        return output
    
    def generate_initial_data(self, n=1):
        # Sobol samples are a good choice for initial sampling in a bounded domain
        X_init = draw_sobol_samples(bounds=self.bounds, n=n, q=1).squeeze(1)
        Y_init = torch.stack([self.objective_1(X_init), self.objective_2(X_init)], dim=-1)
        return X_init, Y_init
        # Train GP model
    
    def train_gp_model(self, X, Y):
        # Assuming Y is standardized
        model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=Y.shape[-1]), input_transform=Normalize(d=self.n))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        return model
    
    # Sequential optimization
    def optimize_acqf_and_get_observation(self, acq_func):
        """Optimizes the acquisition function, and returns a new observation."""
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.tensor([[0.]*self.n, [5.]*self.n]),
            q=1,
            num_restarts=10,
            raw_samples=512,  # Increase for better results
        )
        new_x = candidates.clone().detach()
        new_obj = torch.stack([self.objective_1(new_x), self.objective_2(new_x)], dim=-1)
        return new_x, new_obj
    
    def run_optimization(self):
        # Initialize data
        X, Y = self.generate_initial_data(n=1)

        # Main loop for Bayesian optimization
        for i in range(self.n_iterations):  # Number of iterations
            model = self.train_gp_model(X, Y)
            partitioning = NondominatedPartitioning(ref_point=torch.tensor([-5., -5.]), Y=Y)
            acq_func = ExpectedHypervolumeImprovement(model=model, partitioning=partitioning, ref_point=[-5., -5.])
            new_x, new_obj = self.optimize_acqf_and_get_observation(acq_func)
            X = torch.cat([X, new_x])
            Y = torch.cat([Y, new_obj])
        
         # Extract the Pareto front
        X_np = X.numpy()
        Y_np = Y.numpy()
        pareto_mask = is_non_dominated(Y)
        pareto_front = Y_np[pareto_mask]
        pareto_candidate = X_np[pareto_mask]
        
        return Y_np, pareto_front, pareto_candidate

# # Objective functions
# def objective_1(x):
#     return -(x[:, 0] - 2)**2 - (x[:, 1] + 1)**2

# def objective_2(x):
#     return (x[:, 0] + 1)**2 + (x[:, 1] - 3)**2

## Define the search space
#bounds = torch.tensor([[-5.0, -5.0], [5.0, 5.0]])

# Generate initial data
# def generate_initial_data(n=10):
#     # Sobol samples are a good choice for initial sampling in a bounded domain
#     X_init = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(1)
#     Y_init = torch.stack([objective_1(X_init), objective_2(X_init)], dim=-1)
#     return X_init, Y_init

# # Train GP model
# def train_gp_model(X, Y):
#     # Assuming Y is standardized
#     model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=Y.shape[-1]), input_transform=Normalize(d=2))
#     mll = ExactMarginalLogLikelihood(model.likelihood, model)
#     fit_gpytorch_model(mll)
#     return model

# Sequential optimization
# def optimize_acqf_and_get_observation(acq_func):
#     """Optimizes the acquisition function, and returns a new observation."""
#     candidates, _ = optimize_acqf(
#         acq_function=acq_func,
#         bounds=bounds,
#         q=1,
#         num_restarts=5,
#         raw_samples=20,  # Increase for better results
#     )
#     new_x = candidates.clone().detach()
#     new_obj = torch.stack([objective_1(new_x), objective_2(new_x)], dim=-1)
#     return new_x, new_obj

# # Initialize data
# X, Y = generate_initial_data(n=10)

# # Main loop for Bayesian optimization
# for _ in range(20):  # Number of iterations
#     model = train_gp_model(X, Y)
#     partitioning = NondominatedPartitioning(ref_point=torch.tensor([0.0, 0.0]), Y=Y)
#     acq_func = ExpectedHypervolumeImprovement(model=model, partitioning=partitioning, ref_point=[0.0, 0.0])
#     new_x, new_obj = optimize_acqf_and_get_observation(acq_func)
#     X = torch.cat([X, new_x])
#     Y = torch.cat([Y, new_obj])
#     print(new_x, new_obj)
