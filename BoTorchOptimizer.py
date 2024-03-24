import numpy as np
import pandas as pd
from itertools import chain
from typing import Dict
import torch
import torch.nn.functional as F
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement
from botorch.utils import standardize, draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize

from botorch.models.transforms.input import Normalize


class BoTorchOptimizer:
    
    def __init__(self, public_odd, real_probabilities, event, games_ids, df_probs_dict):
        self.n_iterations = 50
        self.public_odd = public_odd
        self.real_probabilities = real_probabilities
        self.event = event
        self.games_ids = games_ids
        self.df_probs_dict = df_probs_dict
        self.n = len(public_odd)

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


    def objective_function(self, X):
   
        output = self.compute_objective_via_analytical(
            x=X,
            public_odd=self.public_odd,
            real_probabilities=self.real_probabilities,
            event=self.event,
            games_ids=self.games_ids,
            df_probs_dict=self.df_probs_dict
        )

        return output.unsqueeze(1)        
    
    def run_optimization(self):
        
        train_X = draw_sobol_samples(bounds=torch.tensor([[0.001]*self.n, [0.1]*self.n]), n=1, q=5).squeeze(0).double()  # 5 initial points
        train_Y = self.objective_function(train_X)

        # Initialize best observed point and value
        best_value = train_Y.max()
        best_candidate = train_X[train_Y.argmax()]

        for iteration in range(self.n_iterations):
            # Our standard models assume that the input data is normalized (train
            # inputs in the unit cube, train targets with zero mean and unit variance).
            # While we do indicate this in the tutorials/docstrings, a good amount of
            # the issues we see reported is b/c users do not standardize the data.
            # by Balandat (contributor from Meta)

            # Standardize the outputs
            train_Y_standardized = standardize(train_Y)

            # Define and fit the GP model
            gp_model = SingleTaskGP(train_X, train_Y_standardized, input_transform=Normalize(d=self.n))
            mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
            fit_gpytorch_model(mll)
            
            # Define the acquisition function
            acq_func = ExpectedImprovement(model=gp_model, best_f=train_Y_standardized.max(), maximize=True)
            
            # Optimize the acquisition function to find new candidate
            candidate, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.tensor([[-5.]*self.n, [5.]*self.n]), # bounds that best behave when mapping via softmax in the end
                q=1,  # Number of points to generate
                num_restarts=10,  # Number of restarts in optimization
                raw_samples=512,  # Number of samples for initialization
            )
            
            # Evaluate the objective function at the new candidate
            new_y = self.objective_function(candidate)
            
            # Update training data
            train_X = torch.cat([train_X, candidate])
            train_Y = torch.cat([train_Y, new_y])

            #print(f"Iteration {iteration+1}/{self.n_iterations}, new point: {candidate.numpy()}, objective: {new_y.item()}")

            # Update the best observed value and candidate if the new candidate is better
            if new_y > best_value:
                best_value = new_y
                best_candidate = candidate

        return best_candidate.numpy().ravel()
