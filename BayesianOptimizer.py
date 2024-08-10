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

from botorch.models.transforms.input import Normalize


class BaseBoTorchOptimizer:
    
    def __init__(self, n_iterations, public_odd, real_probabilities, event, games_ids, df_probs_dict):
        self.n_iterations = n_iterations
        self.public_odd = public_odd
        self.real_probabilities = real_probabilities
        self.event = event
        self.games_ids = games_ids
        self.df_probs_dict = df_probs_dict
        self.n = len(public_odd)

    def expectation(self, allocation, public_odd, real_probabilities):
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
        term1 = torch.sum((public_odd * allocation) ** 2 * real_probabilities, dim=1)
        term2_list = []

        n = len(public_odd)

        for k in allocation:
            term2_sublist = []  # n x n size
            for i in range(n):
                for j in range(n):
                    if i != j:
                        theta_i = public_odd[i] * k[i]
                        theta_j = public_odd[j] * k[j]
                        theta_ij = theta_i * theta_j

                        prob_ij = 0
                        event_i = np.transpose(np.array(event[i]).reshape(7, 7))
                        event_j = np.transpose(np.array(event[j]).reshape(7, 7))

                        if games_ids[i] == games_ids[j]:
                            event_intersection_matrix = event_i * event_j
                            event_intersection_matrix_prob = event_intersection_matrix * df_probs_dict[games_ids[i]].to_numpy()
                            prob_ij = sum(list(chain(*event_intersection_matrix_prob)))
                        else:
                            event_i_prob = event_i * df_probs_dict[games_ids[i]].to_numpy()
                            event_j_prob = event_j * df_probs_dict[games_ids[j]].to_numpy()
                            prob_i = sum(list(chain(*event_i_prob)))
                            prob_j = sum(list(chain(*event_j_prob)))
                            prob_ij = prob_i * prob_j

                        term2_sublist.append(theta_ij * prob_ij)
            term2_list.append(torch.stack(term2_sublist))

        term2_list = torch.stack(term2_list)
        term2 = torch.sum(term2_list, dim=1)
        
        return term1 + term2

    def variance(self, second_moment, expectation):
        return second_moment - (expectation) ** 2

    def compute_objective_via_analytical(
        self,
        x: np.ndarray,
        public_odd: np.ndarray,
        real_probabilities: np.ndarray,
        event: np.ndarray,
        games_ids: np.ndarray,
        df_probs_dict: Dict[str, pd.DataFrame],
    ) -> np.float64:
        raise NotImplementedError("This method should be implemented by subclasses.")

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
        raise NotImplementedError("This method should be implemented by subclasses.")


class BoTorchOptimizer(BaseBoTorchOptimizer):
    
    def __init__(self, n_iterations, public_odd, real_probabilities, event, games_ids, df_probs_dict):
        super().__init__(n_iterations, public_odd, real_probabilities, event, games_ids, df_probs_dict)

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
        my_second_moment = self.second_moment(allocation=x,
                                              public_odd=public_odd,
                                              real_probabilities=real_probabilities,
                                              event=event,
                                              games_ids=games_ids,
                                              df_probs_dict=df_probs_dict)
        my_sigma = np.sqrt(self.variance(my_second_moment, my_expectation))
        output = my_expectation / my_sigma

        return output

    def run_optimization(self):
        train_X = draw_sobol_samples(
            bounds=torch.tensor([[0.001] * self.n, [0.1] * self.n]),
            n=1,
            q=5,
            seed=47,
        ).squeeze(0).double()  # 5 initial points
        train_Y = self.objective_function(train_X)

        best_value = train_Y.max()
        best_candidate = train_X[train_Y.argmax()]

        for iteration in range(self.n_iterations):
            train_Y_standardized = standardize(train_Y)

            gp_model = SingleTaskGP(train_X, train_Y_standardized, input_transform=Normalize(d=self.n))
            mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
            fit_gpytorch_model(mll)
            
            acq_func = ExpectedImprovement(model=gp_model, best_f=train_Y_standardized.max(), maximize=True)
            
            candidate, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.tensor([[0.] * self.n, [10.] * self.n]),
                q=1,
                num_restarts=10,
                raw_samples=512,
            )
            
            new_y = self.objective_function(candidate)
            
            train_X = torch.cat([train_X, candidate])
            train_Y = torch.cat([train_Y, new_y])

            if new_y > best_value:
                best_value = new_y
                best_candidate = candidate

        return best_candidate.numpy().ravel()


class BoTorchOptimizerVariableStake(BaseBoTorchOptimizer):
    
    def __init__(self, n_iterations, public_odd, real_probabilities, event, games_ids, df_probs_dict):
        super().__init__(n_iterations, public_odd, real_probabilities, event, games_ids, df_probs_dict)

    def compute_objective_via_analytical(
        self,
        x: np.ndarray,
        public_odd: np.ndarray,
        real_probabilities: np.ndarray,
        event: np.ndarray,
        games_ids: np.ndarray,
        df_probs_dict: Dict[str, pd.DataFrame],
    ) -> np.float64:
        gamma = x[:, 0]
        x = x[:, 1:]
        x = F.softmax(x, dim=-1)

        my_expectation = self.expectation(allocation=x,
                                          public_odd=public_odd,
                                          real_probabilities=real_probabilities)
        my_second_moment = self.second_moment(allocation=x,
                                              public_odd=public_odd,
                                              real_probabilities=real_probabilities,
                                              event=event,
                                              games_ids=games_ids,
                                              df_probs_dict=df_probs_dict)
        my_sigma = np.sqrt(self.variance(my_second_moment, my_expectation))
        output = (1 - gamma + gamma * my_expectation) / (gamma * my_sigma)

        return output

    def run_optimization(self):
        train_X = draw_sobol_samples(
            bounds=torch.tensor([[0.001] * self.n, [0.1] * self.n]),
            n=1,
            q=5,
            seed=47,
        ).squeeze(0).double()
        to_prepend = torch.full((train_X.size(0), 1), 0.4, dtype=torch.float64)
        train_X = torch.cat((to_prepend, train_X), dim=1)
        train_Y = self.objective_function(train_X)

        best_value = train_Y.max()
        best_candidate = train_X[train_Y.argmax()]

        for iteration in range(self.n_iterations):
            train_Y_standardized = standardize(train_Y)

            gp_model = SingleTaskGP(train_X, train_Y_standardized, input_transform=Normalize(d=self.n + 1))
            mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
            fit_gpytorch_model(mll)
            
            acq_func = ExpectedImprovement(model=gp_model, best_f=train_Y_standardized.max(), maximize=True)
            
            candidate, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.tensor([[0.05] + [0.] * self.n, [0.5] + [10.] * self.n]),
                q=1,
                num_restarts=10,
                raw_samples=512,
            )
            
            new_y = self.objective_function(candidate)
            
            train_X = torch.cat([train_X, candidate])
            train_Y = torch.cat([train_Y, new_y])

            if new_y > best_value:
                best_value = new_y
                best_candidate = candidate

        return best_candidate.numpy().ravel()
