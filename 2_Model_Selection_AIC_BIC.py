"""
Module 2: Maximum Likelihood Estimation & Model Selection
Description: 
Calibrates the generator matrix Q using longitudinal observational data. 
Implements Akaike (AIC) and Bayesian (BIC) Information Criteria to penalize 
model complexity, distinguishing true biological signals from measurement noise.

Author: Yefan Wu
"""
import numpy as np
from scipy.optimize import minimize
from 1_Matrix_Exponential_Engine import AnalyticalFluxEngine # Import Module 1

class ModelCalibrator:
    def __init__(self, data):
        self.data = data # Expected pandas DataFrame: [start_stage, end_stage, dt]
        self.engine = AnalyticalFluxEngine()

    def negative_log_likelihood(self, params, model_type='parsimonious'):
        # Impose physical bounds (rates >= 0)
        if any(p < 0 for p in params): return 1e10
        
        # Build Q based on model complexity
        if model_type == 'parsimonious':
            Q = self.engine.build_generator_matrix(params[0], params[1])
        # elif model_type == 'saturated': ... handle complex Q matrix
        
        nll = 0.0
        for _, row in self.data.iterrows():
            P = self.engine.get_transition_probabilities(Q, row['dt'])
            # Extract the likelihood of the observed transition
            prob = max(P[int(row['end_stage']), int(row['start_stage'])], 1e-12)
            nll -= np.log(prob)
        return nll

    @staticmethod
    def calculate_ic(nll, k, n):
        """
        Computes Information Criteria to prevent overfitting.
        k: number of parameters, n: sample size.
        """
        aic = 2 * k + 2 * nll
        bic = k * np.log(n) + 2 * nll
        return aic, bic
