"""
Module 1: Analytical Propagator via Matrix Exponential
Description: 
Solves the time-homogeneous linear ODE system dF/dt = Q * F(t) analytically.
By computing P(t) = exp(Q*t), we obtain the exact transition probability matrix 
without the cumulative errors associated with numerical stepping algorithms.

Author: Yefan Wu
"""
import numpy as np
from scipy.linalg import expm

class AnalyticalFluxEngine:
    def __init__(self, n_states=5):
        self.n_states = n_states

    def build_generator_matrix(self, k_fwd, k_bck):
        """
        Constructs a mass-conserving infinitesimal generator matrix Q.
        Each column sums to 0.
        """
        Q = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states - 1):
            Q[i, i] -= k_fwd
            Q[i+1, i] += k_fwd
            Q[i+1, i+1] -= k_bck
            Q[i, i+1] += k_bck
        return Q

    def get_transition_probabilities(self, Q, dt):
        """
        Computes the exact propagator matrix P(dt) = exp(Q * dt).
        """
        # scipy.linalg.expm uses the Padé approximation algorithm
        return expm(Q * dt)

if __name__ == "__main__":
    engine = AnalyticalFluxEngine()
    Q_demo = engine.build_generator_matrix(k_fwd=0.15, k_bck=0.05)
    P_demo = engine.get_transition_probabilities(Q_demo, dt=3.5)
    print("Exact Transition Matrix for dt=3.5 years:\n", np.round(P_demo, 3))
