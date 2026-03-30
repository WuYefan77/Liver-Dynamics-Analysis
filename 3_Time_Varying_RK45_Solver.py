"""
Time-Inhomogeneous Compartmental ODE Solver (via RK45)
Author: Yefan Wu
Date: March 2026

Description:
This module solves a non-autonomous linear ODE system (dF/dt = Q(t) * F(t)) 
where the transition rate matrix Q(t) is time-dependent. 
It utilizes scipy's solve_ivp (RK45) to handle matrices that do not commute 
at different time steps ([Q(t1), Q(t2)] != 0), a limitation of the standard Matrix Exponential.

Use Case: 
Simulating the probability flux of disease progression (or credit rating migrations)
under dynamic, continuous covariates (e.g., patient weight trajectories).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

class DynamicFluxEngine:
    def __init__(self, k_fwd_base, k_bck_base, gamma_weight):
        """
        Initialize the baseline parameters and the covariate multiplier.
        gamma_weight: The exponential hazard multiplier for the weight covariate.
        """
        self.k_fwd_base = k_fwd_base
        self.k_bck_base = k_bck_base
        self.gamma = gamma_weight
        self.n_states = 5 # F0 to F4

    def _build_Q_t(self, t, weight_func):
        """
        Constructs the time-dependent Infinitesimal Generator Matrix Q(t).
        """
        # 1. Get current weight from the interpolated trajectory
        current_weight = weight_func(t)
        
        # 2. Scale the forward progression rate dynamically
        # e.g., higher weight exponentially accelerates progression
        k_fwd_t = self.k_fwd_base * np.exp(self.gamma * current_weight)
        k_bck_t = self.k_bck_base # Assuming regression is constant for parsimony
        
        # 3. Build the 5x5 rate matrix (Mass-conserving)
        Q = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states - 1):
            Q[i, i] -= k_fwd_t
            Q[i+1, i] += k_fwd_t
            Q[i+1, i+1] -= k_bck_t
            Q[i, i+1] += k_bck_t
        return Q

    def _ode_system(self, t, F, weight_func):
        """
        The differential equation dF/dt = Q(t) * F(t) for the RK45 solver.
        """
        Q_t = self._build_Q_t(t, weight_func)
        return Q_t @ F # Matrix-vector multiplication

    def simulate(self, t_span, y0, weight_times, weight_values, t_eval=None):
        """
        Solves the time-inhomogeneous system for a given weight trajectory.
        """
        # Interpolate discrete weight measurements into a continuous function W(t)
        weight_func = interp1d(weight_times, weight_values, 
                               kind='linear', fill_value="extrapolate")
        
        # Solve the Initial Value Problem using Runge-Kutta 4(5)
        sol = solve_ivp(
            fun=lambda t, F: self._ode_system(t, F, weight_func),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6, atol=1e-8 # Tight tolerances for probability conservation
        )
        return sol

# ==========================================
# GitHub Portfolio Demonstration
# ==========================================
if __name__ == "__main__":
    # Setup the engine (Toy parameters)
    # gamma = 0.05 means every 1 unit of weight increases risk by exp(0.05) ~ 5%
    engine = DynamicFluxEngine(k_fwd_base=0.05, k_bck_base=0.08, gamma_weight=0.05)
    
    t_span = (0, 20)
    t_eval = np.linspace(0, 20, 200)
    y0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0]) # Start 100% healthy
    
    # --- Scenario A: Patient remains overweight (Constant 20 units above baseline) ---
    times_A = [0, 20]
    weights_A = [20, 20] 
    sol_A = engine.simulate(t_span, y0, times_A, weights_A, t_eval)
    
    # --- Scenario B: Lifestyle Intervention (Loses weight at Year 5) ---
    times_B = [0, 5, 8, 20]
    weights_B = [20, 20, 0, 0] # Drops weight between year 5 and 8, then maintains
    sol_B = engine.simulate(t_span, y0, times_B, weights_B, t_eval)
    
    # --- Visualization (The "Money" Plot) ---
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Plotting only the probability of reaching Cirrhosis (State F4) to show the policy impact
    plt.plot(sol_A.t, sol_A.y[4, :], color='#d62728', lw=3, label='Scenario A: Constant High Weight')
    plt.plot(sol_B.t, sol_B.y[4, :], color='#2ca02c', lw=3, linestyle='--', label='Scenario B: Weight Loss at Year 5')
    
    # Highlight the intervention window
    plt.axvspan(5, 8, color='grey', alpha=0.1, label='Lifestyle Intervention Window')
    
    plt.title("Counterfactual Simulation via Time-Inhomogeneous ODEs", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Years from Baseline", fontsize=12)
    plt.ylabel("Probability of reaching Stage F4 (Cirrhosis)", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # plt.savefig("time_varying_intervention.png")
    plt.show()
