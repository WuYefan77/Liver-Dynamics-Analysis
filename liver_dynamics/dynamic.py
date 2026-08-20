"""Time-inhomogeneous compartmental dynamics integrated with RK45."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .markov import AnalyticalFluxEngine


class DynamicFluxEngine:
    """Integrate a generator whose forward rate depends on a covariate path."""

    def __init__(
        self,
        k_fwd_base: float,
        k_bck_base: float,
        gamma_weight: float,
        n_states: int = 5,
    ) -> None:
        validator = AnalyticalFluxEngine(n_states=n_states)
        self.k_fwd_base = validator._validate_rate(k_fwd_base, "k_fwd_base")
        self.k_bck_base = validator._validate_rate(k_bck_base, "k_bck_base")
        self.gamma = float(gamma_weight)
        if not np.isfinite(self.gamma):
            raise ValueError("gamma_weight must be finite")
        self.n_states = validator.n_states

    def _build_q_t(self, t: float, weight_func) -> np.ndarray:
        current_weight = float(weight_func(t))
        k_fwd_t = self.k_fwd_base * np.exp(self.gamma * current_weight)
        if not np.isfinite(k_fwd_t):
            raise FloatingPointError("time-varying forward rate is not finite")

        generator = np.zeros((self.n_states, self.n_states), dtype=float)
        for state in range(self.n_states - 1):
            generator[state, state] -= k_fwd_t
            generator[state + 1, state] += k_fwd_t
            generator[state + 1, state + 1] -= self.k_bck_base
            generator[state, state + 1] += self.k_bck_base
        return generator

    def _ode_system(self, t: float, state: np.ndarray, weight_func) -> np.ndarray:
        return self._build_q_t(t, weight_func) @ state

    def simulate(
        self,
        t_span: tuple[float, float],
        y0: Sequence[float],
        weight_times: Sequence[float],
        weight_values: Sequence[float],
        t_eval: Sequence[float] | None = None,
        *,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> Any:
        """Solve the non-autonomous compartmental system."""
        start, stop = map(float, t_span)
        if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
            raise ValueError("t_span must contain two finite, increasing times")

        initial = np.asarray(y0, dtype=float)
        if initial.shape != (self.n_states,) or not np.all(np.isfinite(initial)):
            raise ValueError(f"y0 must contain {self.n_states} finite probabilities")
        if np.any(initial < 0.0) or not np.isclose(initial.sum(), 1.0, atol=1e-8):
            raise ValueError("y0 must be non-negative and sum to one")

        times = np.asarray(weight_times, dtype=float)
        values = np.asarray(weight_values, dtype=float)
        if times.ndim != 1 or values.ndim != 1 or len(times) != len(values) or len(times) < 2:
            raise ValueError("weight_times and weight_values must be equal-length vectors")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
            raise ValueError("weight trajectory must contain only finite values")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("weight_times must be strictly increasing")
        if times[0] > start or times[-1] < stop:
            raise ValueError("weight trajectory must cover t_span")

        evaluation_times = None if t_eval is None else np.asarray(t_eval, dtype=float)
        if evaluation_times is not None:
            if evaluation_times.ndim != 1 or not np.all(np.isfinite(evaluation_times)):
                raise ValueError("t_eval must be a vector of finite values")
            if np.any(np.diff(evaluation_times) <= 0.0):
                raise ValueError("t_eval must be strictly increasing")
            if len(evaluation_times) and (
                evaluation_times[0] < start or evaluation_times[-1] > stop
            ):
                raise ValueError("t_eval values must lie within t_span")

        weight_func = interp1d(times, values, kind="linear", bounds_error=True)
        solution = solve_ivp(
            fun=lambda t, state: self._ode_system(t, state, weight_func),
            t_span=(start, stop),
            y0=initial,
            t_eval=evaluation_times,
            method="RK45",
            rtol=rtol,
            atol=atol,
        )
        if not solution.success:
            raise RuntimeError(f"RK45 integration failed: {solution.message}")
        return solution
