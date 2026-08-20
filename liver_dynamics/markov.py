"""Time-homogeneous continuous-time Markov dynamics."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


class AnalyticalFluxEngine:
    """Build a nearest-neighbour generator and evaluate its propagator.

    The package uses a column-vector convention: ``dF/dt = Q @ F``.
    Consequently, each column of ``Q`` sums to zero.
    """

    def __init__(self, n_states: int = 5) -> None:
        if not isinstance(n_states, (int, np.integer)) or n_states < 2:
            raise ValueError("n_states must be an integer greater than or equal to 2")
        self.n_states = int(n_states)

    @staticmethod
    def _validate_rate(value: float, name: str) -> float:
        rate = float(value)
        if not np.isfinite(rate) or rate < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return rate

    def build_generator_matrix(self, k_fwd: float, k_bck: float) -> np.ndarray:
        """Construct a mass-conserving nearest-neighbour generator matrix."""
        k_fwd = self._validate_rate(k_fwd, "k_fwd")
        k_bck = self._validate_rate(k_bck, "k_bck")

        generator = np.zeros((self.n_states, self.n_states), dtype=float)
        for state in range(self.n_states - 1):
            generator[state, state] -= k_fwd
            generator[state + 1, state] += k_fwd
            generator[state + 1, state + 1] -= k_bck
            generator[state, state + 1] += k_bck
        return generator

    def get_transition_probabilities(self, generator: np.ndarray, dt: float) -> np.ndarray:
        """Evaluate ``P(dt) = exp(Q * dt)`` for a time-homogeneous system."""
        matrix = np.asarray(generator, dtype=float)
        if matrix.shape != (self.n_states, self.n_states):
            raise ValueError(
                f"generator must have shape ({self.n_states}, {self.n_states})"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("generator must contain only finite values")
        if not np.allclose(matrix.sum(axis=0), 0.0, atol=1e-10):
            raise ValueError("generator columns must sum to zero")

        elapsed = float(dt)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("dt must be finite and non-negative")
        return expm(matrix * elapsed)

    transition_matrix = get_transition_probabilities
