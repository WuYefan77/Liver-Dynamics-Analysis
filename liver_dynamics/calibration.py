"""Maximum-likelihood calibration and information criteria."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from .markov import AnalyticalFluxEngine


class ModelCalibrator:
    """Calibrate transition rates from longitudinal stage observations.

    Parameters
    ----------
    data:
        A pandas-like data frame containing ``start_stage``, ``end_stage``
        and ``dt`` columns.
    n_states:
        Number of ordered disease states.
    """

    required_columns = ("start_stage", "end_stage", "dt")

    def __init__(self, data, n_states: int = 5) -> None:
        missing = [column for column in self.required_columns if column not in data.columns]
        if missing:
            raise ValueError(f"data is missing required columns: {', '.join(missing)}")
        if len(data) == 0:
            raise ValueError("data must contain at least one transition")

        self.data = data.loc[:, self.required_columns].copy()
        self.engine = AnalyticalFluxEngine(n_states=n_states)
        self._validate_observations()

    def _validate_observations(self) -> None:
        start = np.asarray(self.data["start_stage"], dtype=float)
        end = np.asarray(self.data["end_stage"], dtype=float)
        intervals = np.asarray(self.data["dt"], dtype=float)

        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            raise ValueError("stage indices must be finite")
        if not np.all(start == np.floor(start)) or not np.all(end == np.floor(end)):
            raise ValueError("stage indices must be integers")
        if np.any(start < 0) or np.any(start >= self.engine.n_states):
            raise ValueError("start_stage contains an out-of-range state")
        if np.any(end < 0) or np.any(end >= self.engine.n_states):
            raise ValueError("end_stage contains an out-of-range state")
        if not np.all(np.isfinite(intervals)) or np.any(intervals < 0.0):
            raise ValueError("dt values must be finite and non-negative")

    def negative_log_likelihood(
        self,
        params: Sequence[float],
        model_type: str = "parsimonious",
    ) -> float:
        """Return the negative log-likelihood for the selected model."""
        if model_type != "parsimonious":
            raise ValueError("supported model_type: 'parsimonious'")

        rates = np.asarray(params, dtype=float)
        if rates.shape != (2,) or not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
            return float("inf")

        generator = self.engine.build_generator_matrix(rates[0], rates[1])
        nll = 0.0
        for row in self.data.itertuples(index=False):
            transition = self.engine.get_transition_probabilities(generator, row.dt)
            probability = transition[int(row.end_stage), int(row.start_stage)]
            if not np.isfinite(probability) or probability <= 0.0:
                return float("inf")
            nll -= float(np.log(probability))
        return nll

    def fit(
        self,
        initial_params: Sequence[float] = (0.1, 0.1),
        model_type: str = "parsimonious",
        method: str = "L-BFGS-B",
    ) -> OptimizeResult:
        """Estimate non-negative transition rates by maximum likelihood."""
        initial = np.asarray(initial_params, dtype=float)
        if initial.shape != (2,) or not np.all(np.isfinite(initial)) or np.any(initial < 0.0):
            raise ValueError("initial_params must contain two finite non-negative rates")

        result = minimize(
            self.negative_log_likelihood,
            x0=initial,
            args=(model_type,),
            method=method,
            bounds=((0.0, None), (0.0, None)),
        )
        return result

    @staticmethod
    def calculate_ic(nll: float, k: int, n: int) -> tuple[float, float]:
        """Compute Akaike and Bayesian information criteria."""
        if not np.isfinite(nll):
            raise ValueError("nll must be finite")
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        aic = 2.0 * k + 2.0 * nll
        bic = k * np.log(n) + 2.0 * nll
        return float(aic), float(bic)
