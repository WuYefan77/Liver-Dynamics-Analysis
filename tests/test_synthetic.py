"""Synthetic numerical tests: no clinical data."""
import numpy as np
import pandas as pd
from liver_dynamics import AnalyticalFluxEngine, DynamicFluxEngine, ModelCalibrator

def test_transition_matrix():
    engine = AnalyticalFluxEngine(3)
    q = engine.build_generator_matrix(0.2, 0.1)
    p = engine.transition_matrix(q, 2.0)
    assert np.all(p >= -1e-12)
    np.testing.assert_allclose(p.sum(axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(engine.transition_matrix(q, 0.0), np.eye(3))

def test_synthetic_likelihood():
    data = pd.DataFrame({"start_stage":[0,1,2,1], "end_stage":[1,1,1,2], "dt":[1.0,2.0,1.0,0.5]})
    model = ModelCalibrator(data, n_states=3)
    assert np.isfinite(model.negative_log_likelihood([0.2,0.1]))

def test_constant_rate_rk45():
    model = DynamicFluxEngine(0.2, 0.1, 0.0, n_states=3)
    y0 = np.array([1.0,0.0,0.0])
    sol = model.simulate((0.0,2.0), y0, [0.0,2.0], [0.0,0.0], t_eval=[2.0], rtol=1e-10, atol=1e-12)
    eng = AnalyticalFluxEngine(3)
    expected = eng.transition_matrix(eng.build_generator_matrix(0.2,0.1),2.0) @ y0
    np.testing.assert_allclose(sol.y[:,-1],expected,atol=1e-8)
