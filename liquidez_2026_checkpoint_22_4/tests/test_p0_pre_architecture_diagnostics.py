from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from p0_diagnostics import auxiliary_head_metrics, compare_residual_runs, diagnose_patience, interval_calibration_by_horizon

def test_heads():
    m = auxiliary_head_metrics(event_target=np.array([0,1]), event_probability=np.array([.1,.9]),
        magnitude_target=np.array([0.,1.]), magnitude_prediction=np.array([0.,1.5]),
        direction_target=np.array([0,1,2]), direction_prediction=np.array([0,1,2]))
    assert m["event"]["f1"] == 1 and m["direction"]["macro_f1"] == 1

def test_intervals_by_horizon():
    f = pd.DataFrame({"tipo_serie":["saldo","saldo"],"horizon_step":[1,2],"actual_orig":[1.,5.],
        "lower_ci":[0.,0.],"upper_ci":[2.,4.],"isTrain":[False,False]})
    out = interval_calibration_by_horizon(f)
    assert list(out.empirical_coverage) == [1.,0.]

def test_residual_and_patience():
    a = compare_residual_runs({"m":.8},{"m":1.},metric="m")
    assert a["relative_improvement"] == pytest.approx(.2)
    d = diagnose_patience([SimpleNamespace(validation_objective=x) for x in [2.,1.8,1.9,1.7]], configured_patience=5)
    assert d.best_epoch == 4 and d.recommended_patience >= 5
