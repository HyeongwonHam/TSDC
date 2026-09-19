import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from tsdc.data import build_pair_features
from tsdc.factors import AVERAGE_TSDC_MASK, ENDPOINT_TSDC_MASK, TSDC, raw_doppler_factors, tsdc_mask_factors
from tsdc.features import WLS_MOTION_FEATURES, base_feature_names, build_pair_input_features, tsdc_feature_indices
from tsdc.fgo import block_tridiagonal_solve, make_direct_factor_terms
from tsdc.metrics import score_errors
from tsdc.model import TSDCNet

ROOT = Path(__file__).resolve().parents[1]
SIGNALS = ["BDS_B1I", "BDS_B1_I", "BDS_B2A_P", "GAL_E1", "GAL_E1_C_P", "GAL_E5A", "GAL_E5A_Q", "GLO_G1", "GLO_G1_CA",
           "GPS_L1", "GPS_L1_CA", "GPS_L5", "GPS_L5_Q", "QZS_J1", "QZS_J5", "QZS_L1_CA", "QZS_L5_Q"]
CODES = ["C", "I", "Q", "X"]


def pair_rows():
    common = dict(Svid=5, ConstellationType=1, SignalType="GPS_L1", CodeType="C", CarrierFrequencyHz=1575420000.0,
                  SvPositionYEcefMeters=-15e6, SvPositionZEcefMeters=15e6, SvAzimuthDegrees=90.0, SvElevationDegrees=45.0)
    prev = SimpleNamespace(utcTimeMillis=1000, PseudorangeRateMetersPerSecond=-100.0, AccumulatedDeltaRangeMeters=50.0,
                           SvPositionXEcefMeters=-10e6, **common)
    row = SimpleNamespace(utcTimeMillis=2000, PseudorangeRateMetersPerSecond=-101.0, AccumulatedDeltaRangeMeters=-50.4,
                          SvPositionXEcefMeters=-10e6 + 800.0, **common)
    rx = np.array([-2.7e6, -4.3e6, 3.85e6])
    return prev, row, rx, rx + np.array([10.0, 0.0, 0.0])


def test_tdcp_target():
    prev, row, rx0, rx1 = pair_rows()
    _, target, tdcp_rate = build_pair_features(prev, row, rx0, rx1, SIGNALS, CODES)
    assert tdcp_rate == (-50.4 - 50.0) / 1.0
    assert target == tdcp_rate - (-101.0)


def test_feature_dimension_and_order():
    names = base_feature_names(SIGNALS, CODES)
    assert len(names) == 77
    selected = [names[i] for i in tsdc_feature_indices(names)]
    assert len(selected) == 68
    assert not set(WLS_MOTION_FEATURES) & set(selected)
    assert selected[:5] == ["dt_s", "pr_rate_0_mps", "pr_rate_1_mps", "pr_rate_mean_mps", "pr_rate_delta_mps"]
    assert selected[-6:] == ["signal_unknown", "code_C", "code_I", "code_Q", "code_X", "code_unknown"]
    prev, row, rx0, rx1 = pair_rows()
    feats = build_pair_input_features(prev, row, rx0, rx1, SIGNALS, CODES)
    values = dict(zip(names, feats))
    assert len(feats) == 77
    assert values["dt_s"] == 1.0 and values["pr_rate_mean_mps"] == -100.5 and values["pr_rate_delta_mps"] == -1.0
    assert values["constellation_1"] == 1.0 and values["signal_GPS_L1"] == 1.0 and values["code_C"] == 1.0


def test_model_output_shape():
    model = TSDCNet(68).eval()
    assert sum(p.numel() for p in model.parameters()) == 69_794
    out = model(100.0 * torch.randn(16, 68))
    assert out.shape == (16, 2)
    assert out[:, 1].min() >= -5.0 and out[:, 1].max() <= 2.0


def contexts():
    base = dict(i0=0, i1=1, dt_s=1.0, pr_rate0=-100.0, pr_rate1=-101.0, los0=np.array([0.6, 0.0, 0.8]),
                los1=np.array([0.6, 0.0, 0.8]), nominal_delta=-100.2, sat_clock_delta=0.1, mu_mps=0.3, scale_mps=0.5)
    return [base, {**base, "mu_mps": 3.5}, {**base, "scale_mps": 1.2}]


def test_average_baseline_and_gates():
    factors = tsdc_mask_factors(contexts())
    assert len(factors[TSDC]) == 1  # |mu| > 3 m/s and b > 1 m/s are rejected
    assert np.isclose(factors[AVERAGE_TSDC_MASK][0].y, -100.5 + 0.1 + 100.2)
    assert np.isclose(factors[ENDPOINT_TSDC_MASK][0].y, -101.0 + 0.1 + 100.2)
    assert np.isclose(factors[TSDC][0].y, -101.0 + 0.3 + 0.1 + 100.2)
    assert factors[TSDC][0].sigma == max(1.0, 5.0 * 0.5 * 1.0)
    raw = [{**contexts()[0], "max_prr_uncertainty_mps": 0.1}]
    assert np.isclose(raw_doppler_factors(raw, "average", 1.0)[0].y, factors[AVERAGE_TSDC_MASK][0].y)


def test_direct_factor_algebra():
    c0, c1, y = make_direct_factor_terms(np.array([0.3, -0.4, 0.8]), np.array([0.31, -0.41, 0.79]), 12.5, 12.1, 0.02)
    assert np.array_equal(c0, [-0.3, 0.4, -1.0]) and np.array_equal(c1, [0.31, -0.41, 1.0])
    assert np.isclose(y, 12.5 + 0.02 - 12.1)
    rng = np.random.default_rng(0)
    n = 6
    upper = rng.normal(size=(n - 1, 3, 3))
    a = rng.normal(size=(n, 3, 3))
    diag = 10.0 * np.eye(3) + a @ a.transpose(0, 2, 1)
    dense = np.zeros((3 * n, 3 * n))
    for i in range(n):
        dense[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = diag[i]
    for i in range(n - 1):
        dense[3 * i : 3 * i + 3, 3 * i + 3 : 3 * i + 6] = upper[i]
        dense[3 * i + 3 : 3 * i + 6, 3 * i : 3 * i + 3] = upper[i].T
    rhs = rng.normal(size=(n, 3))
    assert np.allclose(block_tridiagonal_solve(diag, upper, rhs).ravel(), np.linalg.solve(dense, rhs.ravel()))


def test_score_function():
    p50, p95, score = score_errors(np.arange(1.0, 101.0))
    assert p50 == 50.5 and np.isclose(p95, 95.05) and score == 0.5 * (p50 + p95)


def test_leakage_group_mapping_load():
    mapping = pd.read_csv(ROOT / "splits/route_group_mapping.csv")
    splits = json.loads((ROOT / "splits/gsdc_5splits.json").read_text())
    assert {split: len(routes) for split, routes in splits.items()} == {
        "source_train": 32, "source_val": 10, "target_adapt": 8, "target_test": 91}
    assert sorted(mapping["route_group"]) == sorted(r for routes in splits.values() for r in routes)
    assert (mapping.groupby("route_leakage_group")["split"].nunique() == 1).all()
    groups = mapping.groupby("split")["route_leakage_group"].nunique().to_dict()
    assert groups == {"source_train": 27, "source_val": 4, "target_adapt": 7, "target_test": 58}
    manifest = pd.read_csv(ROOT / "splits/source_trace_manifest.csv")
    assert manifest.groupby("split").size().to_dict() == {"source_train": 77, "source_val": 20}
