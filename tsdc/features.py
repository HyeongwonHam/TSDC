from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


C_MPS_PER_NSPS = 0.299792458

# The pair cache stores the 77 base columns below. TSDC uses the 68 columns left
# after removing these nine features derived from adjacent-epoch WLS receiver
# motion (Online Resource 1, Table S1 and Section S1.3).
WLS_MOTION_FEATURES = [
    "los_delta_x",
    "los_delta_y",
    "los_delta_z",
    "geom_rate_mps",
    "geom_minus_pr1_mps",
    "wls_dx_m",
    "wls_dy_m",
    "wls_dz_m",
    "wls_speed_mps",
]


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    if not math.isfinite(x):
        return default
    return x


def normalize_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    s = str(value)
    if s == "nan":
        return ""
    return s


def one_hot(value: str, categories: list[str]) -> list[float]:
    out = [0.0] * (len(categories) + 1)
    try:
        idx = categories.index(value)
    except ValueError:
        idx = len(categories)
    out[idx] = 1.0
    return out


def one_hot_constellation(value: Any) -> list[float]:
    out = [0.0] * 8
    try:
        idx = int(float(value))
    except Exception:
        idx = 0
    if 1 <= idx <= 7:
        out[idx - 1] = 1.0
    else:
        out[-1] = 1.0
    return out


def collect_categories(trace_dirs: list[Path], max_categories: int) -> tuple[list[str], list[str]]:
    signal_values: set[str] = set()
    code_values: set[str] = set()
    for trace_dir in trace_dirs:
        cols = pd.read_csv(trace_dir / "device_gnss.csv", nrows=0).columns
        usecols = [c for c in ["SignalType", "CodeType"] if c in cols]
        if not usecols:
            continue
        df = pd.read_csv(trace_dir / "device_gnss.csv", usecols=usecols, low_memory=False)
        if "SignalType" in df.columns:
            signal_values.update(normalize_str(x) for x in df["SignalType"].dropna().unique())
        if "CodeType" in df.columns:
            code_values.update(normalize_str(x) for x in df["CodeType"].dropna().unique())
    signals = sorted(x for x in signal_values if x)[:max_categories]
    codes = sorted(x for x in code_values if x)[:max_categories]
    return signals, codes


def base_feature_names(signal_types: list[str], code_types: list[str]) -> list[str]:
    names = [
        "dt_s",
        "pr_rate_0_mps",
        "pr_rate_1_mps",
        "pr_rate_mean_mps",
        "pr_rate_delta_mps",
        "pr_unc_0_log1p",
        "pr_unc_1_log1p",
        "cn0_0_dbhz",
        "cn0_1_dbhz",
        "cn0_delta_dbhz",
        "elev_0_deg",
        "elev_1_deg",
        "sin_az_0",
        "cos_az_0",
        "sin_az_1",
        "cos_az_1",
        "los0_x",
        "los0_y",
        "los0_z",
        "los1_x",
        "los1_y",
        "los1_z",
        "los_delta_x",
        "los_delta_y",
        "los_delta_z",
        "geom_rate_mps",
        "geom_minus_pr1_mps",
        "wls_dx_m",
        "wls_dy_m",
        "wls_dz_m",
        "wls_speed_mps",
        "sv_vel_x_mps",
        "sv_vel_y_mps",
        "sv_vel_z_mps",
        "sv_clock_drift_mps",
        "sv_clock_bias_m",
        "raw_pr_unc_log1p",
        "iono_m",
        "tropo_m",
        "isrb_m",
        "rx_clock_drift_mps",
        "rx_bias_unc_m",
        "agc_db",
        "snr_db",
        "carrier_freq_ghz",
        "multipath_indicator",
    ]
    names.extend([f"constellation_{i}" for i in range(1, 8)])
    names.append("constellation_unknown")
    names.extend([f"signal_{x}" for x in signal_types])
    names.append("signal_unknown")
    names.extend([f"code_{x}" for x in code_types])
    names.append("code_unknown")
    return names


def tsdc_feature_indices(base_names: list[str]) -> list[int]:
    return [index for index, name in enumerate(base_names) if name not in WLS_MOTION_FEATURES]


def get_col(row: Any, col: str, default: float = 0.0) -> float:
    return finite_float(getattr(row, col, default), default)


def get_str_col(row: Any, col: str) -> str:
    return normalize_str(getattr(row, col, ""))


# Inputs of one adjacent-epoch pair. No ADR or carrier-phase column is read here.
def build_pair_input_features(
    prev: Any,
    row: Any,
    rx0: np.ndarray,
    rx1: np.ndarray,
    signal_types: list[str],
    code_types: list[str],
) -> list[float] | None:
    t0 = int(get_col(prev, "utcTimeMillis"))
    t1 = int(get_col(row, "utcTimeMillis"))
    dt_s = (t1 - t0) / 1000.0
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        return None

    pr0 = get_col(prev, "PseudorangeRateMetersPerSecond")
    pr1 = get_col(row, "PseudorangeRateMetersPerSecond")

    sv0 = np.array(
        [
            get_col(prev, "SvPositionXEcefMeters", np.nan),
            get_col(prev, "SvPositionYEcefMeters", np.nan),
            get_col(prev, "SvPositionZEcefMeters", np.nan),
        ],
        dtype=float,
    )
    sv1 = np.array(
        [
            get_col(row, "SvPositionXEcefMeters", np.nan),
            get_col(row, "SvPositionYEcefMeters", np.nan),
            get_col(row, "SvPositionZEcefMeters", np.nan),
        ],
        dtype=float,
    )
    if not (np.isfinite(sv0).all() and np.isfinite(sv1).all()):
        return None

    r0 = rx0 - sv0
    r1 = rx1 - sv1
    rho0 = float(np.linalg.norm(r0))
    rho1 = float(np.linalg.norm(r1))
    if rho0 < 1e-3 or rho1 < 1e-3:
        return None
    los0 = r0 / rho0
    los1 = r1 / rho1
    geom_rate = (rho1 - rho0) / dt_s
    wls_disp = rx1 - rx0
    wls_speed = float(np.linalg.norm(wls_disp) / dt_s)

    az0 = math.radians(get_col(prev, "SvAzimuthDegrees"))
    az1 = math.radians(get_col(row, "SvAzimuthDegrees"))
    pr_unc0 = math.log1p(max(get_col(prev, "PseudorangeRateUncertaintyMetersPerSecond"), 0.0))
    pr_unc1 = math.log1p(max(get_col(row, "PseudorangeRateUncertaintyMetersPerSecond"), 0.0))
    raw_pr_unc = math.log1p(max(get_col(row, "RawPseudorangeUncertaintyMeters"), 0.0))
    rx_clock_drift = get_col(row, "DriftNanosPerSecond") * C_MPS_PER_NSPS
    rx_bias_unc = get_col(row, "BiasUncertaintyNanos") * C_MPS_PER_NSPS

    feats = [
        dt_s,
        pr0,
        pr1,
        0.5 * (pr0 + pr1),
        pr1 - pr0,
        pr_unc0,
        pr_unc1,
        get_col(prev, "Cn0DbHz"),
        get_col(row, "Cn0DbHz"),
        get_col(row, "Cn0DbHz") - get_col(prev, "Cn0DbHz"),
        get_col(prev, "SvElevationDegrees"),
        get_col(row, "SvElevationDegrees"),
        math.sin(az0),
        math.cos(az0),
        math.sin(az1),
        math.cos(az1),
        los0[0],
        los0[1],
        los0[2],
        los1[0],
        los1[1],
        los1[2],
        los1[0] - los0[0],
        los1[1] - los0[1],
        los1[2] - los0[2],
        geom_rate,
        geom_rate - pr1,
        wls_disp[0],
        wls_disp[1],
        wls_disp[2],
        wls_speed,
        get_col(row, "SvVelocityXEcefMetersPerSecond"),
        get_col(row, "SvVelocityYEcefMetersPerSecond"),
        get_col(row, "SvVelocityZEcefMetersPerSecond"),
        get_col(row, "SvClockDriftMetersPerSecond"),
        get_col(row, "SvClockBiasMeters"),
        raw_pr_unc,
        get_col(row, "IonosphericDelayMeters"),
        get_col(row, "TroposphericDelayMeters"),
        get_col(row, "IsrbMeters"),
        rx_clock_drift,
        rx_bias_unc,
        get_col(row, "AgcDb"),
        get_col(row, "SnrInDb"),
        get_col(row, "CarrierFrequencyHz") / 1e9,
        get_col(row, "MultipathIndicator"),
    ]
    feats.extend(one_hot_constellation(get_col(row, "ConstellationType", np.nan)))
    feats.extend(one_hot(get_str_col(row, "SignalType"), signal_types))
    feats.extend(one_hot(get_str_col(row, "CodeType"), code_types))
    return feats


# Statistics come from the source-training pairs only and are stored in each
# checkpoint; indicator columns are standardized like every other input.
def compute_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std[std < 1e-6] = 1.0
    return mean, std


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (x - mean) / std
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
