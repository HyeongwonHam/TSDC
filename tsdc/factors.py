from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from tsdc.data import PAIR_GROUP_COLUMNS, adr_valid_mask
from tsdc.features import build_pair_input_features, get_col
from tsdc.fgo import TdcpFactor, make_direct_factor_terms
from tsdc.geo import ecef_to_enu_matrix
from tsdc.inference import MAX_ABS_MU_MPS, MAX_SCALE_MPS, ModelBundle, output_gate, predict_residuals


TSDC = "TSDC-FGO"
TSDC_FIXED_SIGMA = "TSDC-FGO, fixed sigma"
ENDPOINT_TSDC_MASK = "Endpoint Doppler FGO, TSDC mask"
AVERAGE_TSDC_MASK = "Averaged Doppler FGO, TSDC mask"
ENDPOINT_COMMON = "Endpoint, common ADR-valid mask"
TDCP_COMMON = "Screened TDCP, common ADR-valid mask"
TSDC_COMMON = "TSDC, common ADR-valid mask"
ENDPOINT = "Endpoint Doppler FGO"
AVERAGE = "Averaged Doppler FGO"
SCREENED_TDCP = "Screened TDCP-FGO"


def nominal_geometry(wls: pd.DataFrame) -> tuple[dict[int, int], np.ndarray, list[np.ndarray]]:
    epoch_to_idx = {int(t): i for i, t in enumerate(wls["utcTimeMillis"].to_numpy(dtype=np.int64))}
    rx_ecef = wls[
        ["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]
    ].to_numpy(dtype=float)
    enu_matrices = [ecef_to_enu_matrix(float(row.lat), float(row.lon)) for row in wls.itertuples(index=False)]
    return epoch_to_idx, rx_ecef, enu_matrices


# Adjacent-epoch pairs of the same satellite signal on consecutive nominal epochs,
# and the network input of each pair. ADR is read only to flag pairs that also pass
# the screened-TDCP checks (common ADR-valid mask of Table 2).
def build_pair_contexts(
    raw: pd.DataFrame,
    wls: pd.DataFrame,
    bundle: ModelBundle,
    min_dt_s: float = 0.8,
    max_dt_s: float = 1.2,
    label_gate_mps: float = 1.5,
    use_sat_clock_bias: bool = True,
) -> tuple[list[dict[str, Any]], list[list[float]]]:
    epoch_to_idx, rx_ecef, enu_matrices = nominal_geometry(wls)
    group_cols = [column for column in PAIR_GROUP_COLUMNS if column in raw.columns]
    feature_cols = [
        "utcTimeMillis",
        "PseudorangeRateMetersPerSecond",
        "PseudorangeRateUncertaintyMetersPerSecond",
        "AccumulatedDeltaRangeState",
        "AccumulatedDeltaRangeMeters",
        "Cn0DbHz",
        "SvElevationDegrees",
        "SvAzimuthDegrees",
        "SvPositionXEcefMeters",
        "SvPositionYEcefMeters",
        "SvPositionZEcefMeters",
        "SvVelocityXEcefMetersPerSecond",
        "SvVelocityYEcefMetersPerSecond",
        "SvVelocityZEcefMetersPerSecond",
        "SvClockDriftMetersPerSecond",
        "SvClockBiasMeters",
        "RawPseudorangeUncertaintyMeters",
        "IonosphericDelayMeters",
        "TroposphericDelayMeters",
        "IsrbMeters",
        "DriftNanosPerSecond",
        "BiasUncertaintyNanos",
        "AgcDb",
        "SnrInDb",
        "CarrierFrequencyHz",
        "MultipathIndicator",
        "SignalType",
        "CodeType",
        "ConstellationType",
        "Svid",
    ]
    needed = [column for column in sorted(set(group_cols + feature_cols)) if column in raw.columns]
    work = raw[needed].dropna(
        subset=[
            "utcTimeMillis",
            "PseudorangeRateMetersPerSecond",
            "SvPositionXEcefMeters",
            "SvPositionYEcefMeters",
            "SvPositionZEcefMeters",
        ]
    )
    work = work.sort_values(group_cols + ["utcTimeMillis"]).reset_index(drop=True)
    work["adr_valid"] = (
        adr_valid_mask(work["AccumulatedDeltaRangeState"]) if "AccumulatedDeltaRangeState" in work else False
    )

    contexts: list[dict[str, Any]] = []
    features: list[list[float]] = []
    for _, group in work.groupby(group_cols, dropna=False, sort=False):
        previous = None
        for row in group.itertuples(index=False):
            if previous is None:
                previous = row
                continue
            t0 = int(get_col(previous, "utcTimeMillis"))
            t1 = int(get_col(row, "utcTimeMillis"))
            dt_s = (t1 - t0) / 1000.0
            if not math.isfinite(dt_s) or dt_s < min_dt_s or dt_s > max_dt_s:
                previous = row
                continue
            i0 = epoch_to_idx.get(t0)
            i1 = epoch_to_idx.get(t1)
            if i0 is None or i1 is None or i1 != i0 + 1:
                previous = row
                continue
            feature = build_pair_input_features(
                previous, row, rx_ecef[i0], rx_ecef[i1], bundle.signal_types, bundle.code_types
            )
            if feature is None:
                previous = row
                continue
            sv0 = np.asarray(
                [
                    get_col(previous, "SvPositionXEcefMeters", np.nan),
                    get_col(previous, "SvPositionYEcefMeters", np.nan),
                    get_col(previous, "SvPositionZEcefMeters", np.nan),
                ]
            )
            sv1 = np.asarray(
                [
                    get_col(row, "SvPositionXEcefMeters", np.nan),
                    get_col(row, "SvPositionYEcefMeters", np.nan),
                    get_col(row, "SvPositionZEcefMeters", np.nan),
                ]
            )
            r0 = rx_ecef[i0] - sv0
            r1 = rx_ecef[i1] - sv1
            rho0 = float(np.linalg.norm(r0))
            rho1 = float(np.linalg.norm(r1))
            if rho0 < 1e-3 or rho1 < 1e-3:
                previous = row
                continue
            adr0 = get_col(previous, "AccumulatedDeltaRangeMeters", np.nan)
            adr1 = get_col(row, "AccumulatedDeltaRangeMeters", np.nan)
            adr_present = math.isfinite(adr0) and math.isfinite(adr1)
            state_valid = bool(getattr(previous, "adr_valid", False)) and bool(getattr(row, "adr_valid", False))
            pr_rate0 = get_col(previous, "PseudorangeRateMetersPerSecond", np.nan)
            pr_rate1 = get_col(row, "PseudorangeRateMetersPerSecond", np.nan)
            tdcp_rate = (adr1 - adr0) / dt_s if adr_present else np.nan
            target = tdcp_rate - pr_rate1
            clean = bool(adr_present and state_valid and math.isfinite(target) and abs(target) <= label_gate_mps)
            sat_clock_delta = 0.0
            if use_sat_clock_bias:
                sat_clock_delta = get_col(row, "SvClockBiasMeters") - get_col(previous, "SvClockBiasMeters")
                if not math.isfinite(sat_clock_delta):
                    sat_clock_delta = 0.0
            contexts.append(
                {
                    "i0": i0,
                    "i1": i1,
                    "dt_s": dt_s,
                    "pr_rate0": pr_rate0,
                    "pr_rate1": pr_rate1,
                    "dadr": adr1 - adr0 if adr_present else np.nan,
                    "clean": clean,
                    "los0": enu_matrices[i0] @ (r0 / rho0),
                    "los1": enu_matrices[i1] @ (r1 / rho1),
                    "nominal_delta": rho1 - rho0,
                    "sat_clock_delta": sat_clock_delta,
                }
            )
            features.append(feature)
            previous = row
    return contexts, features


def add_predictions(contexts: list[dict[str, Any]], features: list[list[float]], bundle: ModelBundle) -> None:
    mu, scale = predict_residuals(features, bundle)
    for index, context in enumerate(contexts):
        context["mu_mps"] = float(mu[index])
        context["scale_mps"] = float(scale[index])


def make_factor(context: dict[str, Any], range_change_m: float, sigma_m: float) -> TdcpFactor:
    c0, c1, y = make_direct_factor_terms(
        context["los0"],
        context["los1"],
        measured_range_change_m=range_change_m,
        nominal_range_change_m=float(context["nominal_delta"]),
        satellite_clock_change_m=float(context["sat_clock_delta"]),
    )
    return TdcpFactor(i0=int(context["i0"]), i1=int(context["i1"]), c0=c0, c1=c1, y=y, sigma=float(sigma_m))


def learned_sigma_m(context: dict[str, Any], floor_m: float, multiplier: float) -> float:
    # sigma = max(1 m, 5 b dt): b is a rate (m/s), dt converts it to metres.
    return max(floor_m, multiplier * max(float(context["scale_mps"]), 1e-4) * float(context["dt_s"]))


# Endpoint, average and TSDC on the TSDC acceptance mask (|mu| <= 3 m/s, b <= 1 m/s).
def tsdc_mask_factors(
    contexts: list[dict[str, Any]],
    max_abs_mu_mps: float = MAX_ABS_MU_MPS,
    max_scale_mps: float = MAX_SCALE_MPS,
    fixed_sigma_m: float = 1.0,
    sigma_floor_m: float = 1.0,
    sigma_multiplier: float = 5.0,
) -> dict[str, list[TdcpFactor]]:
    factors: dict[str, list[TdcpFactor]] = {
        ENDPOINT_TSDC_MASK: [],
        AVERAGE_TSDC_MASK: [],
        TSDC: [],
        TSDC_FIXED_SIGMA: [],
    }
    for context in contexts:
        if not output_gate(context["mu_mps"], context["scale_mps"], max_abs_mu_mps, max_scale_mps):
            continue
        dt_s = float(context["dt_s"])
        pr0 = float(context["pr_rate0"])
        pr1 = float(context["pr_rate1"])
        mu = float(context["mu_mps"])
        sigma_m = learned_sigma_m(context, sigma_floor_m, sigma_multiplier)
        factors[ENDPOINT_TSDC_MASK].append(make_factor(context, pr1 * dt_s, fixed_sigma_m))
        factors[AVERAGE_TSDC_MASK].append(make_factor(context, 0.5 * (pr0 + pr1) * dt_s, fixed_sigma_m))
        factors[TSDC].append(make_factor(context, (pr1 + mu) * dt_s, sigma_m))
        factors[TSDC_FIXED_SIGMA].append(make_factor(context, (pr1 + mu) * dt_s, fixed_sigma_m))
    return factors


# Table 2: identical satellite-epoch factors and fixed sigma for all three
# measurements; a pair must pass both the TSDC gates and the screened-TDCP checks.
def common_mask_factors(
    contexts: list[dict[str, Any]],
    max_abs_mu_mps: float = MAX_ABS_MU_MPS,
    max_scale_mps: float = MAX_SCALE_MPS,
    fixed_sigma_m: float = 1.0,
) -> dict[str, list[TdcpFactor]]:
    factors: dict[str, list[TdcpFactor]] = {ENDPOINT_COMMON: [], TDCP_COMMON: [], TSDC_COMMON: []}
    for context in contexts:
        if not (context["clean"] and output_gate(context["mu_mps"], context["scale_mps"], max_abs_mu_mps, max_scale_mps)):
            continue
        dt_s = float(context["dt_s"])
        pr1 = float(context["pr_rate1"])
        factors[ENDPOINT_COMMON].append(make_factor(context, float(pr1 * dt_s), fixed_sigma_m))
        factors[TSDC_COMMON].append(make_factor(context, float((pr1 + float(context["mu_mps"])) * dt_s), fixed_sigma_m))
        factors[TDCP_COMMON].append(make_factor(context, float(context["dadr"]), fixed_sigma_m))
    return factors


# Model-free Doppler baselines, tuned on source validation: all pairs, no
# PRR-uncertainty gate selected, sigma 5 m (endpoint) and 1 m (average).
def build_raw_doppler_contexts(
    raw: pd.DataFrame,
    wls: pd.DataFrame,
    min_dt_s: float = 0.8,
    max_dt_s: float = 1.2,
    use_sat_clock_bias: bool = True,
) -> list[dict[str, Any]]:
    epoch_to_idx, rx_ecef, enu_matrices = nominal_geometry(wls)
    group_cols = [column for column in PAIR_GROUP_COLUMNS if column in raw.columns]
    needed_columns = [
        "utcTimeMillis",
        "PseudorangeRateMetersPerSecond",
        "PseudorangeRateUncertaintyMetersPerSecond",
        "SvPositionXEcefMeters",
        "SvPositionYEcefMeters",
        "SvPositionZEcefMeters",
        "SvClockBiasMeters",
    ]
    needed = [column for column in sorted(set(group_cols + needed_columns)) if column in raw]
    work = raw[needed].dropna(
        subset=[
            "utcTimeMillis",
            "PseudorangeRateMetersPerSecond",
            "SvPositionXEcefMeters",
            "SvPositionYEcefMeters",
            "SvPositionZEcefMeters",
        ]
    )
    work = work.sort_values(group_cols + ["utcTimeMillis"]).reset_index(drop=True)
    contexts: list[dict[str, Any]] = []
    for _, group in work.groupby(group_cols, dropna=False, sort=False):
        previous = None
        for row in group.itertuples(index=False):
            if previous is None:
                previous = row
                continue
            t0 = int(get_col(previous, "utcTimeMillis"))
            t1 = int(get_col(row, "utcTimeMillis"))
            dt_s = (t1 - t0) / 1000.0
            if not math.isfinite(dt_s) or dt_s < min_dt_s or dt_s > max_dt_s:
                previous = row
                continue
            i0 = epoch_to_idx.get(t0)
            i1 = epoch_to_idx.get(t1)
            if i0 is None or i1 is None or i1 != i0 + 1:
                previous = row
                continue
            sv0 = np.asarray(
                [
                    get_col(previous, "SvPositionXEcefMeters", np.nan),
                    get_col(previous, "SvPositionYEcefMeters", np.nan),
                    get_col(previous, "SvPositionZEcefMeters", np.nan),
                ],
                dtype=float,
            )
            sv1 = np.asarray(
                [
                    get_col(row, "SvPositionXEcefMeters", np.nan),
                    get_col(row, "SvPositionYEcefMeters", np.nan),
                    get_col(row, "SvPositionZEcefMeters", np.nan),
                ],
                dtype=float,
            )
            r0 = rx_ecef[i0] - sv0
            r1 = rx_ecef[i1] - sv1
            rho0 = float(np.linalg.norm(r0))
            rho1 = float(np.linalg.norm(r1))
            if rho0 < 1e-3 or rho1 < 1e-3:
                previous = row
                continue
            sat_clock_delta = 0.0
            if use_sat_clock_bias:
                sat_clock_delta = get_col(row, "SvClockBiasMeters") - get_col(previous, "SvClockBiasMeters")
                if not math.isfinite(sat_clock_delta):
                    sat_clock_delta = 0.0
            uncertainty0 = get_col(previous, "PseudorangeRateUncertaintyMetersPerSecond", math.inf)
            uncertainty1 = get_col(row, "PseudorangeRateUncertaintyMetersPerSecond", math.inf)
            max_uncertainty = max(uncertainty0, uncertainty1)
            if not math.isfinite(max_uncertainty):
                max_uncertainty = math.inf
            contexts.append(
                {
                    "i0": i0,
                    "i1": i1,
                    "dt_s": dt_s,
                    "pr_rate0": get_col(previous, "PseudorangeRateMetersPerSecond", np.nan),
                    "pr_rate1": get_col(row, "PseudorangeRateMetersPerSecond", np.nan),
                    "max_prr_uncertainty_mps": max_uncertainty,
                    "los0": enu_matrices[i0] @ (r0 / rho0),
                    "los1": enu_matrices[i1] @ (r1 / rho1),
                    "nominal_delta": rho1 - rho0,
                    "sat_clock_delta": sat_clock_delta,
                }
            )
            previous = row
    return contexts


def raw_doppler_factors(
    contexts: list[dict[str, Any]], rate_mode: str, sigma_m: float, prr_uncertainty_gate_mps: float = math.inf
) -> list[TdcpFactor]:
    factors: list[TdcpFactor] = []
    for context in contexts:
        if float(context["max_prr_uncertainty_mps"]) > prr_uncertainty_gate_mps:
            continue
        if rate_mode == "average":
            pr_rate_mps = 0.5 * (float(context["pr_rate0"]) + float(context["pr_rate1"]))
        elif rate_mode == "endpoint":
            pr_rate_mps = float(context["pr_rate1"])
        else:
            raise ValueError(f"Unknown rate mode: {rate_mode}")
        factors.append(make_factor(context, pr_rate_mps * float(context["dt_s"]), sigma_m))
    return factors


# Screened ADR-derived TDCP-FGO (diagnostic reference): valid ADR without reset or
# cycle slip, |TDCP rate - PRR_k| <= 1.5 m/s, sigma 2 m. Intervals without valid
# TDCP are dropped, not filled with Doppler.
def screened_tdcp_factors(
    df_raw: pd.DataFrame,
    wls: pd.DataFrame,
    tdcp_sigma_m: float = 2.0,
    slip_threshold_mps: float = 1.5,
    min_dt_s: float = 0.8,
    max_dt_s: float = 1.2,
    use_sat_clock_bias: bool = True,
) -> list[TdcpFactor]:
    epoch_to_idx, rx_ecef, enu_mats = nominal_geometry(wls)
    group_cols = [col for col in PAIR_GROUP_COLUMNS if col in df_raw.columns]
    needed = [
        "utcTimeMillis",
        "AccumulatedDeltaRangeState",
        "AccumulatedDeltaRangeMeters",
        "PseudorangeRateMetersPerSecond",
        "SvPositionXEcefMeters",
        "SvPositionYEcefMeters",
        "SvPositionZEcefMeters",
    ]
    if use_sat_clock_bias and "SvClockBiasMeters" in df_raw.columns:
        needed.append("SvClockBiasMeters")
    work = df_raw[group_cols + needed].dropna(
        subset=[
            "utcTimeMillis",
            "AccumulatedDeltaRangeMeters",
            "SvPositionXEcefMeters",
            "SvPositionYEcefMeters",
            "SvPositionZEcefMeters",
        ]
    )
    work = work.sort_values(group_cols + ["utcTimeMillis"])
    work["adr_valid"] = adr_valid_mask(work["AccumulatedDeltaRangeState"])

    factors: list[TdcpFactor] = []
    for _, grp in work.groupby(group_cols, dropna=False, sort=False):
        prev = None
        for row in grp.itertuples(index=False):
            if prev is None:
                prev = row
                continue
            t0 = int(getattr(prev, "utcTimeMillis"))
            t1 = int(getattr(row, "utcTimeMillis"))
            dt_s = (t1 - t0) / 1000.0
            if not np.isfinite(dt_s) or dt_s < min_dt_s or dt_s > max_dt_s:
                prev = row
                continue
            if not (bool(getattr(prev, "adr_valid")) and bool(getattr(row, "adr_valid"))):
                prev = row
                continue
            dadr = float(getattr(row, "AccumulatedDeltaRangeMeters")) - float(getattr(prev, "AccumulatedDeltaRangeMeters"))
            tdcp_rate = dadr / dt_s
            pr_rate = float(getattr(row, "PseudorangeRateMetersPerSecond"))
            if abs(tdcp_rate - pr_rate) > slip_threshold_mps:
                prev = row
                continue
            i0 = epoch_to_idx.get(t0)
            i1 = epoch_to_idx.get(t1)
            if i0 is None or i1 is None or i1 != i0 + 1:
                prev = row
                continue
            sv0 = np.array(
                [
                    float(getattr(prev, "SvPositionXEcefMeters")),
                    float(getattr(prev, "SvPositionYEcefMeters")),
                    float(getattr(prev, "SvPositionZEcefMeters")),
                ],
                dtype=float,
            )
            sv1 = np.array(
                [
                    float(getattr(row, "SvPositionXEcefMeters")),
                    float(getattr(row, "SvPositionYEcefMeters")),
                    float(getattr(row, "SvPositionZEcefMeters")),
                ],
                dtype=float,
            )
            r0_vec = rx_ecef[i0] - sv0
            r1_vec = rx_ecef[i1] - sv1
            rho0 = float(np.linalg.norm(r0_vec))
            rho1 = float(np.linalg.norm(r1_vec))
            if rho0 < 1e-3 or rho1 < 1e-3:
                prev = row
                continue
            los0_enu = enu_mats[i0] @ (r0_vec / rho0)
            los1_enu = enu_mats[i1] @ (r1_vec / rho1)
            sat_clock_delta = 0.0
            if use_sat_clock_bias and "SvClockBiasMeters" in work.columns:
                sat_clock_delta = float(getattr(row, "SvClockBiasMeters", 0.0)) - float(
                    getattr(prev, "SvClockBiasMeters", 0.0)
                )
                if not np.isfinite(sat_clock_delta):
                    sat_clock_delta = 0.0
            c0, c1, y = make_direct_factor_terms(
                los0_enu,
                los1_enu,
                measured_range_change_m=dadr,
                nominal_range_change_m=rho1 - rho0,
                satellite_clock_change_m=sat_clock_delta,
            )
            factors.append(TdcpFactor(i0=i0, i1=i1, c0=c0, c1=c1, y=y, sigma=tdcp_sigma_m))
            prev = row
    return factors
