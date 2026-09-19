from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tsdc.features import base_feature_names, build_pair_input_features, get_col
from tsdc.geo import ecef_to_geodetic


PAIR_GROUP_COLUMNS = ["ConstellationType", "Svid", "CarrierFrequencyHz", "CodeType", "SignalType"]


def required_columns() -> list[str]:
    return [
        "utcTimeMillis",
        "Svid",
        "ConstellationType",
        "PseudorangeRateMetersPerSecond",
        "AccumulatedDeltaRangeState",
        "AccumulatedDeltaRangeMeters",
        "SvPositionXEcefMeters",
        "SvPositionYEcefMeters",
        "SvPositionZEcefMeters",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]


def read_device_gnss(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in required_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}. Available: {list(df.columns)}")
    df["utcTimeMillis"] = df["utcTimeMillis"].astype(np.int64)
    return df


# Nominal trajectory: one WLS ECEF position per epoch, taken from device_gnss.csv.
def make_wls_state(df_raw: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "utcTimeMillis",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    wls = df_raw[cols].dropna().groupby("utcTimeMillis", as_index=False).first()
    geodetic = [
        ecef_to_geodetic(
            row.WlsPositionXEcefMeters,
            row.WlsPositionYEcefMeters,
            row.WlsPositionZEcefMeters,
        )
        for row in wls.itertuples(index=False)
    ]
    wls["lat"] = [x[0] for x in geodetic]
    wls["lon"] = [x[1] for x in geodetic]
    wls["alt"] = [x[2] for x in geodetic]
    return wls.sort_values("utcTimeMillis").reset_index(drop=True)


# Android ADR state: bit 0 = valid, bit 1 = reset, bit 2 = cycle slip.
def adr_valid_mask(state: pd.Series) -> pd.Series:
    s = state.fillna(0).astype(int)
    return ((s & 1) != 0) & ((s & 6) == 0)


def load_ground_truth(trace_dir: Path) -> pd.DataFrame:
    gt = pd.read_csv(trace_dir / "ground_truth.csv")
    if {"UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"}.issubset(gt.columns):
        columns = ["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        if "AltitudeMeters" in gt.columns:
            columns.append("AltitudeMeters")
        out = gt[columns].copy()
    elif {"millisSinceGpsEpoch", "latDeg", "lngDeg"}.issubset(gt.columns):
        # GSDC 2021: kept on the GPS-epoch time base used by the converted device_gnss.csv.
        columns = ["millisSinceGpsEpoch", "latDeg", "lngDeg"]
        if "heightAboveWgs84EllipsoidM" in gt.columns:
            columns.append("heightAboveWgs84EllipsoidM")
        out = gt[columns].copy()
        out = out.rename(
            columns={
                "millisSinceGpsEpoch": "UnixTimeMillis",
                "latDeg": "LatitudeDegrees",
                "lngDeg": "LongitudeDegrees",
                "heightAboveWgs84EllipsoidM": "AltitudeMeters",
            }
        )
    else:
        raise ValueError(f"Unsupported ground_truth.csv columns in {trace_dir}")
    out["UnixTimeMillis"] = out["UnixTimeMillis"].astype(np.int64)
    return out.sort_values("UnixTimeMillis")


def load_route_groups(split_json: Path, split_name: str) -> list[str]:
    with split_json.open("r", encoding="utf-8") as f:
        splits = json.load(f)
    if split_name not in splits:
        raise SystemExit(f"Missing split '{split_name}'. Available: {sorted(splits)}")
    return list(splits[split_name])


def expand_trace_dirs(data_root: Path, route_groups: list[str]) -> list[Path]:
    trace_dirs: list[Path] = []
    for group_id in route_groups:
        route_dir = data_root / group_id
        if not route_dir.exists():
            continue
        for trace_dir in sorted(p for p in route_dir.iterdir() if p.is_dir()):
            if (trace_dir / "device_gnss.csv").exists() and (trace_dir / "ground_truth.csv").exists():
                trace_dirs.append(trace_dir)
    return trace_dirs


# Source training uses one release trace per physical phone session
# (splits/source_trace_manifest.csv); repeated releases of the same session are not reused.
def load_trace_manifest(
    manifest_path: Path, data_root: Path, split_name: str, route_groups: list[str]
) -> list[Path]:
    frame = pd.read_csv(manifest_path)
    frame = frame[frame["split"] == split_name]
    if frame.empty:
        raise SystemExit(f"Trace manifest has no rows for split '{split_name}'")
    unknown_routes = sorted(set(frame["route_group"]) - set(route_groups))
    if unknown_routes:
        raise SystemExit(f"Trace manifest contains routes outside split '{split_name}': {unknown_routes}")
    trace_dirs: list[Path] = []
    for value in frame["trace"]:
        trace_dir = data_root / str(value)
        if not (trace_dir / "device_gnss.csv").exists():
            raise SystemExit(f"Manifest trace has no device_gnss.csv: {trace_dir}")
        if not (trace_dir / "ground_truth.csv").exists():
            raise SystemExit(f"Manifest trace has no ground_truth.csv: {trace_dir}")
        trace_dirs.append(trace_dir)
    return sorted(trace_dirs)


# TDCP is used only to construct the training target:
# y = (ADR_k - ADR_{k-1}) / dt - PRR_k.
def build_pair_features(
    prev: Any,
    row: Any,
    rx0: np.ndarray,
    rx1: np.ndarray,
    signal_types: list[str],
    code_types: list[str],
) -> tuple[list[float] | None, float | None, float | None]:
    t0 = int(get_col(prev, "utcTimeMillis"))
    t1 = int(get_col(row, "utcTimeMillis"))
    dt_s = (t1 - t0) / 1000.0
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        return None, None, None

    adr0 = get_col(prev, "AccumulatedDeltaRangeMeters", np.nan)
    adr1 = get_col(row, "AccumulatedDeltaRangeMeters", np.nan)
    if not math.isfinite(adr0) or not math.isfinite(adr1):
        return None, None, None
    tdcp_rate = (adr1 - adr0) / dt_s
    pr1 = get_col(row, "PseudorangeRateMetersPerSecond")
    target = tdcp_rate - pr1
    feats = build_pair_input_features(prev, row, rx0, rx1, signal_types, code_types)
    if feats is None:
        return None, None, None
    return feats, target, tdcp_rate


def extract_training_pairs(
    trace_dir: Path,
    signal_types: list[str],
    code_types: list[str],
    min_dt_s: float,
    max_dt_s: float,
    label_gate_mps: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    df = read_device_gnss(trace_dir / "device_gnss.csv")
    wls = make_wls_state(df)
    epoch_to_idx = {int(t): i for i, t in enumerate(wls["utcTimeMillis"].to_numpy(dtype=np.int64))}
    rx = wls[
        ["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]
    ].to_numpy(dtype=float)

    # Same satellite, carrier frequency, code and signal across the two epochs.
    group_cols = [col for col in PAIR_GROUP_COLUMNS if col in df.columns]
    needed = sorted(
        set(
            group_cols
            + [
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
            ]
        )
    )
    needed = [c for c in needed if c in df.columns]
    work = df[needed].dropna(
        subset=[
            "utcTimeMillis",
            "PseudorangeRateMetersPerSecond",
            "AccumulatedDeltaRangeMeters",
            "SvPositionXEcefMeters",
            "SvPositionYEcefMeters",
            "SvPositionZEcefMeters",
        ]
    )
    work = work.sort_values(group_cols + ["utcTimeMillis"]).reset_index(drop=True)
    work["adr_valid"] = adr_valid_mask(work["AccumulatedDeltaRangeState"])

    features: list[list[float]] = []
    targets: list[float] = []
    stats = {
        "candidate_pairs": 0,
        "accepted": 0,
        "skip_dt": 0,
        "skip_state": 0,
        "skip_epoch": 0,
        "skip_nonadjacent": 0,
        "skip_label_gate": 0,
        "skip_feature": 0,
    }

    for _, grp in work.groupby(group_cols, dropna=False, sort=False):
        prev = None
        for row in grp.itertuples(index=False):
            if prev is None:
                prev = row
                continue
            stats["candidate_pairs"] += 1
            t0 = int(get_col(prev, "utcTimeMillis"))
            t1 = int(get_col(row, "utcTimeMillis"))
            dt_s = (t1 - t0) / 1000.0
            if not math.isfinite(dt_s) or dt_s < min_dt_s or dt_s > max_dt_s:
                stats["skip_dt"] += 1
                prev = row
                continue
            if not (bool(getattr(prev, "adr_valid")) and bool(getattr(row, "adr_valid"))):
                stats["skip_state"] += 1
                prev = row
                continue
            i0 = epoch_to_idx.get(t0)
            i1 = epoch_to_idx.get(t1)
            if i0 is None or i1 is None:
                stats["skip_epoch"] += 1
                prev = row
                continue
            if i1 != i0 + 1:
                stats["skip_nonadjacent"] += 1
                prev = row
                continue
            feats, target, _ = build_pair_features(prev, row, rx[i0], rx[i1], signal_types, code_types)
            if feats is None or target is None or not math.isfinite(target):
                stats["skip_feature"] += 1
                prev = row
                continue
            if abs(target) > label_gate_mps:
                stats["skip_label_gate"] += 1
                prev = row
                continue
            features.append(feats)
            targets.append(float(target))
            stats["accepted"] += 1
            prev = row

    if features:
        return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32), stats
    return (
        np.empty((0, len(base_feature_names(signal_types, code_types))), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        stats,
    )
