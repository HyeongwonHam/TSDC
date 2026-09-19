from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from tsdc.geo import haversine_m


METRICS = ["p50_m", "p95_m", "score_m"]


def score_errors(errors_m: np.ndarray) -> tuple[float, float, float]:
    p50 = float(np.percentile(errors_m, 50))
    p95 = float(np.percentile(errors_m, 95))
    return p50, p95, 0.5 * (p50 + p95)


# Predictions and ground truth are matched at identical millisecond timestamps,
# without interpolation.
def evaluate_track(gt: pd.DataFrame, pred: pd.DataFrame) -> tuple[float, float, float, int]:
    merged = gt.merge(pred, on="UnixTimeMillis", how="inner")
    if merged.empty:
        raise ValueError("No overlapping timestamps")
    err = haversine_m(
        merged["LatitudeDegrees"].to_numpy(dtype=float),
        merged["LongitudeDegrees"].to_numpy(dtype=float),
        merged["lat"].to_numpy(dtype=float),
        merged["lon"].to_numpy(dtype=float),
    )
    p50, p95, score = score_errors(err)
    return p50, p95, score, len(merged)


def normalize_device(device: str) -> str:
    value = device.lower().replace("google", "").replace("_", "").replace("-", "")
    if "pixel4xlmodded" in value:
        return "Pixel 4 XL Modded"
    if "pixel4modded" in value:
        return "Pixel 4 Modded"
    if "pixel4xl" in value:
        return "Pixel 4 XL"
    if "pixel4" in value:
        return "Pixel 4"
    if "pixel5" in value:
        return "Pixel 5"
    if "pixel6pro" in value:
        return "Pixel 6 Pro"
    if "pixel7pro" in value:
        return "Pixel 7 Pro"
    if "pixel7" in value:
        return "Pixel 7"
    if "mi8" in value or "xiaomi" in value:
        return "Xiaomi Mi 8"
    if "s20ultra" in value or "g988" in value:
        return "Samsung S20 Ultra"
    if "s21ultra" in value:
        return "Samsung S21 Ultra"
    if "s22ultra" in value or "s908" in value:
        return "Samsung S22 Ultra"
    return device


# A physical phone session is a physical collection (repeated GSDC releases of
# the same drive merged) plus a normalized phone model.
def attach_sessions(detail: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    merged = detail.merge(
        mapping[["route_group", "split", "route_leakage_group", "physical_collection_group"]],
        on="route_group",
        how="left",
        validate="many_to_one",
    )
    if merged["physical_collection_group"].isna().any():
        unknown = sorted(merged.loc[merged["physical_collection_group"].isna(), "route_group"].unique())
        raise ValueError(f"Results contain unmapped route groups: {unknown}")
    merged["device_normalized"] = merged["device"].astype(str).map(normalize_device)
    merged["physical_session_group"] = merged["physical_collection_group"] + "|" + merged["device_normalized"]
    return merged


def collapse_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "seed", "physical_session_group", "route_leakage_group"]
    return frame.groupby(keys, as_index=False, dropna=False)[METRICS].mean()


# Session-weighted means per run, then mean and standard deviation over runs.
def aggregate_runs(sessions: pd.DataFrame) -> pd.DataFrame:
    per_run = sessions.groupby(["method", "seed"], as_index=False, dropna=False)[METRICS].mean()
    rows = []
    for method, group in per_run.groupby("method", sort=False):
        row: dict[str, Any] = {
            "method": method,
            "sessions": int(sessions.loc[sessions["method"] == method, "physical_session_group"].nunique()),
            "runs": int(group["seed"].notna().sum()),
        }
        for metric in METRICS:
            row[metric] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def session_mean_across_runs(sessions: pd.DataFrame, method: str) -> pd.DataFrame:
    frame = sessions[sessions["method"] == method]
    return frame.groupby(["physical_session_group", "route_leakage_group"], as_index=False)[METRICS].mean()


def clustered_bootstrap(group_values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(group_values), size=(samples, len(group_values)))
    means = group_values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


# Paired TSDC-minus-reference differences: runs are averaged within each physical
# phone session, sessions are averaged within each leakage group, and the bootstrap
# and Wilcoxon test give every leakage group equal weight.
def paired_group_statistics(
    tsdc_sessions: pd.DataFrame,
    reference_sessions: pd.DataFrame,
    samples: int,
    rng: np.random.Generator,
    metric: str = "score_m",
) -> dict[str, Any]:
    paired = tsdc_sessions.merge(
        reference_sessions,
        on=["physical_session_group", "route_leakage_group"],
        suffixes=("_tsdc", "_reference"),
        validate="one_to_one",
    )
    if len(paired) != len(tsdc_sessions) or len(paired) != len(reference_sessions):
        raise ValueError("Unpaired physical sessions")
    delta = paired[f"{metric}_tsdc"] - paired[f"{metric}_reference"]
    group_delta = delta.groupby(paired["route_leakage_group"]).mean()
    low, high = clustered_bootstrap(group_delta.to_numpy(dtype=float), samples, rng)
    nonzero = group_delta[np.abs(group_delta) > 1e-12]
    test = wilcoxon(nonzero, alternative="two-sided", method="auto") if len(nonzero) else None
    return {
        "physical_sessions": len(paired),
        "leakage_groups": len(group_delta),
        "better_sessions": int((delta < 0).sum()),
        "mean_session_delta_m": float(delta.mean()),
        "mean_group_delta_m": float(group_delta.mean()),
        "ci_low_m": low,
        "ci_high_m": high,
        "wilcoxon_p": float(test.pvalue) if test else math.nan,
    }
