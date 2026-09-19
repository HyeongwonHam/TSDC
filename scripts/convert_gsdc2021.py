"""Write device_gnss.csv for GSDC 2021 traces (derived file + raw log + Google WLS baseline)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


GPS_UNIX_OFFSET_MS = 315_964_782_000
WGS84_A = 6378137.0
WGS84_E2 = 0.00669437999014


RAW_NUMERIC_COLUMNS = [
    "utcTimeMillis",
    "TimeNanos",
    "LeapSecond",
    "TimeUncertaintyNanos",
    "FullBiasNanos",
    "BiasNanos",
    "BiasUncertaintyNanos",
    "DriftNanosPerSecond",
    "DriftUncertaintyNanosPerSecond",
    "HardwareClockDiscontinuityCount",
    "Svid",
    "TimeOffsetNanos",
    "State",
    "ReceivedSvTimeNanos",
    "ReceivedSvTimeUncertaintyNanos",
    "Cn0DbHz",
    "PseudorangeRateMetersPerSecond",
    "PseudorangeRateUncertaintyMetersPerSecond",
    "AccumulatedDeltaRangeState",
    "AccumulatedDeltaRangeMeters",
    "AccumulatedDeltaRangeUncertaintyMeters",
    "CarrierFrequencyHz",
    "CarrierCycles",
    "CarrierPhase",
    "CarrierPhaseUncertainty",
    "MultipathIndicator",
    "SnrInDb",
    "ConstellationType",
    "AgcDb",
]


OUTPUT_COLUMNS = [
    "MessageType",
    "utcTimeMillis",
    "TimeNanos",
    "LeapSecond",
    "TimeUncertaintyNanos",
    "FullBiasNanos",
    "BiasNanos",
    "BiasUncertaintyNanos",
    "DriftNanosPerSecond",
    "DriftUncertaintyNanosPerSecond",
    "HardwareClockDiscontinuityCount",
    "Svid",
    "TimeOffsetNanos",
    "State",
    "ReceivedSvTimeNanos",
    "ReceivedSvTimeUncertaintyNanos",
    "Cn0DbHz",
    "PseudorangeRateMetersPerSecond",
    "PseudorangeRateUncertaintyMetersPerSecond",
    "AccumulatedDeltaRangeState",
    "AccumulatedDeltaRangeMeters",
    "AccumulatedDeltaRangeUncertaintyMeters",
    "CarrierFrequencyHz",
    "CarrierCycles",
    "CarrierPhase",
    "CarrierPhaseUncertainty",
    "MultipathIndicator",
    "SnrInDb",
    "ConstellationType",
    "AgcDb",
    "CodeType",
    "ChipsetElapsedRealtimeNanos",
    "ArrivalTimeNanosSinceGpsEpoch",
    "RawPseudorangeMeters",
    "RawPseudorangeUncertaintyMeters",
    "SignalType",
    "ReceivedSvTimeNanosSinceGpsEpoch",
    "SvPositionXEcefMeters",
    "SvPositionYEcefMeters",
    "SvPositionZEcefMeters",
    "SvElevationDegrees",
    "SvAzimuthDegrees",
    "SvVelocityXEcefMetersPerSecond",
    "SvVelocityYEcefMetersPerSecond",
    "SvVelocityZEcefMetersPerSecond",
    "SvClockBiasMeters",
    "SvClockDriftMetersPerSecond",
    "IsrbMeters",
    "IonosphericDelayMeters",
    "TroposphericDelayMeters",
    "WlsPositionXEcefMeters",
    "WlsPositionYEcefMeters",
    "WlsPositionZEcefMeters",
]


def latlon_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.radians(lat_deg.astype(float))
    lon = np.radians(lon_deg.astype(float))
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (n + alt_m) * cos_lat * np.cos(lon)
    y = (n + alt_m) * cos_lat * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return x, y, z


def ecef_to_latlon(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = np.sqrt(WGS84_A**2 * (1.0 - WGS84_E2))
    ep = np.sqrt((WGS84_A**2 - b**2) / b**2)
    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(WGS84_A * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(
        z + ep**2 * b * np.sin(th) ** 3,
        p - WGS84_E2 * WGS84_A * np.cos(th) ** 3,
    )
    return np.degrees(lat), np.degrees(lon)


def ecef_to_az_el(rx: np.ndarray, sv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat, lon = ecef_to_latlon(rx[:, 0], rx[:, 1], rx[:, 2])
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    diff = sv - rx
    east = -np.sin(lon_r) * diff[:, 0] + np.cos(lon_r) * diff[:, 1]
    north = (
        -np.sin(lat_r) * np.cos(lon_r) * diff[:, 0]
        - np.sin(lat_r) * np.sin(lon_r) * diff[:, 1]
        + np.cos(lat_r) * diff[:, 2]
    )
    up = (
        np.cos(lat_r) * np.cos(lon_r) * diff[:, 0]
        + np.cos(lat_r) * np.sin(lon_r) * diff[:, 1]
        + np.sin(lat_r) * diff[:, 2]
    )
    horiz = np.sqrt(east**2 + north**2)
    el = np.degrees(np.arctan2(up, horiz))
    az = (np.degrees(np.arctan2(east, north)) + 360.0) % 360.0
    return az, el


# The 2021 derived files carry no signal/code labels; they are inferred from
# constellation and carrier frequency.
def signal_type_from_raw(constellation: pd.Series, carrier_hz: pd.Series) -> pd.Series:
    freq = pd.to_numeric(carrier_hz, errors="coerce").to_numpy(dtype=float)
    const = pd.to_numeric(constellation, errors="coerce").fillna(-1).astype(int).to_numpy()
    out = np.full(len(const), "", dtype=object)
    l1 = np.isclose(freq, 1_575_420_000.0, rtol=0.0, atol=2_000_000.0)
    l5 = np.isclose(freq, 1_176_450_000.0, rtol=0.0, atol=2_000_000.0)
    out[(const == 1) & l1] = "GPS_L1"
    out[(const == 1) & l5] = "GPS_L5"
    out[(const == 6) & l1] = "GAL_E1"
    out[(const == 6) & l5] = "GAL_E5A"
    out[const == 3] = "GLO_G1"
    return pd.Series(out, index=carrier_hz.index)


def code_type_from_signal(signal: pd.Series) -> pd.Series:
    mapping = {
        "GPS_L1": "C",
        "GPS_L5": "Q",
        "GAL_E1": "C",
        "GAL_E5A": "Q",
        "GLO_G1": "C",
    }
    return signal.map(mapping).fillna("")


def parse_raw_gnss_log(log_path: Path) -> pd.DataFrame:
    raw_header: list[str] | None = None
    rows: list[list[str]] = []
    with log_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for line in f:
            if line.startswith("# Raw,"):
                raw_header = line[2:].strip().split(",")
            elif line.startswith("Raw,"):
                rows.append(next(csv.reader([line])))

    if raw_header is None:
        raise ValueError(f"Raw header not found: {log_path}")
    if not rows:
        raise ValueError(f"No Raw rows found: {log_path}")

    width = len(raw_header)
    fixed_rows = [row[:width] + [""] * max(width - len(row), 0) for row in rows]
    raw = pd.DataFrame(fixed_rows, columns=raw_header)
    for col in RAW_NUMERIC_COLUMNS:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["gpsMillis"] = raw["utcTimeMillis"] - GPS_UNIX_OFFSET_MS
    raw["SignalType"] = signal_type_from_raw(raw["ConstellationType"], raw["CarrierFrequencyHz"])
    raw = raw[raw["SignalType"] != ""].copy()
    raw["gpsMillis"] = raw["gpsMillis"].round().astype("int64")
    raw["ConstellationType"] = raw["ConstellationType"].astype("Int64")
    raw["Svid"] = raw["Svid"].astype("Int64")
    raw = raw.drop_duplicates(["gpsMillis", "ConstellationType", "Svid", "SignalType"], keep="first")
    return raw


# Google's 2021 baseline latitude/longitude/height, converted to ECEF; this is the
# nominal (WLS) trajectory for 2021 traces.
def load_wls_baseline(baseline_path: Path) -> pd.DataFrame:
    wls = pd.read_csv(baseline_path)
    x, y, z = latlon_to_ecef(
        wls["latDeg"].to_numpy(dtype=float),
        wls["lngDeg"].to_numpy(dtype=float),
        wls["heightAboveWgs84EllipsoidM"].to_numpy(dtype=float),
    )
    wls["WlsPositionXEcefMeters"] = x
    wls["WlsPositionYEcefMeters"] = y
    wls["WlsPositionZEcefMeters"] = z
    return wls[
        [
            "collectionName",
            "phoneName",
            "millisSinceGpsEpoch",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
    ].copy()


def load_ground_truth_times(trace_dir: Path) -> np.ndarray:
    gt_path = trace_dir / "ground_truth.csv"
    if not gt_path.exists() or gt_path.stat().st_size == 0:
        return np.array([], dtype=np.int64)
    gt = pd.read_csv(gt_path)
    if "millisSinceGpsEpoch" in gt.columns:
        values = gt["millisSinceGpsEpoch"]
    elif "UnixTimeMillis" in gt.columns:
        values = gt["UnixTimeMillis"]
    else:
        return np.array([], dtype=np.int64)
    return np.sort(values.dropna().astype("int64").unique())


def snap_times_to_ground_truth(times: pd.Series, gt_times: np.ndarray, tolerance_ms: int) -> tuple[pd.Series, float]:
    if len(gt_times) == 0:
        return times, 0.0
    values = times.to_numpy(dtype=np.int64)
    idx = np.searchsorted(gt_times, values)
    nearest = np.empty(len(values), dtype=np.int64)
    delta = np.full(len(values), np.iinfo(np.int64).max, dtype=np.int64)
    for offset in (-1, 0):
        cand_idx = idx + offset
        valid = (cand_idx >= 0) & (cand_idx < len(gt_times))
        cand = np.empty(len(values), dtype=np.int64)
        cand[valid] = gt_times[cand_idx[valid]]
        cand_delta = np.full(len(values), np.iinfo(np.int64).max, dtype=np.int64)
        cand_delta[valid] = np.abs(cand[valid] - values[valid])
        use = cand_delta < delta
        nearest[use] = cand[use]
        delta[use] = cand_delta[use]
    snap = delta <= tolerance_ms
    out = values.copy()
    out[snap] = nearest[snap]
    return pd.Series(out, index=times.index), float(snap.mean())


def convert_trace(
    trace_dir: Path,
    baseline_wls: pd.DataFrame,
    out_path: Path,
    wls_tolerance_ms: int = 1000,
    gt_snap_tolerance_ms: int = 5,
) -> dict[str, object]:
    derived_files = sorted(trace_dir.glob("*_derived.csv"))
    log_files = sorted(trace_dir.glob("*_GnssLog.txt"))
    if len(derived_files) != 1 or len(log_files) != 1:
        return {"trace": str(trace_dir), "status": "missing_input"}

    derived = pd.read_csv(derived_files[0])
    raw = parse_raw_gnss_log(log_files[0])
    gt_times = load_ground_truth_times(trace_dir)

    # 2021 files are indexed by millisSinceGpsEpoch. It is written to the
    # utcTimeMillis column so that it matches the 2021 ground_truth.csv time base.
    d = derived.rename(
        columns={
            "millisSinceGpsEpoch": "utcTimeMillis",
            "constellationType": "ConstellationType",
            "svid": "Svid",
            "signalType": "SignalType",
            "receivedSvTimeInGpsNanos": "ReceivedSvTimeNanosSinceGpsEpoch",
            "xSatPosM": "SvPositionXEcefMeters",
            "ySatPosM": "SvPositionYEcefMeters",
            "zSatPosM": "SvPositionZEcefMeters",
            "xSatVelMps": "SvVelocityXEcefMetersPerSecond",
            "ySatVelMps": "SvVelocityYEcefMetersPerSecond",
            "zSatVelMps": "SvVelocityZEcefMetersPerSecond",
            "satClkBiasM": "SvClockBiasMeters",
            "satClkDriftMps": "SvClockDriftMetersPerSecond",
            "rawPrM": "RawPseudorangeMeters",
            "rawPrUncM": "RawPseudorangeUncertaintyMeters",
            "isrbM": "IsrbMeters",
            "ionoDelayM": "IonosphericDelayMeters",
            "tropoDelayM": "TroposphericDelayMeters",
        }
    ).copy()
    d["utcTimeMillis"] = d["utcTimeMillis"].astype("int64")
    d["ConstellationType"] = d["ConstellationType"].astype("Int64")
    d["Svid"] = d["Svid"].astype("Int64")

    raw_cols = [
        col
        for col in RAW_NUMERIC_COLUMNS
        if col in raw.columns and col not in {"utcTimeMillis", "ConstellationType", "Svid"}
    ]
    raw_merge = raw[["gpsMillis", "ConstellationType", "Svid", "SignalType"] + raw_cols].copy()
    merged = d.merge(
        raw_merge,
        left_on=["utcTimeMillis", "ConstellationType", "Svid", "SignalType"],
        right_on=["gpsMillis", "ConstellationType", "Svid", "SignalType"],
        how="left",
    ).drop(columns=["gpsMillis"])

    collection = str(d["collectionName"].iloc[0])
    phone = str(d["phoneName"].iloc[0])
    wls = baseline_wls[(baseline_wls["collectionName"] == collection) & (baseline_wls["phoneName"] == phone)].copy()
    if not wls.empty:
        # Nearest-time match of the baseline position (within 1 s).
        merged["_row_order"] = np.arange(len(merged))
        wls = wls.sort_values("millisSinceGpsEpoch")
        merged = pd.merge_asof(
            merged.sort_values("utcTimeMillis"),
            wls[
                [
                    "millisSinceGpsEpoch",
                    "WlsPositionXEcefMeters",
                    "WlsPositionYEcefMeters",
                    "WlsPositionZEcefMeters",
                ]
            ],
            left_on="utcTimeMillis",
            right_on="millisSinceGpsEpoch",
            direction="nearest",
            tolerance=wls_tolerance_ms,
        )
        merged = merged.sort_values("_row_order").drop(columns=["_row_order", "millisSinceGpsEpoch"], errors="ignore")
    else:
        merged["WlsPositionXEcefMeters"] = np.nan
        merged["WlsPositionYEcefMeters"] = np.nan
        merged["WlsPositionZEcefMeters"] = np.nan

    rx = merged[["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]].to_numpy(dtype=float)
    sv = merged[["SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters"]].to_numpy(dtype=float)
    valid_geom = np.isfinite(rx).all(axis=1) & np.isfinite(sv).all(axis=1)
    merged["SvAzimuthDegrees"] = np.nan
    merged["SvElevationDegrees"] = np.nan
    if valid_geom.any():
        az, el = ecef_to_az_el(rx[valid_geom], sv[valid_geom])
        merged.loc[valid_geom, "SvAzimuthDegrees"] = az
        merged.loc[valid_geom, "SvElevationDegrees"] = el

    merged["MessageType"] = "Raw"
    merged["CodeType"] = code_type_from_signal(merged["SignalType"])
    merged["ChipsetElapsedRealtimeNanos"] = np.nan
    merged["utcTimeMillis"], gt_snap_ratio = snap_times_to_ground_truth(
        merged["utcTimeMillis"],
        gt_times,
        tolerance_ms=gt_snap_tolerance_ms,
    )
    merged["ArrivalTimeNanosSinceGpsEpoch"] = merged["utcTimeMillis"].astype("int64") * 1_000_000
    merged["ReceivedSvTimeNanos"] = merged.get("ReceivedSvTimeNanos", np.nan)
    for col in OUTPUT_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    matched_raw_ratio = float(merged["PseudorangeRateMetersPerSecond"].notna().mean())
    matched_wls_ratio = float(merged["WlsPositionXEcefMeters"].notna().mean())
    merged[OUTPUT_COLUMNS].to_csv(out_path, index=False)
    return {
        "trace": str(trace_dir),
        "status": "converted",
        "rows": int(len(merged)),
        "matched_raw_ratio": matched_raw_ratio,
        "matched_wls_ratio": matched_wls_ratio,
        "gt_snap_ratio": gt_snap_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/gsdc")
    parser.add_argument("--split-json", default="splits/gsdc_5splits.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    baseline_wls = load_wls_baseline(data_root / "2021" / "baseline_locations_train.csv")
    splits = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    route_groups = sorted({group for groups in splits.values() for group in groups if group.startswith("2021/")})
    trace_dirs = sorted(
        trace_dir
        for group in route_groups
        if (data_root / group).exists()
        for trace_dir in (data_root / group).iterdir()
        if trace_dir.is_dir() and (trace_dir / "ground_truth.csv").exists()
    )
    print(f"2021 traces: {len(trace_dirs)}")
    for index, trace_dir in enumerate(trace_dirs, start=1):
        out_path = trace_dir / "device_gnss.csv"
        if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
            print(f"[{index:03d}/{len(trace_dirs):03d}] exists {trace_dir}")
            continue
        record = convert_trace(trace_dir, baseline_wls, out_path)
        print(f"[{index:03d}/{len(trace_dirs):03d}] {record['status']} {trace_dir} rows={record.get('rows', 0)}", flush=True)


if __name__ == "__main__":
    main()
