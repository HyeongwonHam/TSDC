"""Evaluate WLS, the Doppler baselines, screened TDCP-FGO and TSDC-FGO on the GSDC
target-evaluation split, then print the main-text GSDC tables."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsdc import factors as F  # noqa: E402
from tsdc.data import expand_trace_dirs, load_ground_truth, load_route_groups, make_wls_state  # noqa: E402
from tsdc.fgo import FgoConfig, solve_track  # noqa: E402
from tsdc.inference import drop_adr_columns, load_model_bundle  # noqa: E402
from tsdc.metrics import (  # noqa: E402
    aggregate_runs,
    attach_sessions,
    collapse_sessions,
    evaluate_track,
    paired_group_statistics,
    session_mean_across_runs,
)


FGO = FgoConfig()
_OPTIONS: dict[str, Any] = {}
_BUNDLES: list[tuple[int, Any]] = []


def init_worker(options: dict[str, Any]) -> None:
    global _OPTIONS, _BUNDLES
    _OPTIONS = options
    _BUNDLES = [
        (seed, load_model_bundle(Path(path), force_cpu=not options["cuda"]))
        for seed, path in options["checkpoints"]
    ]


def evaluate_trace(trace_string: str) -> list[dict[str, Any]]:
    trace = Path(trace_string)
    raw = pd.read_csv(trace / "device_gnss.csv", low_memory=False)
    raw["utcTimeMillis"] = raw["utcTimeMillis"].astype(np.int64)
    wls = make_wls_state(raw)
    truth = load_ground_truth(trace)
    base = {
        "trace": str(trace),
        "route_group": str(trace.parent.relative_to(Path(_OPTIONS["data_root"]))),
        "device": trace.name,
    }
    rows: list[dict[str, Any]] = []

    def record(method: str, seed: int | None, prediction: pd.DataFrame, factor_count: int) -> None:
        p50, p95, score, epochs = evaluate_track(truth, prediction)
        rows.append(
            {**base, "method": method, "seed": seed, "p50_m": p50, "p95_m": p95, "score_m": score,
             "epochs": epochs, "factors": factor_count}
        )

    record("WLS", None, wls.rename(columns={"utcTimeMillis": "UnixTimeMillis"})[["UnixTimeMillis", "lat", "lon"]], 0)
    doppler_contexts = F.build_raw_doppler_contexts(raw, wls)
    for method, rate_mode, sigma_m in ((F.ENDPOINT, "endpoint", 5.0), (F.AVERAGE, "average", 1.0)):
        factors = F.raw_doppler_factors(doppler_contexts, rate_mode, sigma_m)
        record(method, None, solve_track(wls, factors, FGO)[0], len(factors))
    factors = F.screened_tdcp_factors(raw, wls)
    record(F.SCREENED_TDCP, None, solve_track(wls, factors, FGO)[0], len(factors))

    for seed, bundle in _BUNDLES:
        # TSDC is predicted from the table without ADR/carrier-phase columns. The ADR
        # table is walked only for the shared ADR-valid intervals; its network inputs must
        # be identical, so the same predictions serve both.
        contexts, features = F.build_pair_contexts(drop_adr_columns(raw), wls, bundle)
        adr_contexts, adr_features = F.build_pair_contexts(raw, wls, bundle)
        if not np.array_equal(np.asarray(features), np.asarray(adr_features)):
            raise RuntimeError(f"TSDC inputs changed when ADR columns were removed: {trace}")
        F.add_predictions(contexts, features, bundle)
        for context, adr_context in zip(contexts, adr_contexts):
            adr_context["mu_mps"], adr_context["scale_mps"] = context["mu_mps"], context["scale_mps"]
        method_factors = {**F.tsdc_mask_factors(contexts), **F.common_mask_factors(adr_contexts)}
        for method, factors in method_factors.items():
            record(method, seed, solve_track(wls, factors, FGO)[0], len(factors))
    return rows


def run_evaluation(args: argparse.Namespace, detail_path: Path) -> pd.DataFrame:
    data_root = Path(args.data_root)
    traces = expand_trace_dirs(data_root, load_route_groups(Path(args.split_json), args.split_name))
    if args.max_traces:
        traces = traces[: args.max_traces]
    checkpoints = [
        (int(seed), str(Path(args.checkpoint_dir) / args.checkpoint_pattern.format(seed=int(seed))))
        for seed in args.seeds.split(",")
    ]
    print(f"traces={len(traces)} checkpoints={[path for _, path in checkpoints]}", flush=True)
    options = {"data_root": str(data_root), "checkpoints": checkpoints, "cuda": args.cuda}
    rows: list[dict[str, Any]] = []
    with mp.get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(options,)) as pool:
        for index, trace_rows in enumerate(pool.imap_unordered(evaluate_trace, [str(t) for t in traces]), start=1):
            rows.extend(trace_rows)
            print(f"[{index:03d}/{len(traces):03d}] {trace_rows[0]['trace']}", flush=True)
    detail = pd.DataFrame(rows).sort_values(["trace", "method", "seed"]).reset_index(drop=True)
    detail.to_csv(detail_path, index=False)
    return detail


def show(title: str, frame: pd.DataFrame) -> None:
    print(f"\n== {title}")
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def summarize(detail: pd.DataFrame, mapping: pd.DataFrame, out_dir: Path, bootstrap_samples: int) -> None:
    sessions = collapse_sessions(attach_sessions(detail, mapping))
    seeds = sorted(int(seed) for seed in detail["seed"].dropna().unique())
    representative = 42 if 42 in seeds else seeds[0]

    single_run = sessions[sessions["seed"].isna() | (sessions["seed"] == representative)]
    table2 = aggregate_runs(single_run[single_run["method"].isin([F.ENDPOINT_COMMON, F.TDCP_COMMON, F.TSDC_COMMON])])
    averaging = aggregate_runs(single_run[single_run["method"].isin([F.ENDPOINT_TSDC_MASK, F.AVERAGE_TSDC_MASK, F.TSDC])])
    table3 = aggregate_runs(
        sessions[sessions["method"].isin(["WLS", F.ENDPOINT, F.AVERAGE, F.SCREENED_TDCP, F.TSDC])]
    )
    text = aggregate_runs(sessions[sessions["method"].isin([F.ENDPOINT_TSDC_MASK, F.TSDC_FIXED_SIGMA])])
    show(f"Table 2: identical satellite-epoch factor mask (seed {representative})", table2)
    show(f"Section 5.2: TSDC acceptance mask (seed {representative})", averaging)
    show(f"Table 3: 164 physical phone sessions ({len(seeds)} runs, +/- = std over runs)", table3)
    show("Section 5.3/5.4: endpoint Doppler and fixed-sigma TSDC on the TSDC mask", text)

    score = table3.set_index("method")
    tsdc = score.loc[F.TSDC]
    print(
        f"\nTSDC vs WLS: score {100 * (1 - tsdc.score_m / score.loc['WLS', 'score_m']):.2f}% lower, "
        f"P50 {100 * (1 - tsdc.p50_m / score.loc['WLS', 'p50_m']):.2f}%, "
        f"P95 {100 * (1 - tsdc.p95_m / score.loc['WLS', 'p95_m']):.2f}%; "
        f"vs Averaged Doppler FGO: {100 * (1 - tsdc.score_m / score.loc[F.AVERAGE, 'score_m']):.2f}% lower"
    )

    # Bootstrap generators of the original analyses, so that the published
    # intervals are reproduced exactly. The averaging-baseline interval was the
    # sixth draw of one generator (P50, P95 and score of the 77- and 68-feature
    # models); the five earlier draws are replayed here.
    tsdc_mean = session_mean_across_runs(sessions, F.TSDC)
    n_groups = tsdc_mean["route_leakage_group"].nunique()
    averaging_rng = np.random.default_rng(20260811)
    for _ in range(5):
        averaging_rng.integers(0, n_groups, size=(bootstrap_samples, n_groups))
    comparisons = [
        ("WLS", np.random.default_rng(20260722)),
        (F.ENDPOINT_TSDC_MASK, np.random.default_rng(20260723)),
        (F.SCREENED_TDCP, np.random.default_rng(20260724)),
        (F.AVERAGE, averaging_rng),
    ]
    table4 = pd.DataFrame(
        [
            {"comparator": name, **paired_group_statistics(tsdc_mean, session_mean_across_runs(sessions, name), bootstrap_samples, rng)}
            for name, rng in comparisons
        ]
    )
    show("Table 4: paired TSDC-FGO minus comparator (leakage-group bootstrap and Wilcoxon)", table4)

    out_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(out_dir / "physical_session_results.csv", index=False)
    pd.concat([table2, averaging, table3, text], ignore_index=True).to_csv(out_dir / "summary.csv", index=False)
    table4.to_csv(out_dir / "paired_statistics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/gsdc")
    parser.add_argument("--split-json", default=str(ROOT / "splits/gsdc_5splits.json"))
    parser.add_argument("--split-name", default="target_test")
    parser.add_argument("--mapping", default=str(ROOT / "splits/route_group_mapping.csv"))
    parser.add_argument("--checkpoint-dir", default="work/checkpoints")
    parser.add_argument("--checkpoint-pattern", default="tsdc_seed{seed}.pth")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--out-dir", default="work/eval")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-traces", type=int, default=0)
    # Multithreaded CPU and GPU inference can change mu and b in the last float32 bit
    # (below 1e-7 m in the trace scores in our checks).
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--summary-only", action="store_true", help="re-summarize an existing trace_detail.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "trace_detail.csv"
    detail = pd.read_csv(detail_path) if args.summary_only else run_evaluation(args, detail_path)
    summarize(detail, pd.read_csv(args.mapping), out_dir, args.bootstrap_samples)


if __name__ == "__main__":
    main()
