"""Extract source-training and source-validation pairs with TDCP-referenced targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsdc.data import extract_training_pairs, load_route_groups, load_trace_manifest  # noqa: E402
from tsdc.features import base_feature_names, collect_categories, compute_scaler  # noqa: E402


def extract_split(trace_dirs: list[Path], signal_types: list[str], code_types: list[str], args: argparse.Namespace):
    xs, ys, rows = [], [], []
    for index, trace_dir in enumerate(trace_dirs, start=1):
        x, y, stats = extract_training_pairs(
            trace_dir, signal_types, code_types, args.min_dt_s, args.max_dt_s, args.label_gate_mps
        )
        if len(y):
            xs.append(x)
            ys.append(y)
        rows.append({"trace": str(trace_dir), **stats})
        print(f"[{index:03d}/{len(trace_dirs):03d}] {trace_dir} accepted={stats['accepted']}", flush=True)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/gsdc")
    parser.add_argument("--split-json", default=str(ROOT / "splits/gsdc_5splits.json"))
    parser.add_argument("--trace-manifest", default=str(ROOT / "splits/source_trace_manifest.csv"))
    parser.add_argument("--out-dir", default="work/pairs")
    parser.add_argument("--min-dt-s", type=float, default=0.8)
    parser.add_argument("--max-dt-s", type=float, default=1.2)
    parser.add_argument("--label-gate-mps", type=float, default=1.5)
    parser.add_argument("--max-categories", type=int, default=32)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dirs = {
        split: load_trace_manifest(
            Path(args.trace_manifest), data_root, split, load_route_groups(Path(args.split_json), split)
        )
        for split in ("source_train", "source_val")
    }
    print({split: len(traces) for split, traces in trace_dirs.items()})

    # Signal and code categories come from source-training traces only.
    signal_types, code_types = collect_categories(trace_dirs["source_train"], args.max_categories)
    feature_names = base_feature_names(signal_types, code_types)
    (out_dir / "categories.json").write_text(
        json.dumps({"signal_types": signal_types, "code_types": code_types}, indent=2), encoding="utf-8"
    )
    print(f"base features: {len(feature_names)} signal_types={len(signal_types)} code_types={len(code_types)}")

    cache = {}
    for split, traces in trace_dirs.items():
        x, y, stats = extract_split(traces, signal_types, code_types, args)
        np.savez_compressed(out_dir / f"{split}_pairs.npz", x=x, y=y)
        stats.to_csv(out_dir / f"{split}_trace_stats.csv", index=False)
        cache[split] = (x, y, stats)
        print(
            f"{split}: pairs after screens={int(stats[['accepted', 'skip_label_gate']].to_numpy().sum())} "
            f"after residual gate={len(y)}"
        )

    mean, std = compute_scaler(cache["source_train"][0])
    feature_spec = {
        "feature_names": feature_names,
        "signal_types": signal_types,
        "code_types": code_types,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "label_gate_mps": args.label_gate_mps,
        "min_dt_s": args.min_dt_s,
        "max_dt_s": args.max_dt_s,
    }
    (out_dir / "feature_spec.json").write_text(json.dumps(feature_spec, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
