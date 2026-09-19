"""Train the five TSDC checkpoints (seeds 42-46) with the same training entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-dir", default="work/pairs")
    parser.add_argument("--checkpoint-dir", default="work/checkpoints")
    parser.add_argument("--train-dir", default="work/train")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    args = parser.parse_args()

    for seed in [int(value) for value in args.seeds.split(",")]:
        command = [
            sys.executable,
            "-m",
            "tsdc.train",
            "--pairs-dir",
            str(Path(args.pairs_dir).resolve()),
            "--checkpoint",
            str(Path(args.checkpoint_dir).resolve() / f"tsdc_seed{seed}.pth"),
            "--out-dir",
            str(Path(args.train_dir).resolve() / f"seed{seed}"),
            "--seed",
            str(seed),
        ]
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
