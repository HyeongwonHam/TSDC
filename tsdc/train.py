from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from tsdc.features import WLS_MOTION_FEATURES, compute_scaler, standardize, tsdc_feature_indices
from tsdc.model import TSDCNet, laplace_nll


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y[:, None]))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def validation_metrics(prediction: np.ndarray, target: np.ndarray, scale: np.ndarray, losses: list[float]) -> dict[str, float]:
    error = prediction - target
    absolute_error = np.abs(error)
    return {
        "loss": float(np.mean(losses)),
        "mae_mps": float(np.mean(absolute_error)),
        "rmse_mps": float(np.sqrt(np.mean(error**2))),
        "p50_abs_mps": float(np.percentile(absolute_error, 50)),
        "p95_abs_mps": float(np.percentile(absolute_error, 95)),
        "zero_mae_mps": float(np.mean(np.abs(target))),
        "zero_p95_abs_mps": float(np.percentile(np.abs(target), 95)),
        "scale_mean_mps": float(np.mean(scale)),
        "scale_p50_mps": float(np.percentile(scale, 50)),
    }


def train_epoch(model: TSDCNet, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    losses: list[float] = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = laplace_nll(model(xb), yb)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate_loader(model: TSDCNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds, tgts, scales, losses = [], [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(xb)
        losses.append(float(laplace_nll(out, yb).item()))
        preds.append(out[:, 0].detach().cpu().numpy())
        scales.append(torch.exp(out[:, 1]).detach().cpu().numpy())
        tgts.append(yb[:, 0].detach().cpu().numpy())
    return validation_metrics(np.concatenate(preds), np.concatenate(tgts), np.concatenate(scales), losses)


# The paper checkpoints were trained with the whole cache resident on one GPU
# (num_workers=0); mini-batch order comes from torch.randperm on that device.
def train_epoch_gpu_resident(
    model: TSDCNet, x: torch.Tensor, y: torch.Tensor, batch_size: int, optimizer: torch.optim.Optimizer
) -> float:
    model.train()
    order = torch.randperm(len(y), device=x.device)
    losses: list[float] = []
    for start in range(0, len(y), batch_size):
        indices = order[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss = laplace_nll(model(x[indices]), y[indices])
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate_gpu_resident(model: TSDCNet, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> dict[str, float]:
    model.eval()
    preds, tgts, scales, losses = [], [], [], []
    for start in range(0, len(y), batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        out = model(xb)
        losses.append(float(laplace_nll(out, yb).item()))
        preds.append(out[:, 0].detach().cpu().numpy())
        scales.append(torch.exp(out[:, 1]).detach().cpu().numpy())
        tgts.append(yb[:, 0].detach().cpu().numpy())
    return validation_metrics(np.concatenate(preds), np.concatenate(tgts), np.concatenate(scales), losses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one TSDC checkpoint on the source-training pairs.")
    parser.add_argument("--pairs-dir", default="work/pairs")
    parser.add_argument("--checkpoint", default="work/checkpoints/tsdc_seed42.pth")
    parser.add_argument("--out-dir", default="work/train/seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--cpu", action="store_true", help="DataLoader training on CPU (not the paper path)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    pairs_dir = Path(args.pairs_dir)
    out_dir = Path(args.out_dir)
    checkpoint_path = Path(args.checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    base_spec: dict[str, Any] = json.loads((pairs_dir / "feature_spec.json").read_text(encoding="utf-8"))
    base_names = list(base_spec["feature_names"])
    selected_indices = tsdc_feature_indices(base_names)
    selected_names = [base_names[index] for index in selected_indices]

    train_data = np.load(pairs_dir / "source_train_pairs.npz")
    val_data = np.load(pairs_dir / "source_val_pairs.npz")
    if train_data["x"].shape[1] != len(base_names) or val_data["x"].shape[1] != len(base_names):
        raise SystemExit("Cached feature dimensions do not match feature_spec.json")
    x_train_raw = train_data["x"][:, selected_indices]
    x_val_raw = val_data["x"][:, selected_indices]
    y_train = np.asarray(train_data["y"], dtype=np.float32)
    y_val = np.asarray(val_data["y"], dtype=np.float32)

    mean, std = compute_scaler(x_train_raw)
    x_train = standardize(x_train_raw, mean, std)
    x_val = standardize(x_val_raw, mean, std)
    feature_spec = {
        **base_spec,
        "feature_names": selected_names,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "input_feature_indices": selected_indices,
        "base_feature_dim": len(base_names),
        "excluded_features": list(WLS_MOTION_FEATURES),
        "ablation_name": "no_wls_motion",
    }
    save_json(out_dir / "feature_spec.json", feature_spec)

    use_gpu_resident = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_gpu_resident else "cpu")
    if not use_gpu_resident:
        print("CUDA not used: DataLoader training on CPU; results will not match the paper checkpoints bit for bit.")
    print(f"device={device} train/val={len(y_train)}/{len(y_val)} features={len(selected_names)}", flush=True)
    model = TSDCNet(input_dim=len(selected_names), hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    if use_gpu_resident:
        x_train_device = torch.from_numpy(x_train).to(device)
        y_train_device = torch.from_numpy(y_train[:, None]).to(device)
        x_val_device = torch.from_numpy(x_val).to(device)
        y_val_device = torch.from_numpy(y_val[:, None]).to(device)
    else:
        train_loader = make_loader(x_train, y_train, args.batch_size, True, 0)
        val_loader = make_loader(x_val, y_val, args.batch_size, False, 0)

    history: list[dict[str, float | int]] = []
    best_mae = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        if use_gpu_resident:
            train_loss = train_epoch_gpu_resident(model, x_train_device, y_train_device, args.batch_size, optimizer)
            metrics = evaluate_gpu_resident(model, x_val_device, y_val_device, args.batch_size)
        else:
            train_loss = train_epoch(model, train_loader, optimizer, device)
            metrics = evaluate_loader(model, val_loader, device)
        scheduler.step(metrics["mae_mps"])
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics, "lr": float(optimizer.param_groups[0]["lr"])})
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        print(
            f"epoch={epoch:03d} loss={train_loss:.4f} val_mae={metrics['mae_mps']:.5f} "
            f"val_p95={metrics['p95_abs_mps']:.5f} scale50={metrics['scale_p50_mps']:.5f}",
            flush=True,
        )
        # Checkpoint selection: minimum source-validation MAE of mu.
        if metrics["mae_mps"] < best_mae:
            best_mae = metrics["mae_mps"]
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": len(selected_names),
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "feature_spec": feature_spec,
                    "epoch": epoch,
                    "best_val_mae_mps": best_mae,
                    "val_metrics": metrics,
                    "args": vars(args),
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    best = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    summary = {
        "checkpoint": str(checkpoint_path),
        "seed": args.seed,
        "best_epoch": int(best["epoch"]),
        "best_val_mae_mps": float(best["best_val_mae_mps"]),
        "best_val_metrics": best["val_metrics"],
        "train_examples": int(len(y_train)),
        "val_examples": int(len(y_val)),
        "feature_dim": len(selected_names),
    }
    save_json(out_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
