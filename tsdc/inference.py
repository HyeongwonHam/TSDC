from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tsdc.features import standardize
from tsdc.model import TSDCNet


# Output gates selected on source validation (Online Resource 1, Section S2.1).
MAX_ABS_MU_MPS = 3.0
MAX_SCALE_MPS = 1.0

FORBIDDEN_INFERENCE_FEATURES = {
    "AccumulatedDeltaRangeMeters",
    "AccumulatedDeltaRangeState",
    "CarrierCycles",
    "CarrierPhase",
    "CarrierPhaseUncertainty",
    "TDCPRate",
    "ADRLabel",
}


def validate_inference_feature_names(feature_names: list[str]) -> None:
    forbidden = {name.lower() for name in FORBIDDEN_INFERENCE_FEATURES}
    offending = [name for name in feature_names if name.lower() in forbidden]
    if offending:
        raise ValueError(f"Forbidden carrier-phase/ADR inference features: {offending}")


# ADR-free inference: TSDC predictions are computed from a table with every
# ADR and carrier-phase column removed.
def drop_adr_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    adr_like_cols = [
        col
        for col in df_raw.columns
        if col.startswith("AccumulatedDeltaRange")
        or col in {"CarrierCycles", "CarrierPhase", "CarrierPhaseUncertainty"}
    ]
    return df_raw.drop(columns=adr_like_cols)


@dataclass
class ModelBundle:
    model: TSDCNet
    device: torch.device
    mean: np.ndarray
    std: np.ndarray
    signal_types: list[str]
    code_types: list[str]
    batch_size: int
    feature_indices: np.ndarray | None
    feature_names: list[str]


def load_model_bundle(checkpoint_path: Path, batch_size: int = 65536, force_cpu: bool = False) -> ModelBundle:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    spec = ckpt["feature_spec"]
    validate_inference_feature_names(list(spec["feature_names"]))
    if spec.get("target_mode", "right_endpoint") != "right_endpoint":
        raise ValueError(f"{checkpoint_path}: only epoch-k (right-endpoint) target checkpoints are supported")
    # Standardization statistics are read from the checkpoint, never recomputed.
    mean = np.asarray(spec["mean"], dtype=np.float32)
    std = np.asarray(spec["std"], dtype=np.float32)
    model = TSDCNet(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        dropout=float(ckpt["dropout"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)
    model.eval()
    feature_indices_raw = spec.get("input_feature_indices")
    feature_indices = None if feature_indices_raw is None else np.asarray(feature_indices_raw, dtype=np.int64)
    return ModelBundle(
        model=model,
        device=device,
        mean=mean,
        std=std,
        signal_types=list(spec["signal_types"]),
        code_types=list(spec["code_types"]),
        batch_size=batch_size,
        feature_indices=feature_indices,
        feature_names=list(spec["feature_names"]),
    )


def predict_residuals(features: list[list[float]], bundle: ModelBundle) -> tuple[np.ndarray, np.ndarray]:
    if len(features) == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    x_raw = np.asarray(features, dtype=np.float32)
    if bundle.feature_indices is not None:
        x_raw = x_raw[:, bundle.feature_indices]
    x = standardize(x_raw, bundle.mean, bundle.std)
    preds: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), bundle.batch_size):
            xb = torch.from_numpy(x[start : start + bundle.batch_size]).to(bundle.device)
            out = bundle.model(xb)
            preds.append(out[:, 0].detach().cpu().numpy())
            scales.append(torch.exp(out[:, 1]).detach().cpu().numpy())
    return np.concatenate(preds), np.concatenate(scales)


def output_gate(mu_mps: float, scale_mps: float, max_abs_mu_mps: float, max_scale_mps: float) -> bool:
    return abs(mu_mps) <= max_abs_mu_mps and scale_mps <= max_scale_mps
