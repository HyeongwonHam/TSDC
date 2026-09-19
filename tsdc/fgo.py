from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tsdc.geo import latlon_to_local_m, local_m_to_latlon


@dataclass(frozen=True)
class FgoConfig:
    prior_sigma_m: float = 5.0  # horizontal prior from the nominal trajectory
    huber_delta: float = 1.5
    clock_anchor_sigma_m: float = 1.0  # first clock state only (gauge)
    clock_weak_sigma_m: float = 1e6  # all clock states
    irls_iters: int = 4


@dataclass
class TdcpFactor:
    i0: int
    i1: int
    c0: np.ndarray
    c1: np.ndarray
    y: float
    sigma: float


# State x_k = [delta_e, delta_n, delta_c]: corrections to the nominal trajectory
# and the receiver clock. los*_enu are satellite-to-receiver unit vectors in ENU.
def make_direct_factor_terms(
    los0_enu: np.ndarray,
    los1_enu: np.ndarray,
    measured_range_change_m: float,
    nominal_range_change_m: float,
    satellite_clock_change_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    c0 = np.concatenate((-np.asarray(los0_enu[:2]), [-1.0]))
    c1 = np.concatenate((np.asarray(los1_enu[:2]), [1.0]))
    y = measured_range_change_m + satellite_clock_change_m - nominal_range_change_m
    return c0, c1, float(y)


def block_tridiagonal_solve(diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray, damping: float = 1e-9) -> np.ndarray:
    n = len(diag)
    d = diag.astype(float).copy()
    u = upper.astype(float).copy()
    b = rhs.astype(float).copy()

    state_dim = diag.shape[1]
    eye = np.eye(state_dim)
    c_prime = np.zeros_like(u)
    d_prime = np.zeros((n, state_dim), dtype=float)

    first = d[0] + eye * damping
    if n > 1:
        c_prime[0] = np.linalg.solve(first, u[0])
    d_prime[0] = np.linalg.solve(first, b[0])

    for i in range(1, n):
        lower = u[i - 1].T
        den = d[i] - lower @ c_prime[i - 1] + eye * damping
        if i < n - 1:
            c_prime[i] = np.linalg.solve(den, u[i])
        d_prime[i] = np.linalg.solve(den, b[i] - lower @ d_prime[i - 1])

    x = np.zeros((n, state_dim), dtype=float)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] @ x[i + 1]
    return x


def add_weighted_factor(
    diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray, factor: TdcpFactor, robust_weight: float
) -> None:
    w = robust_weight / max(float(factor.sigma), 1e-6) ** 2
    i0 = factor.i0
    i1 = factor.i1
    diag[i0] += w * np.outer(factor.c0, factor.c0)
    diag[i1] += w * np.outer(factor.c1, factor.c1)
    upper[i0] += w * np.outer(factor.c0, factor.c1)
    rhs[i0] += w * factor.c0 * factor.y
    rhs[i1] += w * factor.c1 * factor.y


# Huber IRLS on the normal equations. Every factor links epochs k-1 and k only,
# so the system is block tridiagonal in the per-epoch 3x3 states.
def solve_direct_huber(n: int, factors: list[TdcpFactor], config: FgoConfig) -> tuple[np.ndarray, float]:
    if not factors:
        return np.zeros((n, 3), dtype=float), 0.0

    state_dim = len(factors[0].c0)
    position_dimensions = state_dim - 1
    clock_indexes = range(position_dimensions, state_dim)
    x = np.zeros((n, state_dim), dtype=float)
    mean_robust_weight = 1.0
    for iteration in range(max(config.irls_iters, 1)):
        diag = np.zeros((n, state_dim, state_dim), dtype=float)
        upper = np.zeros((max(n - 1, 0), state_dim, state_dim), dtype=float)
        rhs = np.zeros((n, state_dim), dtype=float)

        pos_w = 1.0 / max(config.prior_sigma_m, 1e-6) ** 2
        for position_index in range(position_dimensions):
            diag[:, position_index, position_index] += pos_w
        if np.isfinite(config.clock_weak_sigma_m) and config.clock_weak_sigma_m > 0:
            for clock_index in clock_indexes:
                diag[:, clock_index, clock_index] += 1.0 / config.clock_weak_sigma_m**2
        for clock_index in clock_indexes:
            diag[0, clock_index, clock_index] += 1.0 / max(config.clock_anchor_sigma_m, 1e-6) ** 2

        weights = []
        for factor in factors:
            robust_weight = 1.0
            if iteration > 0 and config.huber_delta > 0:
                residual = float(factor.c0 @ x[factor.i0] + factor.c1 @ x[factor.i1] - factor.y)
                z = abs(residual) / max(factor.sigma, 1e-6)
                if z > config.huber_delta:
                    robust_weight = config.huber_delta / z
            weights.append(robust_weight)
            add_weighted_factor(diag, upper, rhs, factor, robust_weight)

        try:
            x_new = block_tridiagonal_solve(diag, upper, rhs)
        except np.linalg.LinAlgError:
            return np.zeros((n, state_dim), dtype=float), 0.0
        mean_robust_weight = float(np.mean(weights)) if weights else 0.0
        if np.mean(np.linalg.norm(x_new - x, axis=1)) < 1e-4:
            x = x_new
            break
        x = x_new
    return x, mean_robust_weight


def solve_track(wls: pd.DataFrame, factors: list[TdcpFactor], config: FgoConfig) -> tuple[pd.DataFrame, float]:
    delta, mean_weight = solve_direct_huber(len(wls), factors, config)
    lat0 = float(wls["lat"].iloc[0])
    lon0 = float(wls["lon"].iloc[0])
    east, north = latlon_to_local_m(wls["lat"], wls["lon"], lat0, lon0)
    latitude, longitude = local_m_to_latlon(east + delta[:, 0], north + delta[:, 1], lat0, lon0)
    prediction = pd.DataFrame(
        {
            "UnixTimeMillis": wls["utcTimeMillis"].to_numpy(dtype=np.int64),
            "lat": latitude,
            "lon": longitude,
        }
    )
    return prediction, mean_weight
