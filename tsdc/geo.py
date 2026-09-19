from __future__ import annotations

import numpy as np
import pandas as pd


WGS84_A = 6378137.0
WGS84_E2 = 0.00669437999014
EARTH_R = 6378137.0


def ecef_to_latlon(x: float, y: float, z: float) -> tuple[float, float]:
    b = np.sqrt(WGS84_A**2 * (1.0 - WGS84_E2))
    ep = np.sqrt((WGS84_A**2 - b**2) / b**2)
    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(WGS84_A * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(
        z + ep**2 * b * np.sin(th) ** 3,
        p - WGS84_E2 * WGS84_A * np.cos(th) ** 3,
    )
    return float(np.degrees(lat)), float(np.degrees(lon))


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    lat_deg, lon_deg = ecef_to_latlon(x, y, z)
    lat = np.radians(lat_deg)
    horizontal_radius = np.hypot(x, y)
    prime_vertical_radius = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    if abs(np.cos(lat)) > 1e-12:
        height = horizontal_radius / np.cos(lat) - prime_vertical_radius
    else:
        semi_minor = WGS84_A * np.sqrt(1.0 - WGS84_E2)
        height = abs(z) - semi_minor
    return lat_deg, lon_deg, float(height)


def ecef_to_enu_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.array(
        [
            [-np.sin(lon), np.cos(lon), 0.0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )


# Local east/north metres about the first nominal epoch; the FGO corrections are
# added in this frame and mapped back to latitude/longitude.
def latlon_to_local_m(
    lat: pd.Series, lon: pd.Series, lat0: float, lon0: float
) -> tuple[np.ndarray, np.ndarray]:
    north = (lat.to_numpy(dtype=float) - lat0) * np.pi / 180.0 * EARTH_R
    east = (
        (lon.to_numpy(dtype=float) - lon0)
        * np.pi
        / 180.0
        * EARTH_R
        * np.cos(np.radians(lat0))
    )
    return east, north


def local_m_to_latlon(
    east: np.ndarray, north: np.ndarray, lat0: float, lon0: float
) -> tuple[np.ndarray, np.ndarray]:
    lat = lat0 + (north / EARTH_R) * 180.0 / np.pi
    lon = lon0 + (east / (EARTH_R * np.cos(np.radians(lat0)))) * 180.0 / np.pi
    return lat, lon


def haversine_m(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 6371000.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
