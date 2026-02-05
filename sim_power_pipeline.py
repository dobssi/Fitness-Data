# File: sim_power_pipeline.py
# Purpose: Shared utilities for sim power pipeline (v37) — robust RE model fit with NaN-safe features.
# Version: v37_final (2026-01-16)

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

PIPELINE_VERSION = "v37_final"


# ----------------------------
# Small utilities
# ----------------------------

def rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    """Edge-padded rolling median."""
    x = x.astype(float)
    if win is None or win <= 1 or len(x) == 0:
        return x
    win = int(win)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty(len(x), dtype=float)
    for i in range(len(x)):
        out[i] = float(np.median(xp[i:i + win]))
    return out


def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return a / np.maximum(b, eps)


def _nan_sanitize(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------
# Weather helpers
# ----------------------------

def air_density_kg_m3(temp_c: float, pressure_hpa: float = 1013.25, rh_frac: float = 0.60) -> float:
    """Moist-air density approximation."""
    T = float(temp_c) + 273.15
    p = float(pressure_hpa) * 100.0
    rh = max(0.0, min(1.0, float(rh_frac)))

    es_hpa = 6.112 * math.exp((17.62 * float(temp_c)) / (243.12 + float(temp_c)))
    e_hpa = rh * es_hpa
    e = e_hpa * 100.0  # Pa

    Rd = 287.05
    Rv = 461.495
    pd = p - e
    rho = pd / (Rd * T) + e / (Rv * T)
    return float(rho)


def wind_along_track(wind_speed_mps: float, wind_dir_deg_from: float, heading_deg_to: np.ndarray) -> np.ndarray:
    """Return +tailwind/-headwind along runner heading."""
    if heading_deg_to is None or len(heading_deg_to) == 0:
        return np.zeros(0, dtype=float)
    w_to = (float(wind_dir_deg_from) + 180.0) % 360.0
    rel = np.deg2rad(((w_to - heading_deg_to + 540.0) % 360.0) - 180.0)
    return float(wind_speed_mps) * np.cos(rel)


# ----------------------------
# RE model (log-linear ridge)
# ----------------------------

@dataclass
class REModel:
    mu: Dict[str, float]
    sd: Dict[str, float]
    coef: Dict[str, float]  # includes bias
    re_min: float = 0.50
    re_max: float = 1.20

    def _z(self, name: str, x: np.ndarray) -> np.ndarray:
        return (x - self.mu[name]) / max(self.sd[name], 1e-9)

    def predict_re(self, feats: Dict[str, np.ndarray]) -> np.ndarray:
        y = np.full_like(next(iter(feats.values())), self.coef.get("bias", 0.0), dtype=float)
        for k, c in self.coef.items():
            if k == "bias":
                continue
            y += c * self._z(k, feats[k])
        re = np.exp(y)
        return clamp(re, self.re_min, self.re_max)

    def to_json(self) -> str:
        return json.dumps({
            "version": PIPELINE_VERSION,
            "mu": self.mu,
            "sd": self.sd,
            "coef": self.coef,
            "re_min": self.re_min,
            "re_max": self.re_max,
        }, indent=2)

    @staticmethod
    def from_json(s: str) -> "REModel":
        d = json.loads(s)
        return REModel(
            mu=d["mu"],
            sd=d["sd"],
            coef=d["coef"],
            re_min=float(d.get("re_min", 0.50)),
            re_max=float(d.get("re_max", 1.20)),
        )


def build_features(speed_mps: np.ndarray, grade: np.ndarray, wind_along_mps: np.ndarray, rho: np.ndarray) -> Dict[str, np.ndarray]:
    v = np.asarray(speed_mps, dtype=float)
    g = np.asarray(grade, dtype=float)
    wa = np.asarray(wind_along_mps, dtype=float)
    r = np.asarray(rho, dtype=float)
    return {
        "v": v,
        "v2": v * v,
        "g": g,
        "abs_g": np.abs(g),
        "wa": wa,
        "wa_v": wa * v,
        "g_v": g * v,
        "rho": r,
    }


def fit_re_model(
    feats_list: list[Dict[str, np.ndarray]],
    re_obs_list: list[np.ndarray],
    alpha: float = 5.0,
    re_bounds: Tuple[float, float] = (0.50, 1.20),
    min_samples: int = 200,
) -> REModel:
    """Fit log(RE) = bias + X beta with ridge regularisation.

    Robust to NaNs/Infs by sanitising features to 0.0 before masking.
    """
    keys = ["v", "v2", "g", "abs_g", "wa", "wa_v", "g_v", "rho"]

    Xs, ys = [], []
    runs_seen = 0
    runs_used = 0
    rejected_too_short = 0
    rejected_bad_arrays = 0

    for feats, re_obs in zip(feats_list, re_obs_list):
        runs_seen += 1
        try:
            re = clamp(_nan_sanitize(re_obs), re_bounds[0], re_bounds[1])
            y = np.log(re)

            feats_s = {k: _nan_sanitize(feats[k]) for k in keys}

            mask = np.isfinite(y)
            for k in keys:
                mask &= np.isfinite(feats_s[k])

            n = int(mask.sum())
            if n < int(min_samples):
                rejected_too_short += 1
                continue

            Xs.append(np.column_stack([feats_s[k][mask] for k in keys]))
            ys.append(y[mask])
            runs_used += 1
        except Exception:
            rejected_bad_arrays += 1
            continue

    if not Xs:
        raise ValueError(
            f"fit_re_model({PIPELINE_VERSION}): no usable training samples (Xs is empty).\n"
            f"  runs_seen: {runs_seen}\n"
            f"  runs_used: {runs_used}\n"
            f"  rejected_too_short(<{min_samples} samples): {rejected_too_short}\n"
            f"  rejected_bad_arrays: {rejected_bad_arrays}\n"
            f"Loaded from: {__file__}"
        )

    X = np.vstack(Xs)
    y = np.concatenate(ys)

    mu = {k: float(X[:, i].mean()) for i, k in enumerate(keys)}
    sd = {k: float(X[:, i].std(ddof=0) + 1e-9) for i, k in enumerate(keys)}
    Xz = np.column_stack([(X[:, i] - mu[k]) / sd[k] for i, k in enumerate(keys)])

    Xb = np.column_stack([np.ones(len(y)), Xz])
    I = np.eye(Xb.shape[1])
    I[0, 0] = 0.0  # don't regularize bias
    beta = np.linalg.solve(Xb.T @ Xb + alpha * I, Xb.T @ y)

    coef = {"bias": float(beta[0])}
    for i, k in enumerate(keys, start=1):
        coef[k] = float(beta[i])

    return REModel(mu=mu, sd=sd, coef=coef, re_min=re_bounds[0], re_max=re_bounds[1])


def simulate_power_series(
    model: REModel,
    speed_mps: np.ndarray,
    grade: np.ndarray,
    wind_along_mps: np.ndarray,
    rho: np.ndarray,
    mass_kg: float,
    smooth_win: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    feats = build_features(speed_mps, grade, wind_along_mps, rho)
    feats = {k: _nan_sanitize(v) for k, v in feats.items()}
    re_hat = model.predict_re(feats)
    p = safe_div(_nan_sanitize(speed_mps) * float(mass_kg), re_hat)
    p = np.where(_nan_sanitize(speed_mps) < 0.3, 0.0, p)
    p = clamp(p, 0.0, 2000.0)
    p = rolling_median(p, smooth_win)
    return p, re_hat
