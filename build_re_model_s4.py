# File: build_re_model_s4.py
# Purpose: Step A — Fit RE(t) model using S4-era Stryd runs (wind disabled) with clear diagnostics.
# Version: v37_final (2026-01-16)

from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np
import pandas as pd

import sim_power_pipeline as spp
from sim_power_pipeline import air_density_kg_m3, build_features, fit_re_model, rolling_median


def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit RE(t) model using Stryd runs normalized to S4 power scale (S4-only)")
    p.add_argument("--master", required=True)
    p.add_argument("--persec-cache-dir", required=True)
    p.add_argument("--out-model", required=True)
    p.add_argument("--mass-kg", type=float, default=76.0)
    p.add_argument("--alpha", type=float, default=5.0)
    p.add_argument("--re-min", type=float, default=0.50)
    p.add_argument("--re-max", type=float, default=1.20)
    p.add_argument("--min-samples", type=int, default=200)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"[StepA] sim_power_pipeline version: {getattr(spp, 'PIPELINE_VERSION', 'unknown')}")
    print(f"[StepA] sim_power_pipeline loaded from: {spp.__file__}")

    df = pd.read_excel(args.master)

    era = df.get("calibration_era_id", pd.Series([""] * len(df))).astype(str).str.lower()
    has_power = pd.to_numeric(df.get("avg_power_w"), errors="coerce").notna()

    is_s4 = has_power & era.str.contains("s4")

    # Map cache files by stem
    cache_files = {}
    if not os.path.isdir(args.persec_cache_dir):
        raise RuntimeError(f"Cache dir not found: {args.persec_cache_dir}")
    for fp in os.listdir(args.persec_cache_dir):
        if fp.lower().endswith(".npz"):
            cache_files[fp[:-4]] = os.path.join(args.persec_cache_dir, fp)

    temps = pd.to_numeric(df.get("avg_temp_c"), errors="coerce")
    adj = pd.to_numeric(df.get("power_adjuster_to_S4"), errors="coerce").fillna(1.0)

    feats_list = []
    re_list = []

    c_missing_cache = 0
    c_missing_stream = 0
    c_short_mask = 0
    used = 0

    idxs = np.where(is_s4.to_numpy())[0]
    print(f"[StepA] Candidate S4 runs: {len(idxs)}")

    for i in idxs:
        run_id = str(df.at[i, "file"]).strip() if "file" in df.columns else ""
        if not run_id:
            continue
        key = os.path.splitext(os.path.basename(run_id))[0]
        npz_path = cache_files.get(key)
        if not npz_path:
            c_missing_cache += 1
            continue

        z = np.load(npz_path)
        if "power_w" not in z.files or "speed_mps" not in z.files:
            c_missing_stream += 1
            continue

        v = z["speed_mps"].astype(float)
        p = z["power_w"].astype(float) * float(adj.iloc[i])

        grade = z["grade"].astype(float) if "grade" in z.files else np.zeros_like(v)

        # Smoothing consistent with Step B expectations
        v = rolling_median(v, 5)
        p = rolling_median(p, 15)
        grade = rolling_median(grade, 9)

        # Per-run rho; wind disabled for training
        t = float(temps.iloc[i]) if np.isfinite(temps.iloc[i]) else 10.0
        rho = air_density_kg_m3(t)
        rho_arr = np.full_like(v, rho, dtype=float)
        wa = np.zeros_like(v)

        # Mask: moving + reasonable power
        mask = (v >= 0.8) & (p >= 50) & np.isfinite(v) & np.isfinite(p)
        if mask.sum() < 600:
            c_short_mask += 1
            continue

        re_obs = (v[mask] * float(args.mass_kg)) / p[mask]
        re_obs = np.clip(re_obs, args.re_min, args.re_max)

        feats = build_features(v[mask], grade[mask], wa[mask], rho_arr[mask])
        feats_list.append(feats)
        re_list.append(re_obs)
        used += 1
        if used % 200 == 0:
            print(f"...collected {used} runs")

    print("[StepA] Collection summary:")
    print(f"  collected_runs: {used}")
    print(f"  missing_cache: {c_missing_cache}")
    print(f"  missing_streams: {c_missing_stream}")
    print(f"  rejected_short_mask(<600): {c_short_mask}")

    if used < 50:
        raise RuntimeError(f"Too few runs used for RE model fit: {used}")

    model = fit_re_model(
        feats_list,
        re_list,
        alpha=float(args.alpha),
        re_bounds=(args.re_min, args.re_max),
        min_samples=int(args.min_samples),
    )
    js = model.to_json()
    with open(args.out_model, "w", encoding="utf-8") as f:
        f.write(js)

    mid = f"simRE_s4_{_hash_text(js)}"
    print(f"Wrote model: {args.out_model}")
    print(f"Model id: {mid}")
    print(f"Runs used: {used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
