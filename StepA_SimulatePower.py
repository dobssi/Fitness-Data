# File: StepA_SimulatePower_v50.py
# Purpose: Step A - Simulate pre-Stryd power (S4-tuned) using RE model
# Date: 2026-02-06
#
# v50 changes:
#   - Fast cache index via build_cache_index_fast() (~30s -> <1s startup)
#
# v45 changes:
#   - Version bump (see v45_handover.md for details)
#
# v44 changes:
#   - Updated for data-driven Stryd era detection (serial-based)
#   - No functional changes to power simulation (era detection is in rebuild step)
#
# v41 changes:
#   - Renamed from simulate_pre_stryd_s4_v38.py for consistent naming
#   - HR correction removed from rebuild; lives ONLY here in Step A.
#   - Implement HR correction with:
#       * hard floor 300 s
#       * HR floor 120 bpm for ratio inference
#       * explicit standing/walking rest exclusion (speed <0.3 m/s OR power <2.5 w/kg)
#       * indoor runs (from weather_overrides.csv) are never corrected
#   - RF dead time uses (power <2.5 w/kg) AND/OR (speed <0.3 m/s) criterion (OR).
#   - RF + nPower_HR computed here for pre_stryd and overwrite master columns.

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import hashlib
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from sim_power_pipeline import REModel, air_density_kg_m3, simulate_power_series


# ----------------------------
# Small helpers
# ----------------------------

def _coerce_scalar(val):
    if val is None:
        return np.nan
    if isinstance(val, np.generic):
        try:
            return val.item()
        except Exception:
            pass
    if isinstance(val, str):
        try:
            return float(val)
        except Exception:
            return val
    return val


def _roll_med(a: np.ndarray, win: int) -> np.ndarray:
    try:
        import pandas as _pd

        s = _pd.Series(a)
        return s.rolling(int(win), center=True, min_periods=1).median().to_numpy(dtype=float)
    except Exception:
        return np.asarray(a, float)


def _ensure_1d(a, dtype=float) -> np.ndarray:
    return np.asarray(a, dtype=dtype).reshape(-1)


def _hash_file(path: str) -> str:
    b = open(path, "rb").read()
    return hashlib.sha1(b).hexdigest()[:8]


# ----------------------------
# Cache indexing
# ----------------------------

def _build_cache_index(cache_dir: str) -> Tuple[List[int], List[str]]:
    """Build sorted (keys, paths) for bisect cache lookup.
    v50: Uses _cache_index.json for fast startup instead of opening all .npz files."""
    try:
        from rebuild_from_fit_zip import build_cache_index_fast
        return build_cache_index_fast(cache_dir)
    except ImportError:
        pass
    # Fallback: scan .npz files directly
    items: List[Tuple[int, str]] = []
    for fn in os.listdir(cache_dir):
        if not fn.lower().endswith(".npz"):
            continue
        fp = os.path.join(cache_dir, fn)
        try:
            z = np.load(fp, allow_pickle=True)
            ts = z.get("ts", None)
            if ts is None:
                continue
            ts0 = float(_ensure_1d(ts)[0])
            items.append((int(round(ts0)), fp))
        except Exception:
            continue
    items.sort(key=lambda x: x[0])
    return [k for k, _ in items], [p for _, p in items]


def _find_cache(keys: List[int], paths: List[str], epoch_s: float, tol_s: int) -> str | None:
    if not keys:
        return None
    k = int(round(float(epoch_s)))
    j = bisect.bisect_left(keys, k)
    best = None
    best_dt = None
    for idx in (j - 1, j, j + 1):
        if 0 <= idx < len(keys):
            dt = abs(keys[idx] - k)
            if dt <= tol_s and (best_dt is None or dt < best_dt):
                best_dt = dt
                best = paths[idx]
    return best


# ----------------------------
# HR correction (Step B only)
# ----------------------------

def _moving_mask(v: np.ndarray, p: np.ndarray, mass_kg: float) -> np.ndarray:
    """Exclude standing/walking rests: moving if speed>=0.3 and power>=2.5 w/kg."""
    v = np.asarray(v, float)
    p = np.asarray(p, float)
    wkg = p / max(float(mass_kg), 1e-9)
    return np.isfinite(v) & np.isfinite(p) & (v >= 0.3) & (wkg >= 2.5)


def correct_hr_stepb(
    t_s: np.ndarray,
    power_w: np.ndarray,
    hr_bpm: np.ndarray,
    speed_mps: np.ndarray,
    *,
    mass_kg: float,
    hard_floor_s: float = 300.0,
    hr_floor_bpm: float = 120.0,
) -> tuple[np.ndarray, bool, float | None, str | None]:
    """Detect and correct gross HR step artifacts.

    - Early artifact: HR too high early, then drops to plausible levels -> cap early HR downward.
    - Late dropout: HR too low after a point (strap/pod issue) -> raise post-drop HR upward.

    Returns: (hr_corr, applied, drop_t_s, corr_type)
      corr_type in {'early_high_then_drop','late_low_dropout'}
    """
    t = np.asarray(t_s, float).reshape(-1)
    p = np.asarray(power_w, float).reshape(-1)
    h = np.asarray(hr_bpm, float).reshape(-1)
    v = np.asarray(speed_mps, float).reshape(-1)

    if len(t) < 900:
        return h.copy(), False, None, None

    # Smooth for robustness
    p30 = _roll_med(p, 31)
    h5 = _roll_med(h, 5)

    moving = _moving_mask(v, p, mass_kg)
    valid = moving & np.isfinite(p30) & np.isfinite(h5)

    # Only infer ratio where HR is plausibly "on".
    valid_ratio = valid & (h5 >= float(hr_floor_bpm))

    # Hard floor: no detection/correction before this.
    start_i = int(np.searchsorted(t, float(hard_floor_s), side="left"))
    if start_i >= len(t) - 120:
        return h.copy(), False, None, None

    # Find a strong HR step change with stable power.
    win = 60  # seconds
    max_i = int(min(len(t) - win - 1, start_i + 25 * 60))
    drop_i = None
    best_mag = 0.0
    for i in range(max(start_i + win, win), max_i):
        if not (valid[i] and valid[i - 1] and valid[i + 1]):
            continue
        pre_hr = float(np.nanmedian(h5[i - win : i]))
        post_hr = float(np.nanmedian(h5[i : i + win]))
        if not (np.isfinite(pre_hr) and np.isfinite(post_hr)):
            continue
        mag = pre_hr - post_hr
        if mag < 12.0 or mag <= best_mag:
            continue
        pre_p = float(np.nanmedian(p30[i - win : i]))
        post_p = float(np.nanmedian(p30[i : i + win]))
        if np.isfinite(pre_p) and np.isfinite(post_p) and pre_p > 0 and post_p > 0:
            if abs(pre_p - post_p) / max(pre_p, post_p) > 0.20:
                continue
        best_mag = mag
        drop_i = i

    if drop_i is None:
        return h.copy(), False, None, None

    # Ratio medians around the drop
    idx = np.arange(len(t))

    def ratio_med(a: int, b: int) -> float:
        a = max(0, int(a))
        b = min(len(t), int(b))
        w = valid_ratio[a:b]
        if int(w.sum()) < 30:
            return float("nan")
        return float(np.nanmedian(p30[a:b][w] / h5[a:b][w]))

    r_pre = ratio_med(drop_i - 120, drop_i)
    r_post = ratio_med(drop_i, drop_i + 120)

    # Classification by ratio shift (more robust than early/late by wall clock)
    # If HR drops, ratio increases. Use relative change threshold.
    if np.isfinite(r_pre) and np.isfinite(r_post) and r_pre > 0:
        rel = (r_post - r_pre) / r_pre
    else:
        rel = float("nan")

    # Decide correction direction
    is_early_high_then_drop = np.isfinite(rel) and (rel > 0.15) and (t[drop_i] <= 20 * 60)
    is_late_low_dropout = np.isfinite(rel) and (rel > 0.15) and (t[drop_i] > 12 * 60)

    # If ambiguous, fall back to timing heuristic.
    if not (is_early_high_then_drop or is_late_low_dropout):
        is_early_high_then_drop = bool(t[drop_i] < 12 * 60)
        is_late_low_dropout = not is_early_high_then_drop

    h_corr = h.copy()
    applied = False
    corr_type = None

    if is_early_high_then_drop:
        # Use stable post-drop ratio from 2–10 min after drop.
        a = drop_i + 120
        b = min(len(t), drop_i + 600)
        w = valid_ratio[a:b]
        if int(w.sum()) >= 60:
            ratio = float(np.nanmedian(p30[a:b][w] / h5[a:b][w]))
            if np.isfinite(ratio) and ratio > 0:
                exp = p30 / ratio
                pre_mask = idx < drop_i
                # Cap early HR downward only
                h_corr[pre_mask] = np.minimum(h_corr[pre_mask], exp[pre_mask])
                h_corr[pre_mask] = np.maximum(h_corr[pre_mask], 60.0)
                applied = True
                corr_type = "early_high_then_drop"

    if (not applied) and is_late_low_dropout:
        # Use stable pre-drop ratio from 6–0.5 min before drop.
        a = max(0, drop_i - 360)
        b = max(0, drop_i - 30)
        w = valid_ratio[a:b]
        if int(w.sum()) >= 60:
            ratio = float(np.nanmedian(p30[a:b][w] / h5[a:b][w]))
            if np.isfinite(ratio) and ratio > 0:
                exp = p30 / ratio
                post_mask = idx >= drop_i
                # Raise post-drop HR upward only
                h_corr[post_mask] = np.maximum(h_corr[post_mask], exp[post_mask])
                applied = True
                corr_type = "late_low_dropout"

    if applied:
        raw_max = float(np.nanmax(h[np.isfinite(h)])) if np.isfinite(h).any() else 200.0
        cap = min(raw_max + 5.0, 205.0)
        h_corr = np.minimum(h_corr, cap)
        h_corr = _roll_med(h_corr, 5)
        return h_corr, True, float(t[drop_i]), corr_type

    return h.copy(), False, None, None


# ----------------------------
# RF / NP
# ----------------------------

def _calc_np(power_w: np.ndarray, win: int = 30) -> float:
    p = np.asarray(power_w, dtype=float).reshape(-1)
    if p.size == 0:
        return float("nan")
    p = np.where(np.isfinite(p), p, np.nan)
    if np.isnan(p).all():
        return float("nan")
    fill = float(np.nanmedian(p))
    if not np.isfinite(fill):
        fill = 0.0
    p = np.nan_to_num(p, nan=fill, posinf=fill, neginf=fill)
    if len(p) < win:
        return float(np.mean(p))
    c = np.cumsum(np.insert(p, 0, 0.0))
    rm = (c[win:] - c[:-win]) / win
    return float(np.mean(rm**4) ** 0.25)


def _rf_metrics(t: np.ndarray, v: np.ndarray, p: np.ndarray, hr: np.ndarray, *, mass_kg: float) -> dict:
    out = {
        "RF_window_start_s": np.nan,
        "RF_window_end_s": np.nan,
        "RF_window_shifted": False,
        "RF_select_code": "0.0",
        "RF_dead_frac": np.nan,
        "RF_adjusted_mean_W_per_bpm": np.nan,
        "RF_adjusted_median_W_per_bpm": np.nan,
        "RF_drift_pct_per_min": np.nan,
        "RF_drift_r2": np.nan,
        "RF_r2": np.nan,
    }

    t = np.asarray(t, float).reshape(-1)
    v = np.asarray(v, float).reshape(-1)
    p = np.asarray(p, float).reshape(-1)
    hr = np.asarray(hr, float).reshape(-1)
    n = len(t)
    if n < 200:
        return out

    wkg = p / max(float(mass_kg), 1e-9)
    dead = (~np.isfinite(p)) | (~np.isfinite(v)) | (v < 0.3) | (wkg < 2.5)
    out["RF_dead_frac"] = float(np.mean(dead))

    moving = (~dead) & np.isfinite(p) & (p > 0) & np.isfinite(v) & (v <= 6.5)
    if int(moving.sum()) < 120:
        return out

    tmax = float(np.nanmax(t))
    start_s = 600.0 if tmax >= 900.0 else 0.0
    win = moving & (t >= start_s)
    if int(win.sum()) < 120:
        start_s = 0.0
        win = moving
        out["RF_window_shifted"] = True
    if int(win.sum()) < 120:
        return out

    out["RF_window_start_s"] = float(start_s)
    out["RF_window_end_s"] = float(np.nanmax(t[win]))

    hr_win = win & np.isfinite(hr) & (hr >= 120.0)
    if int(hr_win.sum()) < 120:
        return out

    k = 30
    kernel = np.ones(k, dtype=float) / k
    p30 = np.convolve(p, kernel, mode="same")

    ratio = p30[hr_win] / hr[hr_win]
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size < 60:
        return out

    out["RF_adjusted_mean_W_per_bpm"] = float(np.mean(ratio))
    out["RF_adjusted_median_W_per_bpm"] = float(np.median(ratio))
    out["RF_select_code"] = "1.0"

    tt = t[hr_win][: ratio.size].astype(float)
    yy = ratio.astype(float)
    x = tt - float(tt.min())
    if yy.size >= 60 and np.isfinite(x).all():
        b, a = np.polyfit(x, yy, 1)
        yhat = a + b * x
        ss_res = float(np.sum((yy - yhat) ** 2))
        ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        drift_pct_per_min = (b / float(np.mean(yy))) * 60.0 * 100.0
        out["RF_drift_pct_per_min"] = float(drift_pct_per_min) if np.isfinite(drift_pct_per_min) else np.nan
        out["RF_drift_r2"] = float(r2) if np.isfinite(r2) else np.nan
        out["RF_r2"] = out["RF_drift_r2"]

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step A (v50): simulate per-second power for pre-Stryd runs (S4 scale) and fill canonical columns.")
    p.add_argument("--master", required=True)
    p.add_argument("--persec-cache-dir", required=True)
    p.add_argument("--model-json", required=True)
    p.add_argument("--override-file", default="activity_overrides.xlsx", help="Activity overrides Excel file (indoor runs identified by temp_override)")
    p.add_argument("--out", required=True)
    p.add_argument("--mass-kg", type=float, default=76.0)
    p.add_argument("--tz", default="Europe/Stockholm", help="Timezone of master 'date' column (default Europe/Stockholm)")
    p.add_argument("--match-tol-s", type=int, default=900, help="Cache match tolerance in seconds (default 900)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    dfm = pd.read_excel(args.master, engine="openpyxl")

    # Build indoor set from activity overrides (runs with temp_override are indoor)
    indoor_files = set()
    if args.override_file and os.path.exists(args.override_file):
        try:
            odf = pd.read_excel(args.override_file, dtype={'file': str})
            if 'file' in odf.columns and 'temp_override' in odf.columns:
                mask = odf['temp_override'].notna()
                indoor_files = set(str(x) for x in odf.loc[mask, 'file'].dropna().astype(str).tolist())
        except Exception:
            indoor_files = set()

    model = REModel.from_json(open(args.model_json, "r", encoding="utf-8").read())
    mid = f"simRE_s4_{_hash_file(args.model_json)}"

    # Ensure output columns exist
    for c in ["power_source", "sim_model_id"]:
        if c not in dfm.columns:
            dfm[c] = ""
        dfm[c] = dfm[c].astype("object")
    if "sim_ok" not in dfm.columns:
        dfm["sim_ok"] = False

    numeric_cols = [
        "power_mean_w",
        "power_median_w",
        "avg_power_w",
        "npower_w",
        "nPower_HR",
        "RE_avg",
        "RE_normalised",
        "RF_window_start_s",
        "RF_window_end_s",
        "RF_dead_frac",
        "RF_adjusted_mean_W_per_bpm",
        "RF_adjusted_median_W_per_bpm",
        "RF_drift_pct_per_min",
        "RF_drift_r2",
        "RF_r2",
        "sim_coverage",
        "sim_re_med",
        "avg_hr",
        "max_hr",
    ]
    for c in numeric_cols:
        if c not in dfm.columns:
            dfm[c] = np.nan
    if "hr_corrected" not in dfm.columns:
        dfm["hr_corrected"] = False

    # Step B HR correction audit (minimal)
    for c, default in [("hr_corr_applied", False), ("hr_corr_type", ""), ("hr_corr_drop_s", np.nan)]:
        if c not in dfm.columns:
            dfm[c] = default

    if "date" not in dfm.columns:
        raise SystemExit("Master file must contain 'date' column to match caches (start_time_utc not present).")

    keys, paths = _build_cache_index(args.persec_cache_dir)

    # Eras that need power simulation (unreliable or no Stryd data)
    SIMULATED_POWER_ERAS = {"pre_stryd", "v1_late"}

    updated = 0
    skipped_missing_cache = 0
    skipped_low_coverage = 0

    for i, row in dfm.iterrows():
        era = str(row.get("calibration_era_id", ""))
        if era not in SIMULATED_POWER_ERAS:
            continue

        dt_local = row.get("date", None)
        if dt_local is None or (isinstance(dt_local, float) and not np.isfinite(dt_local)):
            skipped_missing_cache += 1
            continue
        tloc = pd.Timestamp(dt_local)
        if getattr(tloc, "tzinfo", None) is None:
            tloc = tloc.tz_localize(args.tz, ambiguous="NaT", nonexistent="shift_forward")
        tutc = tloc.tz_convert("UTC")
        epoch = float(tutc.timestamp())

        cache_fp = _find_cache(keys, paths, epoch, tol_s=int(args.match_tol_s))
        if cache_fp is None:
            skipped_missing_cache += 1
            continue

        z = np.load(cache_fp, allow_pickle=True)
        t = _ensure_1d(z["t"])
        v = _ensure_1d(z["speed_mps"])
        g = _ensure_1d(z.get("grade", np.zeros_like(v)))
        hr = _ensure_1d(z.get("hr_bpm", np.full_like(v, np.nan)))

        lens = [len(t), len(v), len(g), len(hr)]
        n = int(min(lens))
        if n <= 0:
            skipped_missing_cache += 1
            continue
        if len(set(lens)) != 1:
            t, v, g, hr = t[:n], v[:n], g[:n], hr[:n]

        # Weather -> rho (wind disabled in this script)
        temp_c = pd.to_numeric(row.get("avg_temp_c", np.nan), errors="coerce")
        rh_pct = pd.to_numeric(row.get("avg_humidity_pct", np.nan), errors="coerce")
        if not np.isfinite(temp_c):
            temp_c = 10.0
        if not np.isfinite(rh_pct):
            rh_pct = 70.0
        rho = air_density_kg_m3(float(temp_c), 1013.25, float(rh_pct) / 100.0)
        rho_arr = np.full_like(v, rho, dtype=float)
        wa = np.zeros_like(v, dtype=float)

        v_s = _roll_med(v, 15)
        g_s = _roll_med(g, 15)
        g_s = np.clip(g_s, -0.12, 0.12)

        p_sim, _re_hat = simulate_power_series(model, v_s, g_s, wa, rho_arr, mass_kg=float(args.mass_kg), smooth_win=15)
        p_sim = _ensure_1d(p_sim)
        if len(p_sim) != len(v):
            p_sim = p_sim[: len(v)]

        moving_cov = (np.isfinite(v) & (v >= 0.3) & (v <= 6.5) & np.isfinite(p_sim) & (p_sim > 0)).astype(bool)
        cov = float(np.mean(moving_cov)) if moving_cov.size else 0.0
        if cov < 0.80:
            skipped_low_coverage += 1
            continue

        p_mov = p_sim[moving_cov]

        # Fill identity
        dfm.at[i, "power_source"] = "sim_v1"
        dfm.at[i, "sim_model_id"] = mid
        dfm.at[i, "sim_ok"] = True
        dfm.at[i, "sim_coverage"] = cov

        avg_p = float(np.nanmean(p_mov))
        med_p = float(np.nanmedian(p_mov))
        dfm.at[i, "power_mean_w"] = avg_p
        dfm.at[i, "power_median_w"] = med_p
        dfm.at[i, "avg_power_w"] = avg_p

        npw = _calc_np(p_mov, win=30)
        dfm.at[i, "npower_w"] = float(npw) if np.isfinite(npw) else np.nan

        # HR correction (Step B only)
        is_indoor = False
        f = str(row.get("file", ""))
        if f and (f in indoor_files):
            is_indoor = True

        hr_used = hr
        hr_applied = False
        hr_drop_s = np.nan
        hr_corr_type = ""
        if (not is_indoor) and np.isfinite(hr).any():
            try:
                hr_used, hr_applied, hr_drop_s_val, hr_corr_type_val = correct_hr_stepb(
                    t,
                    p_sim,
                    hr,
                    v,
                    mass_kg=float(args.mass_kg),
                    hard_floor_s=300.0,
                    hr_floor_bpm=120.0,
                )
                hr_applied = bool(hr_applied)
                if hr_applied and hr_drop_s_val is not None:
                    hr_drop_s = float(hr_drop_s_val)
                if hr_applied and hr_corr_type_val:
                    hr_corr_type = str(hr_corr_type_val)
            except Exception:
                pass

        # Store correction flags (Step B authoritative)
        dfm.at[i, "hr_corrected"] = bool(hr_applied)
        dfm.at[i, "hr_corr_applied"] = bool(hr_applied)
        dfm.at[i, "hr_corr_type"] = hr_corr_type if hr_applied else ""
        dfm.at[i, "hr_corr_drop_s"] = float(hr_drop_s) if hr_applied and np.isfinite(hr_drop_s) else np.nan

        # avg/max HR from used series over standing-excluded mask
        mv_hr = _moving_mask(v, p_sim, float(args.mass_kg)) & np.isfinite(hr_used) & (hr_used > 30)
        if int(mv_hr.sum()) >= 30:
            dfm.at[i, "avg_hr"] = float(np.nanmean(hr_used[mv_hr]))
            dfm.at[i, "max_hr"] = float(np.nanmax(hr_used[mv_hr]))

        # nPower:HR (use HR floor 120 when possible)
        if np.isfinite(npw):
            hwin = _moving_mask(v, p_sim, float(args.mass_kg)) & np.isfinite(hr_used) & (hr_used >= 120.0)
            if int(hwin.sum()) >= 60:
                dfm.at[i, "nPower_HR"] = float(npw / float(np.nanmean(hr_used[hwin])))
            else:
                hwin2 = _moving_mask(v, p_sim, float(args.mass_kg)) & np.isfinite(hr_used) & (hr_used > 30)
                if int(hwin2.sum()) >= 60:
                    dfm.at[i, "nPower_HR"] = float(npw / float(np.nanmean(hr_used[hwin2])))

        # Canonical run speed
        dist_km = pd.to_numeric(row.get("distance_km", np.nan), errors="coerce")
        mov_s = pd.to_numeric(row.get("moving_time_s", np.nan), errors="coerce")
        speed_run = (float(dist_km) * 1000.0 / float(mov_s)) if (np.isfinite(dist_km) and np.isfinite(mov_s) and mov_s > 0) else float(np.nanmean(v[moving_cov]))

        if np.isfinite(speed_run) and avg_p > 30:
            dfm.at[i, "RE_avg"] = float((speed_run * float(args.mass_kg)) / avg_p)
        if np.isfinite(speed_run) and np.isfinite(npw) and float(npw) > 30:
            dfm.at[i, "RE_normalised"] = float((speed_run * float(args.mass_kg)) / float(npw))

        # sim_re_med diagnostic
        re_hat = (v[moving_cov] * float(args.mass_kg)) / p_sim[moving_cov]
        dfm.at[i, "sim_re_med"] = float(np.nanmedian(re_hat)) if np.isfinite(re_hat).any() else np.nan

        # RF metrics (Step B authoritative)
        rf = _rf_metrics(t, v, p_sim, hr_used, mass_kg=float(args.mass_kg))
        for k, val in rf.items():
            if k not in dfm.columns:
                dfm[k] = np.nan
            dfm.at[i, k] = _coerce_scalar(val)

        updated += 1

    # Rounding
    for c in ["power_mean_w", "power_median_w", "avg_power_w", "npower_w"]:
        if c in dfm.columns:
            dfm[c] = pd.to_numeric(dfm[c], errors="coerce").round(0)

    cols_3dp = [
        "gps_distance_ratio",
        "avg_pace_min_per_km",
        "power_adjuster_to_S4",
        "nPower_HR",
        "RE_avg",
        "RE_normalised",
        "RF_adjusted_mean_W_per_bpm",
        "RF_adjusted_median_W_per_bpm",
        "RF_dead_frac",
        "RF_drift_pct_per_min",
        "RF_drift_r2",
        "RF_r2",
        "sim_re_med",
        "sim_coverage",
        "avg_hr",
        "max_hr",
    ]
    for c in cols_3dp:
        if c in dfm.columns:
            dfm[c] = pd.to_numeric(dfm[c], errors="coerce").round(3)

    dfm.to_excel(args.out, index=False, engine="openpyxl")

    print(f"Wrote: {args.out}")
    print(f"Updated pre-Stryd runs: {updated}")
    print(f"Skipped (missing cache): {skipped_missing_cache}")
    print(f"Skipped (low coverage): {skipped_low_coverage}")
    print(f"Model id: {mid}")
    print(f"Indoor overrides loaded: {len(indoor_files)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
