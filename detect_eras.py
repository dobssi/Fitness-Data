"""
detect_eras.py — Auto-detect Stryd hardware eras from power data.

Compares Stryd power against GAP (Grade Adjusted Pace) simulated power to find
persistent calibration shifts that indicate hardware changes, weight config changes,
or firmware updates.

Algorithm:
  1. Compute Stryd/GAP ratio (nPower_HR / RF_gap_median) for quality-filtered runs
  2. Binary segmentation to find changepoints (max variance reduction)
  3. BIC (Bayesian Information Criterion) selects optimal number of segments
  4. Each segment = one calibration era

Runs AFTER add_gap_power.py (needs RF_gap_median) and BEFORE StepB era adjusters.

Usage:
    # From command line
    python detect_eras.py --master Master_FULL.xlsx [--min-runs 30] [--out era_report.json]

    # From pipeline
    from detect_eras import detect_stryd_eras
    eras = detect_stryd_eras(df)  # returns list of EraSegment

Author: Pipeline v52
Date: 2026-03-08
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────

@dataclass
class EraSegment:
    """One detected calibration era."""
    era_id: int                   # 1-based sequential ID
    label: str                    # Auto-generated label: "era_1", "era_2", ...
    start_date: str               # ISO date of first run in era
    end_date: str                 # ISO date of last run in era
    n_runs: int                   # Number of quality-filtered runs
    n_runs_total: int             # Number of ALL Stryd runs (including filtered)
    mean_ratio: float             # Mean Stryd/GAP ratio
    std_ratio: float              # Std dev of ratio
    step_from_prev_pct: float     # Step change from previous era (%)
    is_anchor: bool               # True if this is the reference era (adjuster=1.0)
    adjuster: float               # Multiplier to normalise to anchor era


@dataclass
class EraDetectionResult:
    """Full result of era detection."""
    eras: List[EraSegment]
    anchor_era_id: int
    n_quality_runs: int           # Runs used for detection
    n_stryd_runs: int             # Total Stryd runs
    n_total_runs: int             # All runs in master
    pre_stryd_runs: int           # Runs with no Stryd power
    method: str                   # "changepoint" or "single_era" or "no_power"


# ─────────────────────────────────────────────────────────────────────
# Quality filtering
# ─────────────────────────────────────────────────────────────────────

def build_ratio_series(df: pd.DataFrame, min_duration_s: float = 1200,
                       pace_min: float = 4.0, pace_max: float = 6.2,
                       gps_ratio_min: float = 0.97, gps_ratio_max: float = 1.03,
                       ) -> pd.DataFrame:
    """
    Build the Stryd/GAP ratio time series from a master DataFrame.

    Filters for:
      - Stryd power present (power_source == 'stryd')
      - RF_gap_median and nPower_HR both valid
      - GPS quality (distance ratio close to 1.0)
      - Minimum duration (skip short reps/tests)
      - Reasonable pace (exclude walks and sprint tests)
      - Plausible ratio range (0.7–1.3)

    Returns a filtered DataFrame sorted by date with 'ratio' column added.
    """
    out = df.copy()

    # Must have Stryd power
    out = out[out.get('power_source', pd.Series(dtype=str)).eq('stryd')]
    if len(out) == 0:
        return out

    # Compute ratio
    rf_stryd = pd.to_numeric(out.get('nPower_HR'), errors='coerce')
    rf_gap = pd.to_numeric(out.get('RF_gap_median'), errors='coerce')
    out['_det_ratio'] = rf_stryd / rf_gap

    # Quality filters
    gps_rat = pd.to_numeric(out.get('gps_distance_ratio'), errors='coerce')
    moving_s = pd.to_numeric(out.get('moving_time_s'), errors='coerce')
    pace = pd.to_numeric(out.get('avg_pace_min_per_km'), errors='coerce')

    mask = (
        out['_det_ratio'].between(0.7, 1.3) &
        gps_rat.between(gps_ratio_min, gps_ratio_max) &
        (moving_s >= min_duration_s) &
        pace.between(pace_min, pace_max) &
        rf_stryd.notna() &
        rf_gap.notna() &
        (rf_gap > 0)
    )

    out = out[mask].sort_values('date').reset_index(drop=True)
    return out


# ─────────────────────────────────────────────────────────────────────
# Changepoint detection (binary segmentation + BIC)
# ─────────────────────────────────────────────────────────────────────

def _rss(segment: np.ndarray) -> float:
    """Residual sum of squares from segment mean."""
    if len(segment) < 2:
        return 0.0
    return float(np.sum((segment - np.mean(segment)) ** 2))


def _find_best_split(data: np.ndarray, min_seg: int) -> Tuple[Optional[int], float]:
    """Find the index that maximises variance reduction when splitting data."""
    n = len(data)
    if n < 2 * min_seg:
        return None, 0.0

    total = _rss(data)

    # Pre-compute cumulative sums for O(n) cost calculation
    cs = np.cumsum(data)
    css = np.cumsum(data ** 2)

    def _seg_rss(start: int, end: int) -> float:
        """RSS for data[start:end] using cumulative sums."""
        length = end - start
        if length < 2:
            return 0.0
        s = cs[end - 1] - (cs[start - 1] if start > 0 else 0.0)
        ss = css[end - 1] - (css[start - 1] if start > 0 else 0.0)
        return float(ss - s * s / length)

    best_gain = 0.0
    best_idx = None

    for i in range(min_seg, n - min_seg):
        gain = total - _seg_rss(0, i) - _seg_rss(i, n)
        if gain > best_gain:
            best_gain = gain
            best_idx = i

    return best_idx, best_gain


def _bic(rss: float, n: int, k_segments: int) -> float:
    """
    BIC score for a segmentation.

    BIC = n * log(RSS/n) + k * log(n)
    where k = 2 * n_segments (each segment has mean + variance).
    Lower is better.
    """
    if rss <= 0 or n <= 0:
        return float('inf')
    return n * np.log(rss / n) + 2 * k_segments * np.log(n)


def find_changepoints(data: np.ndarray, min_segment: int = 30,
                      max_splits: int = 12) -> List[int]:
    """
    Binary segmentation with BIC model selection.

    Greedily splits the time series, recording BIC at each step.
    Returns the changepoint set that minimises BIC.

    Args:
        data: 1D array of Stryd/GAP ratios (sorted by date)
        min_segment: Minimum runs per era
        max_splits: Maximum number of splits to attempt

    Returns:
        List of changepoint indices (sorted)
    """
    n = len(data)
    if n < 2 * min_segment:
        return []

    # Track all configurations and their BIC
    segments_history = []
    segs = [(0, n)]
    total_rss = _rss(data)
    current_bic = _bic(total_rss, n, 1)
    segments_history.append(([], current_bic))

    all_cps = []

    for _ in range(max_splits):
        # Find the segment with the best split
        best_gain = 0.0
        best_global_idx = None
        best_seg_i = None

        for si, (start, end) in enumerate(segs):
            idx, gain = _find_best_split(data[start:end], min_segment)
            if idx is not None and gain > best_gain:
                best_gain = gain
                best_global_idx = start + idx
                best_seg_i = si

        if best_global_idx is None:
            break

        # Apply the split
        old_start, old_end = segs[best_seg_i]
        segs[best_seg_i] = (old_start, best_global_idx)
        segs.insert(best_seg_i + 1, (best_global_idx, old_end))
        all_cps.append(best_global_idx)

        # Compute BIC for this configuration
        total_rss = sum(_rss(data[s:e]) for s, e in segs)
        current_bic = _bic(total_rss, n, len(segs))
        segments_history.append((sorted(all_cps.copy()), current_bic))

    # Select the configuration with minimum BIC
    best_config = min(segments_history, key=lambda x: x[1])
    return best_config[0]


# ─────────────────────────────────────────────────────────────────────
# Era assignment (map changepoints back to all runs)
# ─────────────────────────────────────────────────────────────────────

def _changepoints_to_segments(changepoints: List[int], n: int) -> List[Tuple[int, int]]:
    """Convert sorted changepoint indices to (start, end) segment tuples."""
    boundaries = [0] + sorted(changepoints) + [n]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _assign_era_to_date(date: pd.Timestamp, era_boundaries: List[Tuple[str, str, int]]) -> int:
    """
    Assign an era ID to a run date.

    era_boundaries: list of (start_date, end_date, era_id) sorted by date.
    Returns era_id, or 0 for pre-Stryd.
    """
    for start, end, eid in era_boundaries:
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return eid
    return 0  # pre-Stryd


# ─────────────────────────────────────────────────────────────────────
# Main detection function
# ─────────────────────────────────────────────────────────────────────

def detect_stryd_eras(df: pd.DataFrame, min_segment: int = 30,
                      max_splits: int = 12,
                      verbose: bool = True) -> EraDetectionResult:
    """
    Detect Stryd hardware/calibration eras from a master DataFrame.

    Requires columns: power_source, nPower_HR, RF_gap_median, gps_distance_ratio,
                      moving_time_s, avg_pace_min_per_km, date

    Args:
        df: Master DataFrame (post add_gap_power.py)
        min_segment: Minimum quality-filtered runs per era
        max_splits: Maximum eras to detect
        verbose: Print detection progress

    Returns:
        EraDetectionResult with detected eras and metadata
    """
    n_total = len(df)

    # Count Stryd runs
    power_src = df.get('power_source', pd.Series(dtype=str))
    n_stryd = int((power_src == 'stryd').sum())
    n_pre_stryd = int(n_total - n_stryd)  # Simplified: non-stryd = pre-stryd

    if n_stryd == 0:
        if verbose:
            print("[era] No Stryd power data found — skipping era detection")
        return EraDetectionResult(
            eras=[], anchor_era_id=0, n_quality_runs=0,
            n_stryd_runs=0, n_total_runs=n_total,
            pre_stryd_runs=n_pre_stryd, method="no_power"
        )

    # Build quality-filtered ratio series
    clean = build_ratio_series(df)
    n_quality = len(clean)

    if verbose:
        print(f"[era] Stryd runs: {n_stryd}, quality-filtered: {n_quality} "
              f"({n_quality / n_stryd * 100:.0f}%)")

    if n_quality < 2 * min_segment:
        # Not enough data to split — single era
        if n_quality > 0:
            ratio_mean = float(clean['_det_ratio'].mean())
            ratio_std = float(clean['_det_ratio'].std())
            start_d = str(clean['date'].min().date())
            end_d = str(clean['date'].max().date())
        else:
            ratio_mean = 1.0
            ratio_std = 0.0
            stryd_df = df[power_src == 'stryd'].sort_values('date')
            start_d = str(stryd_df['date'].min().date()) if len(stryd_df) > 0 else "unknown"
            end_d = str(stryd_df['date'].max().date()) if len(stryd_df) > 0 else "unknown"

        era = EraSegment(
            era_id=1, label="era_1", start_date=start_d, end_date=end_d,
            n_runs=n_quality, n_runs_total=n_stryd,
            mean_ratio=ratio_mean, std_ratio=ratio_std,
            step_from_prev_pct=0.0, is_anchor=True, adjuster=1.0
        )
        if verbose:
            print(f"[era] Insufficient data for segmentation (<{2 * min_segment} quality runs) "
                  f"— single era")
        return EraDetectionResult(
            eras=[era], anchor_era_id=1, n_quality_runs=n_quality,
            n_stryd_runs=n_stryd, n_total_runs=n_total,
            pre_stryd_runs=n_pre_stryd, method="single_era"
        )

    # Run changepoint detection
    ratios = clean['_det_ratio'].values
    dates = clean['date'].values

    if verbose:
        print(f"[era] Running changepoint detection (min_segment={min_segment})...")

    changepoints = find_changepoints(ratios, min_segment=min_segment,
                                     max_splits=max_splits)
    segments = _changepoints_to_segments(changepoints, len(ratios))

    if verbose:
        print(f"[era] BIC-optimal: {len(segments)} era(s) detected")

    # Choose anchor: most recent era with enough data
    # (most recent is typically the best-calibrated and what the user cares about)
    anchor_idx = len(segments) - 1
    for i in range(len(segments) - 1, -1, -1):
        s, e = segments[i]
        if (e - s) >= min_segment:
            anchor_idx = i
            break

    anchor_mean = float(np.mean(ratios[segments[anchor_idx][0]:segments[anchor_idx][1]]))

    # Build EraSegment objects
    eras = []
    for i, (start, end) in enumerate(segments):
        seg_data = ratios[start:end]
        seg_mean = float(np.mean(seg_data))
        seg_std = float(np.std(seg_data))

        start_date = str(pd.Timestamp(dates[start]).date())
        end_date = str(pd.Timestamp(dates[end - 1]).date())

        # Count total Stryd runs (not just quality-filtered) in this date range
        d_start = pd.Timestamp(dates[start])
        d_end = pd.Timestamp(dates[end - 1])
        in_range = df[(df['date'] >= d_start) & (df['date'] <= d_end) &
                      (power_src == 'stryd')]
        n_total_era = len(in_range)

        # Step from previous era
        if i > 0:
            prev_mean = float(np.mean(ratios[segments[i - 1][0]:segments[i - 1][1]]))
            step_pct = (seg_mean / prev_mean - 1.0) * 100.0
        else:
            step_pct = 0.0

        # Adjuster: ratio to normalise this era to anchor scale
        # adjuster * stryd_power = stryd_power_at_anchor_scale
        is_anchor = (i == anchor_idx)
        adjuster = anchor_mean / seg_mean if seg_mean > 0 else 1.0

        era = EraSegment(
            era_id=i + 1,
            label=f"era_{i + 1}",
            start_date=start_date,
            end_date=end_date,
            n_runs=end - start,
            n_runs_total=n_total_era,
            mean_ratio=round(seg_mean, 5),
            std_ratio=round(seg_std, 5),
            step_from_prev_pct=round(step_pct, 2),
            is_anchor=is_anchor,
            adjuster=round(adjuster, 5),
        )
        eras.append(era)

    result = EraDetectionResult(
        eras=eras,
        anchor_era_id=anchor_idx + 1,
        n_quality_runs=n_quality,
        n_stryd_runs=n_stryd,
        n_total_runs=n_total,
        pre_stryd_runs=n_pre_stryd,
        method="changepoint",
    )

    if verbose:
        _print_detection_report(result)

    return result


# ─────────────────────────────────────────────────────────────────────
# Assign detected eras to full master DataFrame
# ─────────────────────────────────────────────────────────────────────

def assign_detected_eras(df: pd.DataFrame, result: EraDetectionResult,
                         era_col: str = 'detected_era_id',
                         adjuster_col: str = 'detected_era_adj') -> pd.DataFrame:
    """
    Add era assignment and adjuster columns to the master DataFrame.

    Runs with Stryd power get assigned to the era covering their date.
    Runs without Stryd power get era_id=0 and adjuster=NaN.

    The adjuster normalises Stryd power to the anchor era's scale:
        normalised_power = raw_power * adjuster

    Args:
        df: Master DataFrame
        result: EraDetectionResult from detect_stryd_eras()
        era_col: Column name for era ID
        adjuster_col: Column name for era adjuster

    Returns:
        DataFrame with new columns added
    """
    df = df.copy()
    df[era_col] = 0
    df[adjuster_col] = np.nan

    if not result.eras:
        return df

    power_src = df.get('power_source', pd.Series(dtype=str))
    is_stryd = (power_src == 'stryd')

    for era in result.eras:
        era_start = pd.Timestamp(era.start_date)
        era_end = pd.Timestamp(era.end_date) + pd.Timedelta(days=1)  # inclusive

        mask = is_stryd & (df['date'] >= era_start) & (df['date'] < era_end)
        df.loc[mask, era_col] = era.era_id
        df.loc[mask, adjuster_col] = era.adjuster

    # Stryd runs outside any detected era (shouldn't happen, but handle gracefully)
    # Assign to nearest era
    unassigned = is_stryd & (df[era_col] == 0)
    if unassigned.any():
        for idx in df[unassigned].index:
            run_date = df.at[idx, 'date']
            # Find nearest era by date
            best_era = None
            best_dist = float('inf')
            for era in result.eras:
                mid = pd.Timestamp(era.start_date) + (
                    pd.Timestamp(era.end_date) - pd.Timestamp(era.start_date)) / 2
                dist = abs((run_date - mid).total_seconds())
                if dist < best_dist:
                    best_dist = dist
                    best_era = era
            if best_era:
                df.at[idx, era_col] = best_era.era_id
                df.at[idx, adjuster_col] = best_era.adjuster

    # Sim power (pre-Stryd) runs: calibrate to first Stryd era's scale.
    # Sim power ≈ GAP power (Minetti model). The first era's mean_ratio tells us
    # how that earliest Stryd pod reads relative to GAP: ratio = Stryd_RF / GAP_RF.
    # Scaling sim to the first era (not anchor) keeps the pre-Stryd → first-Stryd
    # transition seamless — no discontinuity at the boundary.
    is_sim = (power_src == 'sim_v1')
    if is_sim.any() and result.eras:
        first_era = result.eras[0]  # Earliest detected era
        sim_adj = first_era.mean_ratio
        df.loc[is_sim, era_col] = 0  # keep era_id=0 (pre-Stryd)
        df.loc[is_sim, adjuster_col] = round(sim_adj, 5)

    return df


# ─────────────────────────────────────────────────────────────────────
# Garmin/wrist power detection
# ─────────────────────────────────────────────────────────────────────

def flag_non_stryd_power(df: pd.DataFrame, result: EraDetectionResult,
                         threshold_std: float = 3.0,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Flag runs where power likely comes from a wrist sensor, not Stryd.

    Uses the detected era ratios as reference. Runs with Stryd/GAP ratio
    far outside the expected range for their era are flagged.

    This catches:
      - Garmin wrist power (typically 10-30% different scale from Stryd)
      - Coros wrist power (similar issue)
      - Any non-Stryd power source with different calibration

    Adds column 'power_suspect' (bool): True if power source is suspect.
    """
    df = df.copy()
    df['power_suspect'] = False

    if not result.eras:
        return df

    # Compute ratio for all runs with power + GAP data
    rf_stryd = pd.to_numeric(df.get('nPower_HR'), errors='coerce')
    rf_gap = pd.to_numeric(df.get('RF_gap_median'), errors='coerce')
    has_both = rf_stryd.notna() & rf_gap.notna() & (rf_gap > 0)
    ratio = rf_stryd / rf_gap

    # For each era, flag runs whose ratio is far from the era mean
    power_src = df.get('power_source', pd.Series(dtype=str))
    is_stryd = (power_src == 'stryd')

    # Build a global expected range from all eras
    all_means = [e.mean_ratio for e in result.eras]
    all_stds = [e.std_ratio for e in result.eras]
    global_min = min(all_means) - threshold_std * max(all_stds)
    global_max = max(all_means) + threshold_std * max(all_stds)

    suspect = has_both & is_stryd & (~ratio.between(global_min, global_max))
    df.loc[suspect, 'power_suspect'] = True

    if verbose and suspect.any():
        print(f"[era] Flagged {suspect.sum()} runs with suspect power "
              f"(ratio outside {global_min:.3f}–{global_max:.3f})")

    return df


# ─────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────

def _print_detection_report(result: EraDetectionResult):
    """Print a human-readable era detection report."""
    print()
    print("=" * 70)
    print("STRYD ERA DETECTION REPORT")
    print("=" * 70)
    print(f"  Total runs:        {result.n_total_runs}")
    print(f"  Stryd power runs:  {result.n_stryd_runs}")
    print(f"  Pre-Stryd runs:    {result.pre_stryd_runs}")
    print(f"  Quality-filtered:  {result.n_quality_runs} "
          f"({result.n_quality_runs / max(result.n_stryd_runs, 1) * 100:.0f}% of Stryd)")
    print(f"  Method:            {result.method}")
    print(f"  Eras detected:     {len(result.eras)}")
    print(f"  Anchor era:        era_{result.anchor_era_id}")
    print()

    for era in result.eras:
        anchor_tag = " ← ANCHOR" if era.is_anchor else ""
        step_tag = f"  step: {era.step_from_prev_pct:+.1f}%" if era.era_id > 1 else ""
        print(f"  {era.label}:  {era.start_date} to {era.end_date}  "
              f"ratio={era.mean_ratio:.4f} ± {era.std_ratio:.4f}  "
              f"adj={era.adjuster:.4f}  "
              f"n={era.n_runs} ({era.n_runs_total} total)"
              f"{step_tag}{anchor_tag}")

    print("=" * 70)


def export_result_json(result: EraDetectionResult, path: str):
    """Export detection result to JSON for review/override."""
    out = {
        "method": result.method,
        "anchor_era_id": result.anchor_era_id,
        "n_quality_runs": result.n_quality_runs,
        "n_stryd_runs": result.n_stryd_runs,
        "n_total_runs": result.n_total_runs,
        "pre_stryd_runs": result.pre_stryd_runs,
        "eras": [asdict(e) for e in result.eras],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[era] Exported detection result to {path}")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Detect Stryd hardware eras from master file")
    ap.add_argument("--master", required=True, help="Master XLSX file (post add_gap_power.py)")
    ap.add_argument("--sheet", default=None, help="Sheet name (auto-detect if not set)")
    ap.add_argument("--min-runs", type=int, default=30,
                    help="Minimum quality runs per era (default: 30)")
    ap.add_argument("--max-eras", type=int, default=12,
                    help="Maximum eras to detect (default: 12)")
    ap.add_argument("--out", default="", help="Output JSON file for era report")
    args = ap.parse_args()

    # Load master
    if args.sheet:
        df = pd.read_excel(args.master, sheet_name=args.sheet)
    else:
        # Auto-detect: try 'Master', then first sheet containing 'Master'
        xls = pd.ExcelFile(args.master)
        sheet = None
        for s in xls.sheet_names:
            if s.lower().startswith('master'):
                sheet = s
                break
        if sheet is None:
            sheet = xls.sheet_names[0]
        print(f"[era] Reading sheet: {sheet}")
        df = pd.read_excel(args.master, sheet_name=sheet)

    # Detect eras
    result = detect_stryd_eras(df, min_segment=args.min_runs,
                               max_splits=args.max_eras, verbose=True)

    # Export
    if args.out:
        export_result_json(result, args.out)
    else:
        out_path = str(Path(args.master).parent / "detected_eras.json")
        export_result_json(result, out_path)


if __name__ == "__main__":
    main()
