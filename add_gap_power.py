"""
GAP Mode Post-Processor - Add simulated power to master file for GAP/HR mode.

This module runs after rebuild_from_fit_zip.py and adds GAP-based simulated
power to all rows, enabling the RF/HR calculation pipeline to work identically
in both Stryd and GAP modes.

In GAP mode:
- Reads per-second cache files (speed, grade from GPS)
- Computes simulated power using Minetti cost model
- Adds power columns to master file (avg_power_w, npower_w, etc.)
- Sets power_source = "gap_simulated"

Usage:
    python add_gap_power.py --master Master.xlsx --cache-dir persec_cache/ --out Master_GAP.xlsx
    
Or programmatically:
    from add_gap_power import add_gap_power_to_master
    add_gap_power_to_master(master_df, cache_dir, mass_kg, re_constant)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from gap_power import compute_gap_power_for_run, compute_gap_for_run

# Try to import rolling_median, or use simple version
try:
    from sim_power_pipeline import rolling_median
except ImportError:
    def rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
        """Simple rolling median implementation."""
        from scipy.ndimage import median_filter
        return median_filter(arr, size=window, mode='nearest')


def compute_gap_power_metrics(
    speed_mps: np.ndarray,
    grade: np.ndarray,
    mass_kg: float,
    re_constant: float
) -> dict:
    """
    Compute power metrics from speed and grade.
    
    Returns same metrics as Stryd power:
    - avg_power_w: Average power over moving segments
    - npower_w: Normalized power (30s rolling average, then 4th-root mean)
    - avg_gap_pace_min_per_km: Average GAP (equivalent flat pace) in min/km
    
    Args:
        speed_mps: Speed array in m/s
        grade: Grade array (fraction)
        mass_kg: Athlete mass in kg
        re_constant: Running economy constant
        
    Returns:
        dict with avg_power_w, npower_w, avg_gap_pace_min_per_km
    """
    # Compute power
    power_w = compute_gap_power_for_run(speed_mps, grade, mass_kg, re_constant)
    
    # Average power (moving only)
    moving_mask = (speed_mps > 0.3) & np.isfinite(power_w)
    if moving_mask.sum() < 10:
        return {'avg_power_w': np.nan, 'npower_w': np.nan,
                'avg_gap_pace_min_per_km': np.nan}
    
    avg_power = float(np.mean(power_w[moving_mask]))
    
    # Normalized Power (NP): 30s rolling average, then 4th-root mean
    # This weights harder efforts more heavily
    power_30s = rolling_median(power_w, 30)
    np_power = float(np.mean(power_30s[moving_mask]**4)**(1/4))
    
    # GAP (Grade Adjusted Pace): equivalent flat speed
    # Uses the Minetti cost model to compute what flat speed would produce
    # the same effort as the actual speed+grade combination.
    gap_speed = compute_gap_for_run(speed_mps, grade)
    gap_moving = gap_speed[moving_mask]
    gap_valid = gap_moving[np.isfinite(gap_moving) & (gap_moving > 0.3)]
    if len(gap_valid) > 0:
        avg_gap_speed_mps = float(np.mean(gap_valid))
        avg_gap_pace = (1000 / avg_gap_speed_mps) / 60  # min/km
    else:
        avg_gap_pace = np.nan
    
    return {
        'avg_power_w': avg_power,
        'npower_w': np_power,
        'avg_gap_pace_min_per_km': round(avg_gap_pace, 2) if np.isfinite(avg_gap_pace) else np.nan,
    }


def add_gap_power_to_run(
    row: pd.Series,
    cache_dir: str,
    mass_kg: float,
    re_constant: float
) -> dict:
    """
    Add GAP-simulated power metrics to a single run.
    
    Args:
        row: DataFrame row with file, distance_km, etc.
        cache_dir: Path to per-second cache directory
        mass_kg: Athlete mass
        re_constant: Running economy constant
        
    Returns:
        dict with power metrics to update row
    """
    # Find cache file
    run_id = str(row.get('file', '')).strip()
    if not run_id:
        return {}
    
    key = os.path.splitext(os.path.basename(run_id))[0]
    npz_path = Path(cache_dir) / f"{key}.npz"
    
    if not npz_path.exists():
        return {}
    
    try:
        z = np.load(npz_path)
    except Exception:
        return {}
    
    # Need speed and grade (or default to flat)
    if 'speed_mps' not in z.files:
        return {}
    
    speed = z['speed_mps'].astype(float)
    grade = z['grade'].astype(float) if 'grade' in z.files else np.zeros_like(speed)
    
    # Compute power metrics
    metrics = compute_gap_power_metrics(speed, grade, mass_kg, re_constant)
    
    return metrics


def add_gap_power_to_master(
    df: pd.DataFrame,
    cache_dir: str,
    mass_kg: float,
    re_constant: float = 0.92,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Add GAP-simulated power to all rows in master dataframe.
    
    Args:
        df: Master dataframe from rebuild_from_fit_zip
        cache_dir: Path to per-second cache files
        mass_kg: Athlete mass in kg
        re_constant: Running economy constant
        verbose: Print progress
        
    Returns:
        Updated dataframe with power columns
    """
    df = df.copy()
    
    # Initialize columns if they don't exist
    if 'avg_power_w' not in df.columns:
        df['avg_power_w'] = np.nan
    if 'npower_w' not in df.columns:
        df['npower_w'] = np.nan
    if 'power_source' not in df.columns:
        df['power_source'] = ''
    if 'avg_gap_pace_min_per_km' not in df.columns:
        df['avg_gap_pace_min_per_km'] = np.nan
    
    # Process each run
    success_count = 0
    gap_pace_count = 0
    for idx in df.index:
        row = df.loc[idx]
        
        has_power = pd.notna(row.get('avg_power_w'))
        has_gap_pace = pd.notna(row.get('avg_gap_pace_min_per_km'))
        
        # Skip if already has both power and GAP pace
        if has_power and has_gap_pace:
            continue
        
        # Compute GAP metrics from cache
        metrics = add_gap_power_to_run(row, cache_dir, mass_kg, re_constant)
        
        if metrics:
            # Only set power if not already present (preserve Stryd)
            if not has_power:
                df.loc[idx, 'avg_power_w'] = metrics['avg_power_w']
                df.loc[idx, 'npower_w'] = metrics['npower_w']
                df.loc[idx, 'power_source'] = 'gap_simulated'
                success_count += 1
            
            # Always set GAP pace if available
            gap_pace = metrics.get('avg_gap_pace_min_per_km')
            if gap_pace is not None and not pd.isna(gap_pace):
                df.loc[idx, 'avg_gap_pace_min_per_km'] = gap_pace
                gap_pace_count += 1
            
            if verbose and (success_count + gap_pace_count) % 200 == 0:
                print(f"  Processed {success_count} power + {gap_pace_count} GAP pace...")
    
    if verbose:
        print(f"\nGAP power summary:")
        print(f"  Runs with GAP power added: {success_count}")
        print(f"  Runs with GAP pace added: {gap_pace_count}")
        print(f"  Runs with Stryd power: {df['power_source'].eq('stryd').sum()}")
        print(f"  Total runs with power: {pd.notna(df['avg_power_w']).sum()}")
    
    return df


def main() -> int:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Add GAP-simulated power to master file for GAP/HR mode"
    )
    parser.add_argument(
        "--master",
        required=True,
        help="Path to master Excel file from rebuild_from_fit_zip"
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Path to per-second cache directory"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for updated master file"
    )
    parser.add_argument(
        "--mass-kg",
        type=float,
        default=76.0,
        help="Athlete mass in kg (default: 76.0)"
    )
    parser.add_argument(
        "--re-constant",
        type=float,
        default=0.92,
        help="Running economy constant (default: 0.92 for male, ~0.94 for female)"
    )
    
    args = parser.parse_args()
    
    print(f"Loading master file: {args.master}")
    df = pd.read_excel(args.master)
    print(f"  Loaded {len(df)} runs")
    
    print(f"\nAdding GAP power...")
    df = add_gap_power_to_master(
        df,
        args.cache_dir,
        args.mass_kg,
        args.re_constant,
        verbose=True
    )
    
    print(f"\nSaving to: {args.out}")
    df.to_excel(args.out, index=False)
    print("Done!")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
