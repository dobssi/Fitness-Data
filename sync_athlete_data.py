"""
sync_athlete_data.py — Sync weight + non-running TSS from intervals.icu
=======================================================================
Phase 1 replacement for export_athlete_data.py (which read from BFW).

Data flow:
  intervals.icu API ---> weight (2023+) + non-running TSS (all years)
  athlete_data.csv  ---> weight (pre-2023, preserved from BFW export)
                    \
                     +---> merged athlete_data.csv (same format as before)

Non-running TSS formula:
  TSS = (calories - BMR_per_hour × elapsed_hours) / 10
  BMR assumed 80 kcal/hour. One universal formula for all sport types.

The output athlete_data.csv is fully backward compatible with StepB.

Usage:
  python sync_athlete_data.py
  python sync_athlete_data.py --athlete-data athlete_data.csv
  python sync_athlete_data.py --dry-run     (preview changes without writing)

Environment variables:
  INTERVALS_API_KEY=your_key
  INTERVALS_ATHLETE_ID=i12345
"""

import os
import sys
import io
import argparse
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

# Import the intervals.icu client from our library module
try:
    from intervals_fetch import IntervalsClient
except ImportError:
    sys.exit("Cannot import intervals_fetch.py — ensure it's in the same directory")


# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_ATHLETE_DATA = "athlete_data.csv"
INTERVALS_WEIGHT_START = "2026-01-01"   # BFW data is authoritative pre-2026
RUNNING_TYPES = {"Run", "VirtualRun"}   # Excluded from non-running TSS

# Non-running TSS formula: (calories - BMR_PER_HOUR * duration_hours) / TSS_DIVISOR
BMR_PER_HOUR = 80     # Approximate BMR in kcal/hour
TSS_DIVISOR = 10      # Scaling factor to align with running TSS scale

# Weight post-processing (for dashboard display ONLY — pipeline always uses
# ATHLETE_MASS_KG = 76.0 from config.py, passed as --mass-kg 76)
WEIGHT_PREFILL_DAYS = 3       # Forward-fill this many days beyond last measurement
WEIGHT_SMOOTH_WINDOW = 7      # Centred moving average window (±3 days)


def read_existing_athlete_data(path: str) -> pd.DataFrame:
    """
    Read existing athlete_data.csv, handling comment headers.
    
    Mirrors _read_athlete_csv() from StepB for consistency.
    """
    if not os.path.exists(path):
        print(f"  No existing {path} — starting fresh")
        return pd.DataFrame(columns=["date", "weight_kg", "garmin_tr", "non_running_tss"])
    
    with open(path, "r") as f:
        raw_lines = f.readlines()
    
    clean_lines = []
    for line in raw_lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            # BFW export bug: header sometimes concatenated after comment
            # e.g. '# comment#date,weight_kg,garmin_tr,non_running_tss'
            last_hash = stripped.rfind("#")
            remainder = stripped[last_hash + 1:].strip()
            # Only treat as data if it looks like a CSV header (contains commas)
            if remainder and "," in remainder and not remainder.startswith("#"):
                clean_lines.append(remainder + "\n")
            # Otherwise skip the comment line entirely
        else:
            clean_lines.append(line)
    
    df = pd.read_csv(io.StringIO("".join(clean_lines)))
    
    # Ensure date column exists and convert to YYYY-MM-DD strings.
    # CSV may contain YYYY-MM-DD (from previous sync) or DD/MM/YYYY (from BFW/user edit).
    # Parse YYYY-MM-DD first (unambiguous), then DD/MM/YYYY for remaining.
    if "date" in df.columns:
        iso = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        dmy = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
        parsed = iso.fillna(dmy)
        n_iso = iso.notna().sum()
        n_dmy = (iso.isna() & dmy.notna()).sum()
        n_fail = parsed.isna().sum()
        if n_dmy > 0 or n_fail > 0:
            print(f"  Date parsing: {n_iso} ISO, {n_dmy} DD/MM, {n_fail} failed")
        df["date"] = parsed.dt.strftime("%Y-%m-%d")
    
    return df


def fetch_weight_from_intervals(client: IntervalsClient) -> dict:
    """
    Fetch weight data from intervals.icu wellness API.
    
    Returns:
        Dict of {date_str: weight_kg}
    """
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"  Fetching weight data from intervals.icu ({INTERVALS_WEIGHT_START} -> {today})...")
    
    weight_data = client.get_weight_data(INTERVALS_WEIGHT_START, today)
    print(f"  -> {len(weight_data)} days with weight data")
    
    if weight_data:
        dates = sorted(weight_data.keys())
        weights = [weight_data[d] for d in dates]
        print(f"  Range: {min(weights):.1f} – {max(weights):.1f} kg")
        print(f"  Latest: {dates[-1]} = {weight_data[dates[-1]]:.1f} kg")
    
    return weight_data


def fetch_non_running_tss(client: IntervalsClient, oldest: str = "2013-01-01") -> dict:
    """
    Fetch non-running TSS from intervals.icu activities using calorie-based formula.
    
    Formula: TSS = (calories - BMR_PER_HOUR * duration_hours) / TSS_DIVISOR
    
    Uses elapsed_time (not moving_time) for BMR subtraction since your body
    burns BMR calories for the entire duration, including rest periods.
    
    Returns:
        Dict of {date_str: total_tss}
    """
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"  Fetching non-running activities ({oldest} -> {today})...")
    
    nr_activities = client.get_non_running_activities(oldest, today)
    print(f"  -> {len(nr_activities)} non-running activities with calories")
    
    # Aggregate by date
    tss_by_date = defaultdict(float)
    sport_counts = defaultdict(int)
    skipped = 0
    
    for act in nr_activities:
        date_str = act.get("start_date_local", "")[:10]
        calories = act.get("calories", 0) or 0
        # elapsed_time in seconds; fall back to moving_time if not available
        elapsed_s = act.get("elapsed_time") or act.get("moving_time") or 0
        sport = act.get("type", "Unknown")
        
        if not date_str or calories <= 0 or elapsed_s <= 0:
            skipped += 1
            continue
        
        duration_hours = elapsed_s / 3600.0
        bmr_calories = BMR_PER_HOUR * duration_hours
        net_calories = max(0, calories - bmr_calories)
        tss = net_calories / TSS_DIVISOR
        
        if tss > 0:
            tss_by_date[date_str] += tss
            sport_counts[sport] += 1
    
    # Show sport breakdown
    if sport_counts:
        print(f"  Sport types: {', '.join(f'{s}({n})' for s, n in sorted(sport_counts.items(), key=lambda x: -x[1]))}")
    if skipped:
        print(f"  Skipped: {skipped} (missing calories or duration)")
    
    total_tss = sum(tss_by_date.values())
    print(f"  Formula: (calories - {BMR_PER_HOUR}/hr × elapsed_hours) / {TSS_DIVISOR}")
    print(f"  Total non-running TSS: {total_tss:.0f} across {len(tss_by_date)} days")
    
    return dict(tss_by_date)


def postprocess_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process weight column to produce daily smoothed values.
    
    Steps:
    1. Create daily date range from first weight to today + WEIGHT_PREFILL_DAYS
    2. Forward-fill gaps (carry last measurement forward)
    3. Apply 7-day centred moving average (±3 days around each date)
    
    The smoothed weight is written back to weight_kg. The raw measurements
    are preserved in weight_kg_raw for reference.
    """
    if "weight_kg" not in df.columns:
        return df
    
    weight_rows = df[df["weight_kg"].notna()].copy()
    if len(weight_rows) == 0:
        return df
    
    # Build daily date range: first weight -> today + prefill days
    first_date = pd.to_datetime(weight_rows["date"].min())
    last_date = pd.Timestamp.now().normalize() + timedelta(days=WEIGHT_PREFILL_DAYS)
    
    daily_dates = pd.date_range(first_date, last_date, freq="D")
    daily_df = pd.DataFrame({"date": daily_dates.strftime("%Y-%m-%d")})
    
    # Merge raw weights onto daily grid
    raw_weights = dict(zip(weight_rows["date"], weight_rows["weight_kg"].astype(float)))
    daily_df["weight_raw"] = daily_df["date"].map(raw_weights)
    
    # Forward-fill: carry last measurement forward
    daily_df["weight_filled"] = daily_df["weight_raw"].ffill()
    
    # 7-day centred moving average (min_periods=1 so edges still get values)
    daily_df["weight_smooth"] = (
        daily_df["weight_filled"]
        .rolling(window=WEIGHT_SMOOTH_WINDOW, center=True, min_periods=1)
        .mean()
        .round(1)
    )
    
    # Build lookup: date -> smoothed weight
    smooth_lookup = dict(zip(daily_df["date"], daily_df["weight_smooth"]))
    
    # Merge back: update existing rows + add new daily rows
    existing_dates = set(df["date"].dropna())
    new_rows = []
    
    for _, row in daily_df.iterrows():
        d = row["date"]
        if d not in existing_dates and pd.notna(row["weight_smooth"]):
            new_rows.append({
                "date": d,
                "weight_kg": row["weight_smooth"],
                "garmin_tr": np.nan,
                "non_running_tss": np.nan,
            })
    
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    
    # Apply smoothed weight to all rows that fall in the daily range
    _sample_key = list(smooth_lookup.keys())[0] if smooth_lookup else None
    _sample_date = df["date"].iloc[0] if len(df) > 0 else None
    print(f"    smooth_lookup key type: {type(_sample_key)}, df date type: {type(_sample_date)}")
    print(f"    smooth_lookup has 2025-12-01: {'2025-12-01' in smooth_lookup}")
    if "2025-12-01" in smooth_lookup:
        print(f"    smooth_lookup[2025-12-01] = {smooth_lookup['2025-12-01']}")
    _dec1_rows = df[df["date"] == "2025-12-01"]
    if len(_dec1_rows):
        print(f"    df Dec 1 before smooth: weight_kg={_dec1_rows['weight_kg'].values}")
    _mapped = df["date"].map(smooth_lookup)
    _n_mapped = _mapped.notna().sum()
    _n_total = len(df)
    print(f"    map result: {_n_mapped}/{_n_total} mapped successfully")
    df["weight_kg"] = _mapped.combine_first(df["weight_kg"])
    _dec1_after = df[df["date"] == "2025-12-01"]
    if len(_dec1_after):
        print(f"    df Dec 1 after smooth: weight_kg={_dec1_after['weight_kg'].values}")
    df = df.sort_values("date").reset_index(drop=True)
    
    # Stats
    n_raw = len(raw_weights)
    n_filled = int(daily_df["weight_filled"].notna().sum())
    latest_smooth = daily_df[daily_df["weight_smooth"].notna()].iloc[-1]
    print(f"  Weight post-processing:")
    print(f"    Raw measurements: {n_raw}")
    print(f"    Daily filled range: {daily_df['date'].iloc[0]} -> {daily_df['date'].iloc[-1]} ({n_filled} days)")
    print(f"    Smoothing: {WEIGHT_SMOOTH_WINDOW}-day centred average")
    print(f"    Latest smoothed: {latest_smooth['weight_smooth']:.1f} kg ({latest_smooth['date']})")
    
    return df


def merge_data(existing_df: pd.DataFrame, 
               weight_intervals: dict, 
               nr_tss_intervals: dict) -> pd.DataFrame:
    """
    Merge existing athlete_data.csv with intervals.icu data.
    
    Strategy:
    - Weight pre-2023: keep existing CSV values (from BFW export)
    - Weight 2023+: overwrite with intervals.icu values
    - Non-running TSS: always use intervals.icu (more complete than BFW)
    - garmin_tr: preserve existing values (dropped for new data — not available via API)
    """
    print("\n--- Merging data ---")
    
    # Build a comprehensive date index
    all_dates = set()
    
    if len(existing_df) > 0 and "date" in existing_df.columns:
        all_dates.update(existing_df["date"].dropna().tolist())
    all_dates.update(weight_intervals.keys())
    all_dates.update(nr_tss_intervals.keys())
    
    all_dates = sorted(d for d in all_dates if d and len(d) == 10)  # valid YYYY-MM-DD
    
    print(f"  Total date range: {all_dates[0]} -> {all_dates[-1]}")
    print(f"  Total unique dates: {len(all_dates)}")
    
    # Create existing lookups
    existing_weight = {}
    existing_garmin_tr = {}
    existing_nr_tss = {}
    
    if len(existing_df) > 0:
        for _, row in existing_df.iterrows():
            d = row.get("date", "")
            if not d or pd.isna(d):
                continue
            
            w = row.get("weight_kg")
            if pd.notna(w) and w > 0:
                existing_weight[d] = float(w)
            
            tr = row.get("garmin_tr")
            if pd.notna(tr):
                existing_garmin_tr[d] = float(tr)
            
            tss = row.get("non_running_tss")
            if pd.notna(tss) and tss > 0:
                existing_nr_tss[d] = float(tss)
    
    # Merge logic
    merged_rows = []
    weight_updates = 0
    weight_preserved = 0
    tss_updates = 0
    
    for date_str in all_dates:
        row = {"date": date_str}
        
        # Weight: pre-2023 from existing, 2023+ ONLY from intervals.icu
        # v51.6: Don't fall back to existing CSV for 2023+ dates — avoids
        # persisting corrupted values from date-format parsing bugs.
        # postprocess_weight will fill gaps via forward-fill + smoothing.
        if date_str >= INTERVALS_WEIGHT_START and date_str in weight_intervals:
            row["weight_kg"] = weight_intervals[date_str]
            weight_updates += 1
        elif date_str < INTERVALS_WEIGHT_START and date_str in existing_weight:
            row["weight_kg"] = existing_weight[date_str]
            weight_preserved += 1
        else:
            row["weight_kg"] = np.nan
        
        # Garmin TR: preserve from existing (not available from intervals.icu)
        if date_str in existing_garmin_tr:
            row["garmin_tr"] = existing_garmin_tr[date_str]
        else:
            row["garmin_tr"] = np.nan
        
        # Non-running TSS: always prefer intervals.icu (more complete)
        if date_str in nr_tss_intervals:
            row["non_running_tss"] = nr_tss_intervals[date_str]
            tss_updates += 1
        elif date_str in existing_nr_tss:
            row["non_running_tss"] = existing_nr_tss[date_str]
        else:
            row["non_running_tss"] = np.nan
        
        # Only include rows that have at least one value
        if pd.notna(row["weight_kg"]) or pd.notna(row["garmin_tr"]) or pd.notna(row["non_running_tss"]):
            merged_rows.append(row)
    
    merged_df = pd.DataFrame(merged_rows)
    
    print(f"  Weight: {weight_updates} from intervals.icu, {weight_preserved} preserved from CSV")
    print(f"  Non-running TSS: {tss_updates} from intervals.icu")
    print(f"  Garmin TR: {len(existing_garmin_tr)} preserved from CSV (not available via API)")
    print(f"  Output rows: {len(merged_df)}")
    
    return merged_df


def write_athlete_data(df: pd.DataFrame, path: str):
    """Write athlete_data.csv with header comments."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    with open(path, "w", newline="") as f:
        f.write(f"# Athlete data synced from intervals.icu on {now}\n")
        f.write(f"# Weight: intervals.icu (2023+) + legacy CSV (pre-2023)\n")
        f.write(f"# Non-running TSS: (calories - 80/hr * elapsed_hours) / 10\n")
    
    df.to_csv(path, mode="a", index=False)
    print(f"\n[OK] Written: {path} ({len(df)} rows)")


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    weight_rows = df[df["weight_kg"].notna()]
    tr_rows = df[df["garmin_tr"].notna()] if "garmin_tr" in df.columns else pd.DataFrame()
    tss_rows = df[df["non_running_tss"].notna()]
    
    print(f"  Total rows:          {len(df)}")
    print(f"  Rows with weight:    {len(weight_rows)}")
    print(f"  Rows with Garmin TR: {len(tr_rows)}")
    print(f"  Rows with NR TSS:    {len(tss_rows)}")
    
    if len(weight_rows) > 0:
        today = datetime.now().strftime("%Y-%m-%d")
        current_weight = weight_rows[weight_rows["date"] <= today]
        if len(current_weight) > 0:
            latest = current_weight.iloc[-1]
            print(f"\n  Latest weight: {latest['weight_kg']:.1f} kg ({latest['date']})")
            future = weight_rows[weight_rows["date"] > today]
            if len(future) > 0:
                print(f"  (+ {len(future)} future pre-filled days to {weight_rows.iloc[-1]['date']})")
    
    if len(tss_rows) > 0:
        latest = tss_rows.iloc[-1]
        print(f"  Latest NR TSS: {latest['non_running_tss']:.0f} ({latest['date']})")
        
        # Annual totals
        tss_rows = tss_rows.copy()
        tss_rows["year"] = tss_rows["date"].str[:4]
        yearly = tss_rows.groupby("year")["non_running_tss"].agg(["sum", "count"])
        print(f"\n  Non-running TSS by year:")
        for year, row in yearly.iterrows():
            print(f"    {year}: {row['sum']:.0f} TSS across {int(row['count'])} days")


def main():
    parser = argparse.ArgumentParser(
        description="Sync weight + non-running TSS from intervals.icu to athlete_data.csv"
    )
    parser.add_argument(
        "--athlete-data",
        default=DEFAULT_ATHLETE_DATA,
        help=f"Path to athlete_data.csv (default: {DEFAULT_ATHLETE_DATA})"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="intervals.icu API key (or set INTERVALS_API_KEY)"
    )
    parser.add_argument(
        "--athlete-id",
        default=None,
        help="intervals.icu athlete ID (or set INTERVALS_ATHLETE_ID)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing"
    )
    parser.add_argument(
        "--nr-tss-oldest",
        default="2013-01-01",
        help="Oldest date for non-running TSS fetch (default: 2013-01-01)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Sync Athlete Data from intervals.icu")
    print("=" * 60)
    
    # Initialize client
    try:
        client = IntervalsClient(api_key=args.api_key, athlete_id=args.athlete_id)
    except ValueError as e:
        print(f"\nERROR: {e}")
        return 1
    
    # Test connection
    print("\nTesting connection...")
    test = client.test_connection()
    if not test.get("ok"):
        print(f"  Connection failed: {test.get('error', 'unknown')}")
        return 1
    print(f"  Connected — {test.get('activities_7d', 0)} activities in last 7 days")
    
    # Read existing data
    print(f"\n--- Reading existing {args.athlete_data} ---")
    existing_df = read_existing_athlete_data(args.athlete_data)
    print(f"  Existing rows: {len(existing_df)}")
    
    # Fetch from intervals.icu
    print("\n--- Fetching from intervals.icu ---")
    weight_data = fetch_weight_from_intervals(client)
    nr_tss_data = fetch_non_running_tss(client, oldest=args.nr_tss_oldest)
    
    # Merge
    merged_df = merge_data(existing_df, weight_data, nr_tss_data)
    
    # Post-process weight: forward-fill + 7-day centred average
    print("\n--- Post-processing weight ---")
    merged_df = postprocess_weight(merged_df)
    
    # Summary
    print_summary(merged_df)
    
    # Write
    if args.dry_run:
        print(f"\n[DRY RUN] Would write {len(merged_df)} rows to {args.athlete_data}")
    else:
        write_athlete_data(merged_df, args.athlete_data)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
