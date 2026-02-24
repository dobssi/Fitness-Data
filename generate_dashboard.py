"""
Mobile Dashboard Generator
==========================
Generates a mobile-friendly HTML dashboard from your Master file.

v46: - REMOVED BFW dependency completely
     - Now reads weight from athlete_data.csv
     - athlete_data.csv can be exported from BFW or manually maintained

v46: - Added Top Race Performances section with 1Y/2Y/3Y/5Y/All time filters
     - Data date now shows today if viewing after last run (CTL/ATL decay)
     - Continued from v44.5 - Power Score affects RF_adj directly

v44: Dashboard now reads most data from Master file's summary sheets:
     - Daily sheet: CTL/ATL/TSB, RFL_Trend (with forward-fill for rest days)
     - Weekly sheet: Weekly RFL_Trend
     - Monthly/Yearly sheets: Volume summaries

Data sources:
- Master file: Run data, CTL/ATL/TSB, RF_Trend, RFL_Trend, CP, race predictions
- athlete_data.csv: Weight (uses last available 7-day smoothed value)

To run:
    pip install pandas openpyxl
    python generate_dashboard.py

Output:
    index.html - Open in any browser (mobile or desktop)

Configuration:
    Edit MASTER_FILE and ATHLETE_DATA_FILE paths below.
"""

import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import json
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
# Paths: env vars override defaults (for multi-athlete support)
MASTER_FILE = os.environ.get("MASTER_FILE", r"Master_FULL_GPSQ_ID_post.xlsx")
ATHLETE_DATA_FILE = os.environ.get("ATHLETE_DATA_FILE", r"athlete_data.csv")
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", r"index.html")

# ============================================================================
# DATA PROCESSING
# ============================================================================
def load_and_process_data():
    """Load Master and athlete_data files, prepare data for dashboard."""
    
    # --- Load from Master (v43 columns) ---
    print(f"Loading {MASTER_FILE}...")
    df = pd.read_excel(MASTER_FILE, sheet_name='Master')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"Loaded {len(df)} runs")
    
    # Get CTL/ATL/TSB from Master (latest row with valid values)
    print(f"Loading CTL/ATL from Master...")
    ctl, atl, tsb = None, None, None
    if 'CTL' in df.columns:
        df_valid = df[df['CTL'].notna()]
        if len(df_valid) > 0:
            latest = df_valid.iloc[-1]
            ctl = round(latest['CTL'], 1) if pd.notna(latest.get('CTL')) else None
            atl = round(latest['ATL'], 1) if pd.notna(latest.get('ATL')) else None
            tsb = round(latest['TSB'], 1) if pd.notna(latest.get('TSB')) else None
            print(f"  Found: CTL={ctl}, ATL={atl}, TSB={tsb} for {latest['date'].strftime('%Y-%m-%d')}")
    
    # --- Load from athlete_data.csv (weight) ---
    print(f"Loading athlete data from '{ATHLETE_DATA_FILE}'...")
    weight = None
    
    if os.path.exists(ATHLETE_DATA_FILE):
        try:
            # Read CSV, handling malformed comment headers from BFW export
            # (comment lines and header can get concatenated onto one line)
            import io
            with open(ATHLETE_DATA_FILE, 'r') as f:
                raw_lines = f.readlines()
            clean_lines = []
            for line in raw_lines:
                stripped = line.lstrip()
                if stripped.startswith('#'):
                    # BFW export bug: header sometimes concatenated after comment
                    # e.g. '# comment#date,weight_kg,garmin_tr,non_running_tss'
                    last_hash = stripped.rfind('#')
                    remainder = stripped[last_hash + 1:].strip()
                    # Only treat as data if it looks like a CSV header (contains commas)
                    if remainder and remainder.startswith('date,'):
                        clean_lines.append(remainder + '\n')
                    # Otherwise skip the comment line entirely
                else:
                    clean_lines.append(line)
            ad_df = pd.read_csv(io.StringIO(''.join(clean_lines)))
            ad_df['date'] = pd.to_datetime(ad_df['date'], format='%Y-%m-%d', errors='coerce')
            ad_df = ad_df.sort_values('date')
            today = pd.Timestamp.now().normalize()
            
            # Get last available weight (up to today — smoothed 7-day centred average)
            if 'weight_kg' in ad_df.columns:
                weight_rows = ad_df[(ad_df['weight_kg'].notna()) & (ad_df['date'] <= today)]
                if len(weight_rows) > 0:
                    weight = round(float(weight_rows.iloc[-1]['weight_kg']), 1)
                    weight_date = weight_rows.iloc[-1]['date'].strftime('%Y-%m-%d')
                    print(f"  Found: Weight={weight}kg (from {weight_date})")
            
        except Exception as e:
            print(f"Warning: Could not load athlete data: {e}")
    else:
        print(f"  Note: {ATHLETE_DATA_FILE} not found - weight will be blank")
    
    # --- Get CP, race predictions, and age grade from Master (v43 columns) ---
    critical_power = None
    age_grade = None
    race_predictions = {}
    
    latest = df.iloc[-1] if len(df) > 0 else None
    if latest is not None:
        # CP
        cp_val = latest.get('CP')
        if pd.notna(cp_val):
            critical_power = int(round(float(cp_val)))
        
        # Predicted 5K age grade
        ag_val = latest.get('pred_5k_age_grade')
        if pd.notna(ag_val):
            age_grade = round(float(ag_val), 2)
        
        # Race predictions (in seconds, need to format)
        def format_seconds(s):
            if pd.isna(s) or s <= 0:
                return '-'
            s = int(s)
            hours = s // 3600
            mins = (s % 3600) // 60
            secs = s % 60
            if hours > 0:
                return f"{hours}:{mins:02d}:{secs:02d}"
            else:
                return f"{mins}:{secs:02d}"
        
        pred_5k = latest.get('pred_5k_s')
        pred_10k = latest.get('pred_10k_s')
        pred_hm = latest.get('pred_hm_s')
        pred_mara = latest.get('pred_marathon_s')
        
        if pd.notna(pred_5k):
            race_predictions['5k'] = format_seconds(pred_5k)
            race_predictions['5k_raw'] = int(pred_5k)
        if pd.notna(pred_10k):
            race_predictions['10k'] = format_seconds(pred_10k)
            race_predictions['10k_raw'] = int(pred_10k)
        if pd.notna(pred_hm):
            race_predictions['Half Marathon'] = format_seconds(pred_hm)
            race_predictions['hm_raw'] = int(pred_hm)
        if pd.notna(pred_mara):
            race_predictions['Marathon'] = format_seconds(pred_mara)
            race_predictions['marathon_raw'] = int(pred_mara)
        
        print(f"  From Master: CP={critical_power}W, Age Grade={age_grade}%")
        print(f"  Race predictions: {race_predictions}")
        
        # Phase 2: GAP and Sim predictions for dashboard mode toggle
        for mode in ('gap', 'sim'):
            mode_preds = {}
            for dist, col, raw_key in [('5k', f'pred_5k_s_{mode}', '5k_raw'),
                                       ('10k', f'pred_10k_s_{mode}', '10k_raw'),
                                       ('Half Marathon', f'pred_hm_s_{mode}', 'hm_raw'),
                                       ('Marathon', f'pred_marathon_s_{mode}', 'marathon_raw')]:
                val = latest.get(col)
                if pd.notna(val):
                    mode_preds[dist] = format_seconds(val)
                    mode_preds[raw_key] = int(val)
            race_predictions[f'_mode_{mode}'] = mode_preds
            
            cp_mode = latest.get(f'CP_{mode}')
            race_predictions[f'_cp_{mode}'] = int(round(float(cp_mode))) if pd.notna(cp_mode) else None
            
            ag_mode = latest.get(f'pred_5k_age_grade_{mode}')
            race_predictions[f'_ag_{mode}'] = round(float(ag_mode), 2) if pd.notna(ag_mode) else None
    
    return df, ctl, atl, tsb, weight, age_grade, critical_power, race_predictions

def format_race_time(time_str):
    """Format race time, removing leading 00: for times under an hour."""
    if not time_str or time_str == '-':
        return '-'
    time_str = str(time_str)
    if time_str.startswith('00:'):
        return time_str[3:]  # Remove "00:"
    return time_str


def get_daily_ctl_atl_lookup(master_file):
    """Get daily CTL/ATL/TSB lookup dict for JavaScript to pick the right day.
    Returns dict: date_str -> {ctl, atl, tsb}
    Includes projected future days (with TSS=0 decay)."""
    try:
        df = pd.read_excel(master_file, sheet_name='Daily')
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Build lookup dict
        lookup = {}
        for _, row in df.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d')
            lookup[date_str] = {
                'ctl': round(row['CTL'], 1) if pd.notna(row.get('CTL')) else None,
                'atl': round(row['ATL'], 1) if pd.notna(row.get('ATL')) else None,
                'tsb': round(row['TSB'], 1) if pd.notna(row.get('TSB')) else None,
            }
        
        print(f"  Loaded {len(lookup)} days of CTL/ATL lookup data")
        return lookup
    except Exception as e:
        print(f"  Warning: Could not load daily CTL/ATL lookup: {e}")
        return {}


def _calc_rfl_delta(df, days_back=14, mode='stryd'):
    """Calculate change in RFL_Trend vs N days ago.
    Returns delta in percentage points (e.g. +2.5 or -1.3), or None."""
    rfl_col = {'gap': 'RFL_gap_Trend', 'sim': 'RFL_sim_Trend'}.get(mode, 'RFL_Trend')
    rfl = pd.to_numeric(df.get(rfl_col), errors='coerce')
    dates = pd.to_datetime(df.get('date'), errors='coerce')
    if rfl.isna().all() or dates.isna().all():
        return None
    
    latest_rfl = rfl.iloc[-1] if pd.notna(rfl.iloc[-1]) else None
    if latest_rfl is None:
        return None
    
    cutoff = dates.max() - timedelta(days=days_back)
    # Find the last run on or before the cutoff
    past_mask = dates <= cutoff
    if not past_mask.any():
        return None
    
    past_rfl = rfl[past_mask].dropna()
    if past_rfl.empty:
        return None
    
    old_rfl = past_rfl.iloc[-1]
    delta = (latest_rfl - old_rfl) * 100  # Convert to percentage points
    return round(delta, 1)


def get_summary_stats(df):
    """Calculate summary statistics."""
    now = datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)
    year_ago = now - timedelta(days=365)
    
    week_runs = df[df['date'] >= week_ago]
    month_runs = df[df['date'] >= month_ago]
    year_runs = df[df['date'] >= year_ago]
    
    latest = df.iloc[-1] if len(df) > 0 else None
    
    # Master uses distance_km, not Distance_m
    dist_col = 'distance_km' if 'distance_km' in df.columns else 'Distance_m'
    hr_col = 'avg_hr' if 'avg_hr' in df.columns else 'Avg_HR'
    
    def get_dist_km(subset):
        if dist_col == 'distance_km':
            return round(subset[dist_col].sum(), 1) if dist_col in subset.columns else 0
        else:
            return round(subset[dist_col].sum() / 1000, 1) if dist_col in subset.columns else 0
    
    # Determine data_date: if today is after latest run, show today (CTL/ATL have decayed)
    latest_run_date = df['date'].max() if len(df) > 0 else None
    if latest_run_date is not None and pd.Timestamp(today) > latest_run_date:
        data_date = today.strftime('%Y-%m-%d')
    else:
        data_date = latest_run_date.strftime('%Y-%m-%d') if latest_run_date is not None else '-'
    
    stats = {
        'total_runs': len(df),
        'latest_date': df['date'].max().strftime('%Y-%m-%d') if len(df) > 0 else '-',
        'data_date': data_date,  # v46: today's date if viewing after last run (for CTL/ATL)
        'week_runs': len(week_runs),
        'week_km': get_dist_km(week_runs),
        'month_runs': len(month_runs),
        'month_km': get_dist_km(month_runs),
        'year_km': round(get_dist_km(year_runs), 0),
        # v44.5: Use RFL_Trend (Power Score now affects RF_adj directly, no separate RFL_Combined)
        'latest_rf': round(latest.get('RF_Trend'), 2) if latest is not None and pd.notna(latest.get('RF_Trend')) else None,
        'latest_rfl': round((latest.get('RFL_Trend') or 0) * 100, 1) if latest is not None and pd.notna(latest.get('RFL_Trend')) else None,
        'rfl_14d_delta': _calc_rfl_delta(df, 14, 'stryd'),
        'rfl_14d_delta_gap': _calc_rfl_delta(df, 14, 'gap'),
        'rfl_14d_delta_sim': _calc_rfl_delta(df, 14, 'sim'),
        'latest_hr': int(latest[hr_col]) if latest is not None and pd.notna(latest.get(hr_col)) else None,
        'latest_dist': round(latest[dist_col], 2) if latest is not None and pd.notna(latest.get(dist_col)) else None,
        # v51: Easy RF gap
        'easy_rfl_gap': round(latest.get('Easy_RFL_Gap', 0) * 100, 1) if latest is not None and pd.notna(latest.get('Easy_RFL_Gap')) else None,
        'easy_rfl_gap_gap': round(latest.get('Easy_RFL_Gap_gap', 0) * 100, 1) if latest is not None and pd.notna(latest.get('Easy_RFL_Gap_gap')) else None,
        'easy_rfl_gap_sim': round(latest.get('Easy_RFL_Gap_sim', 0) * 100, 1) if latest is not None and pd.notna(latest.get('Easy_RFL_Gap_sim')) else None,
        # Phase 2: GAP and Sim RFL for mode toggle
        'latest_rfl_gap': round((latest.get('RFL_gap_Trend') or 0) * 100, 1) if latest is not None and pd.notna(latest.get('RFL_gap_Trend')) else None,
        'latest_rfl_sim': round((latest.get('RFL_sim_Trend') or 0) * 100, 1) if latest is not None and pd.notna(latest.get('RFL_sim_Trend')) else None,
        'first_year': df['date'].min().year if len(df) > 0 else '',
    }
    return stats

def get_rfl_trend_data(df, days=90):
    """Get RFL trend data for chart. Returns exactly 'days' worth of data.
    Shows RFL as percentage (0-100%) instead of RF in W/bpm.
    v51: Also returns Easy_RF_EMA normalised to RFL scale and race flags."""
    # Use end of today for inclusive boundary
    today = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
    cutoff = today - timedelta(days=days)
    recent = df[(df['date'] > cutoff) & (df['date'] <= today)].copy()
    
    # Individual run RFL (RF_adj / peak_RF)
    rfl_col = 'RFL'  
    # v44.5: Use RFL_Trend (no more RFL_Combined)
    trend_col = 'RFL_Trend'
    
    if rfl_col not in recent.columns and trend_col not in recent.columns:
        return [], [], [], [], [], [], [], [], []
    
    cols_to_keep = ['date']
    if rfl_col in recent.columns:
        cols_to_keep.append(rfl_col)
    if trend_col in recent.columns:
        cols_to_keep.append(trend_col)
    # v51: Include Easy_RF_EMA and race_flag
    has_easy = 'Easy_RF_EMA' in recent.columns
    if has_easy:
        cols_to_keep.append('Easy_RF_EMA')
    has_race = 'race_flag' in recent.columns
    if has_race:
        cols_to_keep.append('race_flag')
    # Phase 2: Include GAP/Sim columns
    for gc in ['RFL_gap_Trend', 'RFL_sim_Trend', 'RFL_gap', 'RFL_sim']:
        if gc in recent.columns:
            cols_to_keep.append(gc)
    
    recent = recent[cols_to_keep].copy()
    
    # Drop rows where RFL is NaN
    if rfl_col in recent.columns:
        recent = recent.dropna(subset=[rfl_col])
    elif trend_col in recent.columns:
        recent = recent.dropna(subset=[trend_col])
    
    # Use shorter date format for mobile: "15 Jan 25"
    dates = recent['date'].dt.strftime('%d %b %y').tolist()
    
    # Convert to percentage (0-100)
    if rfl_col in recent.columns:
        rfl_values = [round(v * 100, 1) if pd.notna(v) else None for v in recent[rfl_col].tolist()]
    else:
        rfl_values = [None] * len(dates)
    
    if trend_col in recent.columns:
        rfl_trend = [round(v * 100, 1) if pd.notna(v) else None for v in recent[trend_col].tolist()]
    else:
        rfl_trend = [None] * len(dates)
    
    # v51: Normalise Easy_RF_EMA to RFL scale (divide by peak RF_Trend)
    easy_rfl = [None] * len(dates)
    if has_easy:
        peak_rf = df['RF_Trend'].max() if 'RF_Trend' in df.columns else None
        if peak_rf and peak_rf > 0:
            easy_rfl = [round(v / peak_rf * 100, 1) if pd.notna(v) else None 
                       for v in recent['Easy_RF_EMA'].tolist()]
    
    # v51: Race flags (1 = race, 0 = training)
    race_flags = [0] * len(dates)
    if has_race:
        race_flags = [int(v) if pd.notna(v) and v == 1 else 0 for v in recent['race_flag'].tolist()]
    
    # Phase 2: Extract GAP and Sim RFL_Trend for mode toggle
    gap_trend = [None] * len(dates)
    sim_trend = [None] * len(dates)
    if 'RFL_gap_Trend' in recent.columns:
        gap_trend = [round(v * 100, 1) if pd.notna(v) else None for v in recent['RFL_gap_Trend'].tolist()]
    if 'RFL_sim_Trend' in recent.columns:
        sim_trend = [round(v * 100, 1) if pd.notna(v) else None for v in recent['RFL_sim_Trend'].tolist()]
    # Per-run RFL for GAP and Sim
    gap_values = [None] * len(dates)
    sim_values = [None] * len(dates)
    if 'RFL_gap' in recent.columns:
        gap_values = [round(v * 100, 1) if pd.notna(v) else None for v in recent['RFL_gap'].tolist()]
    if 'RFL_sim' in recent.columns:
        sim_values = [round(v * 100, 1) if pd.notna(v) else None for v in recent['RFL_sim'].tolist()]
    
    return dates, rfl_values, rfl_trend, easy_rfl, race_flags, gap_trend, sim_trend, gap_values, sim_values

def get_weekly_volume(df, weeks=12):
    """Get weekly volume data for chart. Returns exactly 'weeks' worth of data, rolling from today."""
    # Use end of today for inclusive boundary
    today = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
    cutoff = today - timedelta(weeks=weeks)
    recent = df[(df['date'] > cutoff) & (df['date'] <= today)].copy()
    
    # Determine distance column
    dist_col = 'distance_km' if 'distance_km' in recent.columns else 'Distance_m'
    
    recent['week'] = recent['date'].dt.to_period('W').dt.start_time
    weekly = recent.groupby('week').agg({
        dist_col: 'sum',
        'date': 'count'
    }).rename(columns={'date': 'runs'}).reset_index()
    
    # Convert to km if needed
    if dist_col == 'distance_km':
        weekly['distance_km'] = weekly[dist_col]
    else:
        weekly['distance_km'] = weekly[dist_col] / 1000
    
    # Only keep the most recent N weeks
    weekly = weekly.sort_values('week').tail(weeks)
    
    labels = weekly['week'].dt.strftime('%d %b').tolist()
    distances = [round(v, 1) for v in weekly['distance_km'].tolist()]
    runs = weekly['runs'].tolist()
    
    return labels, distances, runs


def get_monthly_volume(df, months=None):
    """Get monthly volume data for chart. If months is None, return all history.
    Shows the most recent N calendar months (including current partial month)."""
    today = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
    recent = df[df['date'] <= today].copy()
    
    # Determine distance column
    dist_col = 'distance_km' if 'distance_km' in recent.columns else 'Distance_m'
    
    recent['month'] = recent['date'].dt.to_period('M').dt.start_time
    monthly = recent.groupby('month').agg({
        dist_col: 'sum',
        'date': 'count'
    }).rename(columns={'date': 'runs'}).reset_index()
    
    # Convert to km if needed
    if dist_col == 'distance_km':
        monthly['distance_km'] = monthly[dist_col]
    else:
        monthly['distance_km'] = monthly[dist_col] / 1000
    monthly = monthly.sort_values('month')
    
    # Only keep the most recent N months
    if months is not None:
        monthly = monthly.tail(months)
    
    labels = monthly['month'].dt.strftime('%b %y').tolist()
    distances = [round(v, 1) for v in monthly['distance_km'].tolist()]
    runs = monthly['runs'].tolist()
    
    return labels, distances, runs


def get_yearly_volume(df, years=None):
    """Get yearly volume data for chart. If years is None, return all calendar years.
    Otherwise shows rolling 12-month periods ending on today's date.
    E.g. for 3 years on 26 Jan 2026: 27/1/23-26/1/24, 27/1/24-26/1/25, 27/1/25-26/1/26"""
    today = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
    
    # Determine distance column
    dist_col = 'distance_km' if 'distance_km' in df.columns else 'Distance_m'
    dist_divisor = 1 if dist_col == 'distance_km' else 1000
    
    if years is None:
        # All time - use calendar years
        recent = df[df['date'] <= today].copy()
        recent['year'] = recent['date'].dt.to_period('Y').dt.start_time
        yearly = recent.groupby('year').agg({
            dist_col: 'sum',
            'date': 'count'
        }).rename(columns={'date': 'runs'}).reset_index()
        
        yearly['distance_km'] = yearly[dist_col] / dist_divisor
        yearly = yearly.sort_values('year')
        
        labels = yearly['year'].dt.strftime('%Y').tolist()
        distances = [round(v, 0) for v in yearly['distance_km'].tolist()]
        runs = yearly['runs'].tolist()
    else:
        # Rolling 12-month periods ending on today
        labels = []
        distances = []
        runs = []
        
        for i in range(years, 0, -1):
            # Period ends i-1 years ago from today, starts i years ago from today
            period_end = today - timedelta(days=(i-1) * 365)
            period_start = today - timedelta(days=i * 365)
            
            period_df = df[(df['date'] > period_start) & (df['date'] <= period_end)]
            
            dist_km = period_df[dist_col].sum() / dist_divisor if len(period_df) > 0 else 0
            run_count = len(period_df)
            
            # Label shows "To dd/mm/yy" for the period end date
            label = f"To {period_end.strftime('%d/%m/%y')}"
            
            labels.append(label)
            distances.append(round(dist_km, 0))
            runs.append(run_count)
    
    return labels, distances, runs


def get_ctl_atl_trend(df, days=90):
    """Get CTL/ATL time series data for chart from Master's Daily sheet.
    Returns all calendar days (not just run days) with their CTL/ATL values.
    Also returns projection data for future days (dashed line)."""
    try:
        # Read from Daily sheet which has all calendar days including future projections
        daily_df = pd.read_excel(MASTER_FILE, sheet_name='Daily')
        daily_df['Date'] = pd.to_datetime(daily_df['Date'])
        
        # Today boundary
        today = pd.Timestamp.now().normalize()
        cutoff = today - timedelta(days=days)
        
        # Historical data (up to and including today)
        df_hist = daily_df[(daily_df['Date'] > cutoff) & (daily_df['Date'] <= today)].copy()
        df_hist = df_hist[df_hist['CTL'].notna()]
        df_hist = df_hist.sort_values('Date')
        
        # Future projection data (after today, up to 14 days ahead for display)
        projection_days = 14
        future_end = today + timedelta(days=projection_days)
        df_future = daily_df[(daily_df['Date'] > today) & (daily_df['Date'] <= future_end)].copy()
        df_future = df_future[df_future['CTL'].notna()]
        df_future = df_future.sort_values('Date')
        
        if len(df_hist) == 0:
            print("  Warning: No CTL/ATL data found in date range")
            return [], [], [], [], [], []
        
        # Historical series
        dates = df_hist['Date'].dt.strftime('%d %b %y').tolist()
        ctl_values = [round(v, 1) if pd.notna(v) else None for v in df_hist['CTL'].tolist()]
        atl_values = [round(v, 1) if pd.notna(v) else None for v in df_hist['ATL'].tolist()]
        
        # Projection series (starts from last historical point for continuity)
        if len(df_future) > 0:
            proj_dates = df_future['Date'].dt.strftime('%d %b %y').tolist()
            proj_ctl = [round(v, 1) if pd.notna(v) else None for v in df_future['CTL'].tolist()]
            proj_atl = [round(v, 1) if pd.notna(v) else None for v in df_future['ATL'].tolist()]
            
            # Combine dates for x-axis
            all_dates = dates + proj_dates
            
            # Historical values with nulls for projection period
            ctl_with_nulls = ctl_values + [None] * len(proj_dates)
            atl_with_nulls = atl_values + [None] * len(proj_dates)
            
            # Projection values: null for historical except last point, then projection
            ctl_proj_line = [None] * (len(dates) - 1) + [ctl_values[-1]] + proj_ctl
            atl_proj_line = [None] * (len(dates) - 1) + [atl_values[-1]] + proj_atl
            
            print(f"  Loaded {len(dates)} days of CTL/ATL data + {len(proj_dates)} days projection")
            return all_dates, ctl_with_nulls, atl_with_nulls, ctl_proj_line, atl_proj_line
        else:
            print(f"  Loaded {len(dates)} days of CTL/ATL data (no projection available)")
            return dates, ctl_values, atl_values, [], []
        
    except Exception as e:
        print(f"  Warning: Could not load CTL/ATL trend: {e}")
        return [], [], [], [], []


def get_daily_rfl_trend(master_file, days=14, rfl_col='RFL_Trend'):
    """Get daily RFL trend from Master's Daily sheet with trendline, projection and 95% CI."""
    try:
        df = pd.read_excel(master_file, sheet_name='Daily')
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Get up to end of today
        today = pd.Timestamp.now().normalize() + timedelta(days=1) - timedelta(microseconds=1)
        df_past = df[df['Date'] <= today].tail(days)
        
        dates = df_past['Date'].dt.strftime('%d %b').tolist()
        # Use specified RFL column, fall back to RFL_Trend if not available
        if rfl_col not in df_past.columns:
            rfl_col = 'RFL_Trend'
        rfl_values = [round(v * 100, 2) if pd.notna(v) else None for v in df_past[rfl_col].tolist()]
        
        # Calculate linear trendline with confidence intervals
        valid_indices = [i for i, v in enumerate(rfl_values) if v is not None]
        if len(valid_indices) >= 2:
            x = np.array(valid_indices, dtype=float)
            y = np.array([rfl_values[i] for i in valid_indices], dtype=float)
            
            # Linear regression: y = mx + b
            n = len(x)
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
            intercept = y_mean - slope * x_mean
            
            # Calculate standard error for confidence intervals
            y_pred = slope * x + intercept
            residuals = y - y_pred
            mse = np.sum(residuals**2) / (n - 2) if n > 2 else 0
            se = np.sqrt(mse)
            
            # Sum of squared deviations from mean x
            ss_x = np.sum((x - x_mean)**2)
            
            # t-value for 95% CI (approximate for small samples)
            t_val = 2.0  # ~95% CI
            
            # Generate all x points (historical + projection)
            all_x = list(range(len(rfl_values))) + list(range(len(rfl_values), len(rfl_values) + 7))
            
            # Trendline values
            full_trendline = [round(slope * xi + intercept, 2) for xi in all_x]
            
            # Confidence intervals (wider as we extrapolate)
            ci_upper = []
            ci_lower = []
            for xi in all_x:
                # Standard error of prediction increases with distance from mean
                se_pred = se * np.sqrt(1 + 1/n + (xi - x_mean)**2 / ss_x) if ss_x > 0 else se
                margin = t_val * se_pred
                pred = slope * xi + intercept
                ci_upper.append(round(pred + margin, 2))
                ci_lower.append(round(pred - margin, 2))
            
            # Project dates 7 days forward
            projection_dates = []
            last_date = df_past['Date'].iloc[-1]
            for i in range(1, 8):
                proj_date = last_date + timedelta(days=i)
                projection_dates.append(proj_date.strftime('%d %b'))
            
            # Combine dates for full x-axis
            all_dates = dates + projection_dates
            
            # RFL values with nulls for projection period
            rfl_with_nulls = rfl_values + [None] * 7
            
            # Projection line (nulls for historical except last point, values for future)
            projection_line = [None] * (len(rfl_values) - 1) + full_trendline[len(rfl_values) - 1:]
            
            # CI lines (only for projection period, starting from last historical point)
            ci_upper_line = [None] * (len(rfl_values) - 1) + ci_upper[len(rfl_values) - 1:]
            ci_lower_line = [None] * (len(rfl_values) - 1) + ci_lower[len(rfl_values) - 1:]
            
            # Trendline only for historical period
            trendline_hist = full_trendline[:len(rfl_values)] + [None] * 7
            
        else:
            all_dates = dates
            rfl_with_nulls = rfl_values
            trendline_hist = [None] * len(rfl_values)
            projection_line = [None] * len(rfl_values)
            ci_upper_line = [None] * len(rfl_values)
            ci_lower_line = [None] * len(rfl_values)
        
        return all_dates, rfl_with_nulls, trendline_hist, projection_line, ci_upper_line, ci_lower_line
    except Exception as e:
        print(f"Warning: Could not load daily RFL trend: {e}")
        return [], [], [], [], [], []


def get_alltime_weekly_rfl(master_file):
    """Get all-time weekly RFL data from Master's Weekly sheet."""
    try:
        df = pd.read_excel(master_file, sheet_name='Weekly')
        
        # v44.5: Use RFL_Trend
        rfl_col = 'RFL_Trend'
        
        # Filter to rows with RFL data
        df = df[df[rfl_col].notna()].copy()
        
        # Create date labels from Week_Start
        df['Week_Start'] = pd.to_datetime(df['Week_Start'])
        dates = df['Week_Start'].dt.strftime('%d %b %y').tolist()
        
        # Convert to percentage (RFL is 0-1 scale)
        rfl_values = [round(v * 100, 1) for v in df[rfl_col].tolist()]
        
        return dates, rfl_values
    except Exception as e:
        print(f"Warning: Could not load all-time weekly RFL: {e}")
        return [], []

def get_recent_runs(df, n=10):
    """Get recent runs and recent races for table."""
    
    def extract_runs(subset):
        """Extract run dicts from a DataFrame subset."""
        # Determine column names
        dist_col = 'distance_km' if 'distance_km' in subset.columns else 'Distance_m'
        hr_col = 'avg_hr' if 'avg_hr' in subset.columns else 'Avg_HR'
        npower_col = 'npower_w' if 'npower_w' in subset.columns else 'nPower'
        moving_col = 'moving_time_s' if 'moving_time_s' in subset.columns else 'Moving_s'
        elapsed_col = 'elapsed_time_s' if 'elapsed_time_s' in subset.columns else 'Elapsed_s'
        official_dist_col = 'official_distance_km' if 'official_distance_km' in subset.columns else 'Official_Dist_km'
        
        runs = []
        for _, row in subset.iterrows():
            # Prefer official distance for races, fall back to distance_km
            official_dist = row.get(official_dist_col)
            if pd.notna(official_dist) and official_dist > 0:
                dist_km = float(official_dist)
            elif dist_col == 'distance_km':
                dist_km = row[dist_col] if pd.notna(row.get(dist_col)) else 0
            else:
                dist_km = row[dist_col] / 1000 if pd.notna(row.get(dist_col)) else 0
            
            is_race = bool(row.get('race_flag', row.get('Race', 0)) == 1)
            
            # Use elapsed time for races (chip time), moving time for training
            if is_race:
                time_s = row.get(elapsed_col, row.get(moving_col, None))
            else:
                time_s = row.get(moving_col, row.get(elapsed_col, None))
            
            if pd.notna(time_s) and dist_km > 0:
                pace_sec = time_s / dist_km
                pace_min = int(pace_sec // 60)
                pace_s = int(pace_sec % 60)
                pace_str = f"{pace_min}:{pace_s:02d}"
            else:
                pace_str = "-"
            
            # Use RFL (individual run) as percentage, not RF_Trend (rolling)
            rfl_val = round(row['RFL'] * 100, 1) if pd.notna(row.get('RFL')) else None
            tss_val = int(round(row['TSS'])) if pd.notna(row.get('TSS')) else None
            hr_val = int(round(row[hr_col])) if pd.notna(row.get(hr_col)) else None
            npower_val = int(round(row[npower_col])) if pd.notna(row.get(npower_col)) else None
            
            # Get activity name - try both column name conventions
            name_val = row.get('activity_name', row.get('Activity_Name', ''))
            
            runs.append({
                'date': row['date'].strftime('%d %b %y'),
                'name': str(name_val) if name_val else '',
                'dist': round(dist_km, 1) if dist_km > 0 else None,
                'pace': pace_str,
                'npower': npower_val,
                'hr': hr_val,
                'tss': tss_val,
                'rfl': rfl_val,
                'rfl_gap': round(row['RFL_gap'] * 100, 1) if pd.notna(row.get('RFL_gap')) else None,
                'rfl_sim': round(row['RFL_sim'] * 100, 1) if pd.notna(row.get('RFL_sim')) else None,
                'race': is_race,
                # v51: trend mover — all three modes
                'delta': round(row['RFL_Trend_Delta'] * 100, 2) if pd.notna(row.get('RFL_Trend_Delta')) else None,
                'delta_gap': round(row['RFL_gap_Trend_Delta'] * 100, 2) if pd.notna(row.get('RFL_gap_Trend_Delta')) else None,
                'delta_sim': round(row['RFL_sim_Trend_Delta'] * 100, 2) if pd.notna(row.get('RFL_sim_Trend_Delta')) else None,
            })
        
        return runs
    
    # Recent runs (all)
    all_runs = extract_runs(df.tail(n).copy())
    
    # Recent races
    race_mask = df.get('race_flag', df.get('Race', pd.Series(dtype=float))).fillna(0) == 1
    races_df = df[race_mask].tail(n).copy()
    race_runs = extract_runs(races_df)
    
    return {'all': all_runs, 'races': race_runs}


def get_alert_data(df):
    """Get current alert status for dashboard banner, per mode.
    
    Returns dict with keys 'stryd', 'gap', 'sim', each a list of alert dicts.
    Also returns 'stryd' as the top-level list for backward compatibility.
    """
    ALERT_DEFS = {
        0: {'name': 'Training more, scoring worse', 'level': 'concern', 'icon': '⚠️'},
        1: {'name': 'Pre-race TSB concern', 'level': 'watch', 'icon': '⏳'},
        2: {'name': 'Deep fatigue', 'level': 'watch', 'icon': '👀'},
        3: {'name': 'Easy run outlier', 'level': 'watch', 'icon': '👀'},
    }
    
    result = {'stryd': [], 'gap': [], 'sim': []}
    
    if len(df) == 0:
        return result
    
    latest = df.iloc[-1]
    
    mode_configs = [
        ('stryd', '', 'RFL_Trend', 'Easy_RF_z'),
        ('gap', '_gap', 'RFL_gap_Trend', 'Easy_RF_z_gap'),
        ('sim', '_sim', 'RFL_sim_Trend', 'Easy_RF_z_sim'),
    ]
    
    for mode_name, suffix, rfl_col, ez_z_col in mode_configs:
        mask_col = f'Alert_Mask{suffix}'
        mask = int(latest.get(mask_col, 0)) if mask_col in df.columns else 0
        if not mask:
            continue
        
        alerts = []
        for bit in range(4):
            if not (mask & (1 << bit)):
                continue
            defn = ALERT_DEFS[bit]
            detail = ''
            if bit == 0:  # Alert 1: CTL/RFL divergence
                try:
                    cutoff = latest['date'] - pd.Timedelta(days=28)
                    earlier = df[df['date'] <= cutoff]
                    if len(earlier) > 0:
                        j = earlier.iloc[-1]
                        rfl_drop = (j[rfl_col] - latest[rfl_col]) * 100
                        ctl_rise = latest['CTL'] - j['CTL']
                        detail = f" (RFL -{rfl_drop:.1f}%, CTL +{ctl_rise:.0f})"
                except Exception:
                    pass
            elif bit == 1:  # Alert 1b: Pre-race TSB projection
                # Extract the projection detail from Alert_Text (StepB writes it there)
                text_col = f'Alert_Text{suffix}'
                raw_text = str(latest.get(text_col, ''))
                # Find the TSB projection substring
                import re as _re
                tsb_match = _re.search(r'TSB projected [^ ]+ for .+?\)', raw_text)
                if tsb_match:
                    detail = f" ({tsb_match.group()})"
                else:
                    detail = ""
            elif bit == 2 and pd.notna(latest.get('TSB')):  # Alert 2: TSB
                detail = f" (TSB {latest['TSB']:.0f})"
            elif bit == 3 and pd.notna(latest.get(ez_z_col)):  # Alert 3b: z-score
                detail = f" (z={latest[ez_z_col]:.1f})"
            alerts.append({
                'name': defn['name'],
                'level': defn['level'],
                'icon': defn['icon'],
                'detail': detail,
                'bit': bit,
            })
        result[mode_name] = alerts
    
    return result


def get_weight_chart_data(master_file, months=12):
    """v51.6: Get distance-weighted weekly average weight data for chart.
    
    Matches BFW approach: each run carries its smoothed daily weight from
    athlete_data, and the weekly average is weighted by run distance.
    """
    try:
        df = pd.read_excel(master_file, sheet_name='Master',
                           usecols=['date', 'weight_kg', 'distance_km'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['weight_kg'])
        # Need distance for weighting; default to 1 if missing
        df['distance_km'] = df['distance_km'].fillna(1.0).clip(lower=0.1)
    except Exception as e:
        print(f"  Warning: Could not load weight from Master: {e}")
        return [], []
    
    if len(df) == 0:
        return [], []
    
    try:
        today = datetime.now()
        cutoff = today - timedelta(days=months * 30)
        df = df[df['date'] > cutoff].copy()
        
        # Distance-weighted weekly averages (matches BFW)
        df['week'] = df['date'].dt.to_period('W').apply(lambda p: p.start_time)
        df['wt_x_dist'] = df['weight_kg'] * df['distance_km']
        weekly = df.groupby('week').agg(
            wt_sum=('wt_x_dist', 'sum'),
            dist_sum=('distance_km', 'sum')
        ).reset_index()
        weekly['weight_kg'] = weekly['wt_sum'] / weekly['dist_sum']
        weekly = weekly.sort_values('week')
        
        dates = weekly['week'].dt.strftime('%d %b %y').tolist()
        values = [round(v, 1) for v in weekly['weight_kg'].tolist()]
        
        return dates, values
    except Exception as e:
        print(f"  Warning: Could not process weight data: {e}")
        return [], []


def _calc_normalised_ag(raw_ag, moving_s, temp_adj, terrain_adj, surface_adj,
                        elevation_adj, avg_temp_c, dist_km=None):
    """Calculate conditions-normalised age grade.
    
    Removes the effect of temperature, terrain, surface, altitude, and
    distance-scaling bias from the raw age grade to answer: "what would
    this race have been worth on a flat road at 10°C, with fair comparison
    across distances?"
    
    The RF adjustment factors represent how much harder conditions made the run.
    Since AG ∝ 1/time and time ∝ 1/power (at steady state), and RF_adj = RF × adj,
    we get: normalised_AG = raw_AG × condition_adj × distance_adj.
    
    Temperature adjustment is duration-scaled (same formula as Power Score):
    short races get less heat penalty because core temperature hasn't risen as much.
    Cold penalty is NOT duration-scaled (vasoconstriction is immediate).
    
    Distance adjustment corrects for the systematic AG bias where longer races
    produce lower AG for equivalent-quality performances. This arises because:
    (a) recreational runners' pace scales with distance at a higher Riegel exponent
        (~1.08-1.10) than the WMA OC tables assume (~1.07);
    (b) WMA age factors penalise longer distances more for older runners.
    The correction is logarithmic in distance, anchored at 0% for 5K and
    +4% at marathon distance.
    """
    if raw_ag is None or not np.isfinite(raw_ag) or raw_ag <= 0:
        return raw_ag
    
    # Duration-scaled temperature adjustment (matches PS logic in StepB)
    _heat_ref_s = 5400.0    # 90 minutes — full heat effect reference
    _heat_max = 1.5         # Cap at 150% of base heat penalty (for ultramarathons)
    _cold_threshold = 5.0   # °C below which cold penalty applies
    _cold_factor = 0.003    # Per-degree cold penalty rate
    
    race_temp_adj = 1.0
    if pd.notna(temp_adj) and np.isfinite(temp_adj) and temp_adj > 0:
        if pd.notna(avg_temp_c) and np.isfinite(avg_temp_c) and avg_temp_c < _cold_threshold:
            # Cold: penalty is immediate, not duration-scaled
            cold_part = max(0, (_cold_threshold - avg_temp_c)) * _cold_factor
            race_temp_adj = 1.0 + cold_part
        else:
            # Heat+humidity: scale by race duration
            heat_part = max(0, temp_adj - 1.0)
            if pd.notna(moving_s) and moving_s > 0:
                heat_mult = min(_heat_max, moving_s / _heat_ref_s)
            else:
                heat_mult = 1.0
            race_temp_adj = 1.0 + heat_part * heat_mult
    
    # Terrain, surface, elevation — use as-is (already appropriate for whole run)
    t_adj = terrain_adj if pd.notna(terrain_adj) and np.isfinite(terrain_adj) and terrain_adj > 0 else 1.0
    s_adj = surface_adj if pd.notna(surface_adj) and np.isfinite(surface_adj) and surface_adj > 0 else 1.0
    e_adj = elevation_adj if pd.notna(elevation_adj) and np.isfinite(elevation_adj) and elevation_adj > 0 else 1.0
    
    # Distance-scaling bias correction
    # Logarithmic in distance, anchored at 5K=0%, marathon=+4%
    # Formula: dist_adj = 1 + 0.04 × ln(dist/5.0) / ln(42.195/5.0)
    # This gives: 5K=1.000, 10K=+1.3%, HM=+2.7%, Marathon=+4.0%
    _dist_ref = 5.0           # Reference distance (no correction)
    _dist_max_boost = 0.04    # +4% correction at marathon distance
    _dist_log_range = math.log(42.195 / _dist_ref)  # ln(42.195/5.0) ≈ 2.134
    
    dist_adj = 1.0
    if dist_km is not None and pd.notna(dist_km) and dist_km > _dist_ref:
        dist_adj = 1.0 + _dist_max_boost * math.log(dist_km / _dist_ref) / _dist_log_range
    
    condition_adj = race_temp_adj * t_adj * s_adj * e_adj
    return round(raw_ag * condition_adj * dist_adj, 2)


def get_top_races(df, n=10):
    """Get top race performances ranked by conditions-normalised Age Grade.
    
    Normalised AG removes the effect of temperature, hills, surface, and
    altitude to answer: "what would this race have been worth on a flat road
    at 10°C?" This means a hilly, hot race that produced 73% AG raw could
    rank higher than a flat, cool race at 75% AG if conditions explain the gap.
    
    Returns dict with keys: '1y', '2y', '3y', '5y', 'all'
    Each value is a list of up to n race dicts sorted by normalised AG descending.
    """
    # Filter to races only
    race_col = 'race_flag' if 'race_flag' in df.columns else 'Race'
    races = df[df[race_col] == 1].copy()
    
    if len(races) == 0:
        return {period: [] for period in ['1y', '2y', '3y', '5y', 'all']}
    
    # Must have age grade to rank
    if 'age_grade_pct' not in races.columns:
        return {period: [] for period in ['1y', '2y', '3y', '5y', 'all']}
    
    # Filter: must have valid AG and minimum quality gate (55% AG)
    # This catches misclassified training runs (e.g. "Tempo incl Parkrun" at 48% AG)
    races = races[races['age_grade_pct'].notna() & (races['age_grade_pct'] >= 55)].copy()
    
    # Filter: exclude races where auto-pause makes moving_time unreliable
    # If elapsed/moving > 2.0, the watch removed too much time (common on track
    # sessions where standing between reps triggers auto-pause). The AG calculation
    # uses moving_time_s, so an artificially short moving_time inflates AG.
    if 'elapsed_time_s' in races.columns and 'moving_time_s' in races.columns:
        _elapsed = pd.to_numeric(races['elapsed_time_s'], errors='coerce')
        _moving = pd.to_numeric(races['moving_time_s'], errors='coerce')
        _ratio = _elapsed / _moving.replace(0, np.nan)
        _suspect = _ratio > 2.0
        if _suspect.any():
            _n_removed = _suspect.sum()
            print(f"  Top races: excluded {_n_removed} race(s) with elapsed/moving ratio > 2.0 (auto-pause suspect)")
        races = races[~_suspect].copy()
    
    if len(races) == 0:
        return {period: [] for period in ['1y', '2y', '3y', '5y', 'all']}
    
    # Resolve official distance column name for normalised AG
    _off_dist_col = 'official_distance_km' if 'official_distance_km' in races.columns else 'Official_Dist_km'
    
    # Compute normalised AG for each race
    races['normalised_ag'] = races.apply(lambda row: _calc_normalised_ag(
        row.get('age_grade_pct'),
        row.get('moving_time_s', row.get('elapsed_time_s')),
        row.get('Temp_Adj'),
        row.get('Terrain_Adj'),
        row.get('surface_adj'),
        row.get('Elevation_Adj'),
        row.get('avg_temp_c'),
        row.get(_off_dist_col, row.get('distance_km')),
    ), axis=1)
    
    sort_col = 'normalised_ag'
    
    now = datetime.now()
    periods = {
        '1y': timedelta(days=365),
        '2y': timedelta(days=730),
        '3y': timedelta(days=1095),
        '5y': timedelta(days=1825),
        'all': None,
    }
    
    # Column name detection
    dist_col = 'distance_km' if 'distance_km' in races.columns else 'Distance_m'
    moving_col = 'moving_time_s' if 'moving_time_s' in races.columns else 'Moving_s'
    elapsed_col = 'elapsed_time_s' if 'elapsed_time_s' in races.columns else 'Elapsed_s'
    official_dist_col = 'official_distance_km' if 'official_distance_km' in races.columns else 'Official_Dist_km'
    
    def format_race(row):
        # Prefer official distance for races
        official_dist = row.get(official_dist_col)
        if pd.notna(official_dist) and official_dist > 0:
            dist_km = float(official_dist)
        elif dist_col == 'distance_km':
            dist_km = row[dist_col] if pd.notna(row.get(dist_col)) else 0
        else:
            dist_km = row[dist_col] / 1000 if pd.notna(row.get(dist_col)) else 0
        
        moving_s = row.get(moving_col, row.get(elapsed_col, None))
        
        # Format time as H:MM:SS or M:SS
        if pd.notna(moving_s) and moving_s > 0:
            hours = int(moving_s // 3600)
            mins = int((moving_s % 3600) // 60)
            secs = int(moving_s % 60)
            if hours > 0:
                time_str = f"{hours}:{mins:02d}:{secs:02d}"
            else:
                time_str = f"{mins}:{secs:02d}"
        else:
            time_str = "-"
        
        # Get activity name
        name_val = row.get('activity_name', row.get('Activity_Name', ''))
        
        # Age grade percentage (normalised)
        nag_val = round(row['normalised_ag'], 1) if pd.notna(row.get('normalised_ag')) else None
        
        # HR and TSS
        hr_val = int(round(row['avg_hr'])) if pd.notna(row.get('avg_hr')) else None
        tss_val = int(round(row['TSS'])) if pd.notna(row.get('TSS')) else None
        
        return {
            'date': row['date'].strftime('%d %b %y'),
            'name': str(name_val) if name_val else '',
            'dist': round(dist_km, 1) if dist_km > 0 else None,
            'dist_group': 'long' if dist_km > 5.5 else 'short',
            'time': time_str,
            'hr': hr_val,
            'tss': tss_val,
            'nag': nag_val,
        }
    
    result = {}
    for period_name, delta in periods.items():
        if delta is None:
            period_races = races
        else:
            cutoff = now - delta
            period_races = races[races['date'] >= cutoff]
        
        # Return all races sorted by normalised AG descending
        # JS will apply distance filter and take top n
        top = period_races.nlargest(len(period_races), sort_col)
        result[period_name] = [format_race(row) for _, row in top.iterrows()]
    
    return result


# ============================================================================
# v51: RACE PREDICTION TREND DATA
# ============================================================================
def get_prediction_trend_data(df):
    """Get predicted vs actual race times — prediction shown only at race dates.
    Returns dict keyed by distance."""
    result = {}
    
    distances = {
        '5k': {'pred_col': 'pred_5k_s', 'official_km': 5.0},
        '10k': {'pred_col': 'pred_10k_s', 'official_km': 10.0},
        'hm': {'pred_col': 'pred_hm_s', 'official_km': 21.097},
        'marathon': {'pred_col': 'pred_marathon_s', 'official_km': 42.195},
    }
    
    for dist_key, info in distances.items():
        pred_col = info['pred_col']
        dist_km = info['official_km']
        tolerance = dist_km * 0.05
        empty = {'dates': [], 'dates_iso': [], 'predicted': [], 'actual': [], 'is_parkrun': [], 'names': [], 'temps': [], 'surfaces': [], 'temp_adjs': [], 'surface_adjs': []}
        
        if pred_col not in df.columns:
            result[dist_key] = empty
            continue
        
        # Find races at this distance
        race_col = 'race_flag' if 'race_flag' in df.columns else 'Race'
        races = df[df[race_col] == 1].copy()
        
        if 'official_distance_km' in races.columns:
            dist_match = races[
                ((races['official_distance_km'] - dist_km).abs() < tolerance) |
                ((races['distance_km'] - dist_km).abs() < tolerance)
            ]
        else:
            dist_match = races[(races['distance_km'] - dist_km).abs() < tolerance]
        
        time_col = 'elapsed_time_s' if 'elapsed_time_s' in dist_match.columns else 'moving_time_s'
        dist_match = dist_match[dist_match[time_col].notna() & (dist_match[time_col] > 0)].copy()
        
        # Exclude races where GPS distance far exceeds official distance
        # (activity includes warmup/cooldown, so elapsed time is meaningless for the race)
        if 'distance_km' in dist_match.columns and 'official_distance_km' in dist_match.columns:
            _before = len(dist_match)
            dist_match = dist_match[
                (dist_match['distance_km'].isna()) |
                (dist_match['official_distance_km'].isna()) |
                (dist_match['distance_km'] <= dist_match['official_distance_km'] * 1.3)
            ]
            _dropped = _before - len(dist_match)
            if _dropped > 0:
                print(f"  {dist_key}: excluded {_dropped} races with GPS distance >130% of official (warmup/cooldown included in activity)")
        
        if len(dist_match) == 0:
            result[dist_key] = empty
            continue
        
        dates = dist_match['date'].dt.strftime('%d %b %y').tolist()
        dates_iso = dist_match['date'].dt.strftime('%Y-%m-%d').tolist()
        actual = [round(v, 0) for v in dist_match[time_col].tolist()]
        
        # Get predicted time, name, temp at each race date
        predicted = []
        race_names = []
        race_temps = []
        race_surfaces = []
        for _, race_row in dist_match.iterrows():
            pred_val = race_row.get(pred_col)
            if pd.notna(pred_val) and pred_val > 0:
                predicted.append(round(pred_val, 0))
            else:
                race_dt = race_row['date']
                nearby = df[(df['date'] - race_dt).abs() <= pd.Timedelta(days=7)]
                nearby_pred = nearby[pred_col].dropna()
                if len(nearby_pred) > 0:
                    predicted.append(round(nearby_pred.iloc[-1], 0))
                else:
                    predicted.append(None)
            
            name = race_row.get('activity_name', '')
            race_names.append(str(name)[:50] if pd.notna(name) else '')
            
            temp = race_row.get('avg_temp_c')
            race_temps.append(round(temp, 0) if pd.notna(temp) else None)
            
            surface = str(race_row.get('surface', '')).strip()
            race_surfaces.append(surface if surface and surface != 'nan' else None)
        
        # Condition adjustment factors for "adjust for conditions" toggle
        race_temp_adjs = []
        race_surface_adjs = []
        for _, race_row in dist_match.iterrows():
            ta = race_row.get('Temp_Adj')
            race_temp_adjs.append(round(float(ta), 4) if pd.notna(ta) else 1.0)
            sa = race_row.get('surface_adj')
            race_surface_adjs.append(round(float(sa), 4) if pd.notna(sa) else 1.0)
        
        is_parkrun = []
        for _, race_row in dist_match.iterrows():
            is_pr = bool(race_row.get('parkrun', 0)) or bool(race_row.get('hf_parkrun', 0))
            is_parkrun.append(1 if is_pr else 0)
        
        result[dist_key] = {
            'dates': dates,
            'dates_iso': dates_iso,
            'predicted': predicted,
            'actual': actual,
            'is_parkrun': is_parkrun,
            'names': race_names,
            'temps': race_temps,
            'surfaces': race_surfaces,
            'temp_adjs': race_temp_adjs,
            'surface_adjs': race_surface_adjs,
        }
        
        # Phase 2: GAP and Sim predicted times at each race date
        for mode in ('gap', 'sim'):
            mode_col = f'{pred_col}_{mode}'
            mode_preds = []
            for _, race_row in dist_match.iterrows():
                pv = race_row.get(mode_col)
                if pd.notna(pv) and float(pv) > 0:
                    mode_preds.append(round(float(pv), 0))
                else:
                    mode_preds.append(None)
            result[dist_key][f'predicted_{mode}'] = mode_preds
        
        # Add full prediction trend line (weekly samples, all rows with predictions)
        # This shows fitness trajectory even between races
        pred_series = df[[pred_col, 'date']].dropna(subset=[pred_col])
        pred_series = pred_series[pred_series[pred_col] > 0].copy()
        if len(pred_series) > 0:
            # Weekly mean, then 4-week rolling smooth to avoid spikes from single bad runs
            pred_series = pred_series.set_index('date').resample('W').mean().dropna(subset=[pred_col]).reset_index()
            pred_series[pred_col] = pred_series[pred_col].rolling(4, min_periods=1, center=True).mean()
            result[dist_key]['trend_dates_iso'] = pred_series['date'].dt.strftime('%Y-%m-%d').tolist()
            result[dist_key]['trend_values'] = [round(v, 0) for v in pred_series[pred_col].tolist()]
            # GAP/Sim trend lines
            for mode in ('gap', 'sim'):
                mode_col = f'{pred_col}_{mode}'
                if mode_col in df.columns:
                    mode_series = df[[mode_col, 'date']].dropna(subset=[mode_col])
                    mode_series = mode_series[mode_series[mode_col] > 0].copy()
                    if len(mode_series) > 0:
                        mode_series = mode_series.set_index('date').resample('W').mean().dropna(subset=[mode_col]).reset_index()
                        mode_series[mode_col] = mode_series[mode_col].rolling(4, min_periods=1, center=True).mean()
                        result[dist_key][f'trend_dates_iso_{mode}'] = mode_series['date'].dt.strftime('%Y-%m-%d').tolist()
                        result[dist_key][f'trend_values_{mode}'] = [round(v, 0) for v in mode_series[mode_col].tolist()]
    
    return result


# ============================================================================
# v51: AGE GRADE TREND DATA
# ============================================================================
def get_age_grade_data(df):
    """Get age grade trend data for chart.
    Returns dict with dates, ag_pct, distance labels, colours, and rolling average."""
    race_col = 'race_flag' if 'race_flag' in df.columns else 'Race'
    races = df[(df[race_col] == 1) & (df['age_grade_pct'].notna())].copy() if 'age_grade_pct' in df.columns else pd.DataFrame()
    
    # Exclude races where GPS distance far exceeds official distance
    if len(races) > 0 and 'distance_km' in races.columns and 'official_distance_km' in races.columns:
        races = races[
            (races['distance_km'].isna()) |
            (races['official_distance_km'].isna()) |
            (races['distance_km'] <= races['official_distance_km'] * 1.3)
        ]
    
    if len(races) == 0:
        return {'dates': [], 'dates_iso': [], 'values': [], 'dist_labels': [], 'dist_codes': [],
                'dist_sizes': [], 'is_parkrun': [], 'rolling_avg': []}
    
    # Categorise distances  
    def categorise_dist(row):
        d = row.get('official_distance_km')
        if pd.isna(d) or d <= 0:
            d = row.get('distance_km', 0)
        if d < 4:
            return '3k-'
        elif d < 7:
            return '5k'
        elif d < 15:
            return '10k'
        elif d < 25:
            return 'HM'
        elif d < 35:
            return '30k'
        else:
            return 'Marathon'
    
    races = races.sort_values('date').copy()
    races['dist_cat'] = races.apply(categorise_dist, axis=1)
    races['is_parkrun'] = ((races.get('parkrun', 0) == 1) | (races.get('hf_parkrun', 0) == 1)).astype(int)
    
    dates = races['date'].dt.strftime('%d %b %y').tolist()
    values = [round(v, 1) for v in races['age_grade_pct'].tolist()]
    dist_labels = races['dist_cat'].tolist()
    is_parkrun = races['is_parkrun'].tolist()
    
    # Colour codes and sizes for each distance category
    colour_map = {
        '3k-': '#8b5cf6',      # purple
        '5k': '#818cf8',       # blue
        '10k': '#16a34a',      # green
        'HM': '#ea580c',       # orange
        '30k': '#dc2626',      # red
        'Marathon': '#991b1b', # dark red
    }
    size_map = {
        '3k-': 2.5,
        '5k': 3,
        '10k': 4,
        'HM': 5,
        '30k': 6,
        'Marathon': 7,
    }
    dist_codes = []
    for i, d in enumerate(dist_labels):
        base = colour_map.get(d, '#6b7280')
        if is_parkrun[i]:
            # Lighter/muted version for parkruns
            parkrun_map = {
                '5k': '#93bbfd',    # muted blue
                '3k-': '#c4b5fd',   # light purple
            }
            dist_codes.append(parkrun_map.get(d, base))
        else:
            dist_codes.append(base)
    dist_sizes = [size_map.get(d, 3) for d in dist_labels]
    
    # Rolling median (20-race window, centered) — robust to outliers
    rolling = races['age_grade_pct'].rolling(window=20, min_periods=5, center=True).median()
    rolling_avg = [round(v, 1) if pd.notna(v) else None for v in rolling.tolist()]
    
    # Also return dates as ISO strings for proper time axis
    dates_iso = races['date'].dt.strftime('%Y-%m-%d').tolist()
    
    return {'dates': dates, 'dates_iso': dates_iso, 'values': values, 'dist_labels': dist_labels, 
            'dist_codes': dist_codes, 'dist_sizes': dist_sizes, 'is_parkrun': is_parkrun, 'rolling_avg': rolling_avg}


# ============================================================================
# v51.7: ZONE & RACE CONFIGURATION
# ============================================================================
from config import PEAK_CP_WATTS as _cfg_cp, ATHLETE_MASS_KG as _cfg_mass
from config import ATHLETE_LTHR as _cfg_lthr, ATHLETE_MAX_HR as _cfg_maxhr
from config import PLANNED_RACES as _cfg_races
from config import POWER_MODE as _cfg_power_mode
from config import ATHLETE_NAME as _cfg_name
PEAK_CP_WATTS_DASH = _cfg_cp

# Load raw zone overrides from athlete.yml (not in dataclass — optional section)
_zone_overrides = {}
try:
    import yaml as _yaml
    from config import _ATHLETE_CONFIG
    _yml_path = getattr(_ATHLETE_CONFIG, '_yaml_path', None)
    if _yml_path is None:
        import os as _os
        _yml_path = _os.environ.get("ATHLETE_CONFIG_PATH", "athlete.yml")
    with open(_yml_path, 'r') as _f:
        _raw_yml = _yaml.safe_load(_f)
    _zone_overrides = _raw_yml.get('zones', {})
except Exception:
    pass

# --- Module-level zone boundaries (visible to get_zone_data AND _generate_zone_html) ---
if _zone_overrides and 'hr_zones' in _zone_overrides:
    _zb = _zone_overrides['hr_zones']
    _z12_hr, _z23_hr, _z34_hr, _z45_hr = int(_zb[0]), int(_zb[1]), int(_zb[2]), int(_zb[3])
else:
    _z12_hr = round(_cfg_lthr * 0.81)
    _z23_hr = round(_cfg_lthr * 0.90)
    _z34_hr = round(_cfg_lthr * 0.955)
    _z45_hr = _cfg_lthr

if _zone_overrides and 'race_hr_zones' in _zone_overrides:
    _rzb = _zone_overrides['race_hr_zones']
    _rhr_mara, _rhr_hm, _rhr_10k, _rhr_5k, _rhr_sub5k = int(_rzb[0]), int(_rzb[1]), int(_rzb[2]), int(_rzb[3]), int(_rzb[4])
else:
    _hr_range = _cfg_maxhr - _cfg_lthr
    _rhr_mara = round(_cfg_lthr * 0.92)
    _rhr_hm = round(_cfg_lthr * 0.96)
    _rhr_10k = _cfg_lthr
    _rhr_5k = round(_cfg_lthr + _hr_range * 0.35)
    _rhr_sub5k = round(_cfg_lthr + _hr_range * 0.70)
ATHLETE_MASS_KG_DASH = _cfg_mass
LTHR_DASH = _cfg_lthr
MAX_HR_DASH = _cfg_maxhr

RACE_POWER_FACTORS_DASH = {
    'Sub-5K': 1.07, '5K': 1.05, '10K': 1.00, 'HM': 0.95, 'Mara': 0.90,
}
RACE_DISTANCES_KM_DASH = {
    'Sub-5K': 3.0, '5K': 5.0, '10K': 10.0, 'HM': 21.097, 'Mara': 42.195,
}
def _distance_km_to_key(km):
    """Map distance_km to race category key."""
    if km <= 3.5: return 'Sub-5K'
    if km <= 7.5: return '5K'
    if km <= 15.0: return '10K'
    if km <= 30.0: return 'HM'
    return 'Mara'

if _cfg_races is not None:
    PLANNED_RACES_DASH = [
        {'name': r['name'], 'date': r['date'], 'distance_key': _distance_km_to_key(r['distance_km']),
         'distance_km': r['distance_km'], 'priority': r.get('priority', 'B'), 'surface': r.get('surface', 'road')}
        for r in _cfg_races
    ]
else:
    PLANNED_RACES_DASH = [
        {'name': '5K London', 'date': '2026-02-27', 'distance_key': '5K', 'distance_km': 5.0, 'priority': 'A', 'surface': 'road'},
        {'name': 'HM Stockholm', 'date': '2026-04-25', 'distance_key': 'HM', 'distance_km': 21.097, 'priority': 'A', 'surface': 'road'},
    ]


def get_zone_data(df):
    """Extract run data with per-second time-in-zone from NPZ cache."""
    import glob, os
    
    recent = df.tail(100).copy()
    npower_col = 'npower_w' if 'npower_w' in recent.columns else 'nPower'
    hr_col = 'avg_hr' if 'avg_hr' in recent.columns else 'Avg_HR'
    moving_col = 'moving_time_s' if 'moving_time_s' in recent.columns else 'Moving_s'
    dist_col = 'distance_km' if 'distance_km' in recent.columns else 'Distance_m'
    
    # Use mode-appropriate RFL column
    _rfl_col_name = {'gap': 'RFL_gap_Trend', 'sim': 'RFL_sim_Trend'}.get(_cfg_power_mode, 'RFL_Trend')
    if _rfl_col_name not in df.columns or df[_rfl_col_name].dropna().empty:
        _rfl_col_name = 'RFL_Trend'  # fallback
    current_rfl = float(df.iloc[-1].get(_rfl_col_name, 0.90)) if len(df) > 0 else 0.90
    if pd.isna(current_rfl):
        current_rfl = 0.90
    current_cp = round(PEAK_CP_WATTS_DASH * current_rfl)
    
    re_col = 'RE_avg'
    if re_col in df.columns:
        recent_re = df.tail(100)[re_col].dropna()
        re_p90 = round(float(recent_re.quantile(0.90)), 4) if len(recent_re) > 0 else 0.92
    else:
        re_p90 = 0.92
    
    # Zone boundaries — 5-zone model, RF×HR anchored
    # Power zones derived from RF ratio (CP/LTHR) × HR boundary
    # HR zone boundaries are module-level (_z12_hr etc) from athlete.yml or formula
    RF_THR = current_cp / LTHR_DASH  # RF at lactate threshold
    hr_bounds = [0, _z12_hr, _z23_hr, _z34_hr, _z45_hr, 9999]
    hr_znames = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
    # Power zones (5): RF×HR anchored — boundaries shift with fitness
    pw_z12 = round(RF_THR * _z12_hr)
    pw_z23 = round(RF_THR * _z23_hr)
    pw_z34 = round(RF_THR * _z34_hr)
    pw_z45 = round(RF_THR * _z45_hr)
    pw_bounds = [0, pw_z12, pw_z23, pw_z34, pw_z45, 9999]
    pw_znames = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
    # Race power zones: contiguous midpoint bands
    m_pw = current_cp * 0.90
    h_pw = current_cp * 0.95
    t_pw = current_cp * 1.00
    f_pw = current_cp * 1.05
    race_pw_bounds = [0, round(m_pw*0.93), round((m_pw+h_pw)/2), round((h_pw+t_pw)/2), round((t_pw+f_pw)/2), round(f_pw*1.05), 9999]
    race_pw_names = ['Other', 'Mara', 'HM', '10K', '5K', 'Sub-5K']
    # Race HR zones: from module-level (_rhr_mara etc) — athlete.yml or formula
    race_hr_bounds = [0, _rhr_mara, _rhr_hm, _rhr_10k, _rhr_5k, _rhr_sub5k, 9999]
    race_hr_names = ['Other', 'Mara', 'HM', '10K', '5K', 'Sub-5K']

    # Race GAP pace zones (sec/km) — built from GAP predictions
    # These mirror the JS makeRacePaceZones() logic
    # Note: pace is inverted — faster pace = lower number = harder effort
    gap_target_paces_local = {}
    latest_row = df.iloc[-1] if len(df) > 0 else None
    if latest_row is not None:
        for dk, dkm in [('5k', 5.0), ('10k', 10.0), ('hm', 21.097), ('marathon', 42.195)]:
            col = f'pred_{dk}_s_gap'
            if col in df.columns and pd.notna(latest_row.get(col)):
                gap_target_paces_local[dk] = round(latest_row[col] / dkm)
    p5 = gap_target_paces_local.get('5k', 240)
    p10 = gap_target_paces_local.get('10k', 252)
    ph = gap_target_paces_local.get('hm', 265)
    pm = gap_target_paces_local.get('marathon', 282)
    # Bounds: fastest (lowest sec/km) to slowest — same midpoint logic as JS
    race_pace_bounds_fast_to_slow = [
        0,                          # Sub-5K lower (fastest possible)
        round(p5 * 0.97),           # Sub-5K / 5K boundary
        round((p5 + p10) / 2),      # 5K / 10K boundary
        round((p10 + ph) / 2),      # 10K / HM boundary
        round((ph + pm) / 2),       # HM / Mara boundary
        round(pm * 1.07),           # Mara / Other boundary
        9999                        # Other upper
    ]
    race_pace_names = ['Sub-5K', '5K', '10K', 'HM', 'Mara', 'Other']

    def _time_in_zones(values, bounds, names, rolling_s=30):
        """Compute minutes in each zone using rolling average."""
        import numpy as np
        vals = np.array(values, dtype=float)
        valid = ~np.isnan(vals) & (vals > 0)
        if valid.sum() < 10:
            return {n: 0.0 for n in names}
        # Rolling average
        kernel = min(rolling_s, valid.sum())
        if kernel > 1:
            rolled = pd.Series(vals).rolling(kernel, min_periods=1).mean().values
        else:
            rolled = vals
        result = {n: 0.0 for n in names}
        for v in rolled:
            if np.isnan(v) or v <= 0:
                continue
            for i in range(len(bounds)-2, -1, -1):
                if v >= bounds[i]:
                    result[names[i]] += 1.0
                    break
        # Convert seconds to minutes
        return {n: round(v / 60.0, 1) for n, v in result.items()}

    def _time_in_pace_zones(pace_skm_values, bounds_fast_to_slow, names, rolling_s=30):
        """Compute minutes in each pace zone using rolling average.
        
        Pace zones are inverted vs power/HR: lower value = faster = harder effort.
        bounds_fast_to_slow: [0, sub5k_bound, 5k_10k_bound, ..., 9999]
        names: ['Sub-5K', '5K', '10K', 'HM', 'Mara', 'Other']
        A pace value falls in zone i if bounds[i] <= pace < bounds[i+1].
        """
        import numpy as np
        vals = np.array(pace_skm_values, dtype=float)
        valid = ~np.isnan(vals) & (vals > 0) & (vals < 1200)  # sane pace range
        if valid.sum() < 10:
            return {n: 0.0 for n in names}
        # Rolling average
        kernel = min(rolling_s, valid.sum())
        if kernel > 1:
            rolled = pd.Series(vals).rolling(kernel, min_periods=1).mean().values
        else:
            rolled = vals
        result = {n: 0.0 for n in names}
        for v in rolled:
            if np.isnan(v) or v <= 0:
                continue
            # Assign to zone: find which bracket this pace falls in
            assigned = False
            for i in range(len(bounds_fast_to_slow) - 1):
                if bounds_fast_to_slow[i] <= v < bounds_fast_to_slow[i + 1]:
                    result[names[i]] += 1.0
                    assigned = True
                    break
            if not assigned:
                result[names[-1]] += 1.0  # slowest zone
        # Convert seconds to minutes
        return {n: round(v / 60.0, 1) for n, v in result.items()}
    
    # Find NPZ cache directory
    npz_dir = None
    _master_dir = os.path.dirname(os.path.abspath(MASTER_FILE))
    for candidate in ['persec_cache_FULL', '../persec_cache_FULL', 'persec_cache',
                       os.path.join(_master_dir, 'persec_cache_FULL'),
                       os.path.join(_master_dir, 'persec_cache'),
                       os.path.join(_master_dir, '..', 'persec_cache_FULL'),
                       os.path.join(_master_dir, '..', 'persec_cache')]:
        if os.path.isdir(candidate):
            npz_dir = candidate
            break
    
    # Build index of available NPZ files by date prefix
    npz_index = {}
    npz_by_date = {}  # date-only prefix -> list of paths (for fallback matching)
    if npz_dir:
        for fp in glob.glob(os.path.join(npz_dir, '*.npz')):
            base = os.path.basename(fp).replace('.npz', '')
            npz_index[base] = fp
            date_prefix = base[:10]  # '2026-02-17'
            if date_prefix not in npz_by_date:
                npz_by_date[date_prefix] = []
            npz_by_date[date_prefix].append(fp)
        print(f"  NPZ cache: {npz_dir} ({len(npz_index)} files)")
    
    runs = []
    npz_hits = 0
    for _, row in recent.iterrows():
        npower = row.get(npower_col)
        hr = row.get(hr_col)
        moving_s = row.get(moving_col)
        dist = row.get(dist_col, 0)
        name = row.get('activity_name', '')
        is_race = bool(row.get('race_flag', 0) == 1)
        if pd.isna(npower) or pd.isna(hr) or pd.isna(moving_s):
            continue
        duration_min = moving_s / 60.0
        if duration_min < 5:
            continue
        
        run_date = row['date']
        run_id = run_date.strftime('%Y-%m-%d_%H-%M-%S')
        
        # NPZ file lookup: try run_id column first (matches FIT filename),
        # then date-based ID, then file-based ID, then date-prefix fallback
        master_run_id = str(row.get('run_id', '')).strip() if pd.notna(row.get('run_id')) else ''
        npz_key = master_run_id.replace('&', '_').replace('?', '_').replace('=', '_') if master_run_id else ''
        
        # Fallback: extract activity ID from file column (e.g. '18502680351.fit' -> '18502680351')
        file_id = ''
        if not npz_key and pd.notna(row.get('file')):
            file_id = str(row['file']).replace('.fit', '').replace('.FIT', '').strip()
        
        run_entry = {
            'date': run_date.strftime('%Y-%m-%d'),
            'name': str(name)[:60] if pd.notna(name) else '',
            'avg_hr': round(float(hr), 1),
            'npower': round(float(npower)),
            'duration_min': round(duration_min, 1),
            'distance_km': round(float(dist), 1) if pd.notna(dist) else 0,
            'avg_pace_skm': round(moving_s / float(dist)) if pd.notna(dist) and float(dist) > 0 else 0,
            'race': is_race,
        }
        
        # Try to load NPZ for per-second time-in-zone
        npz_path = npz_index.get(npz_key) if npz_key else None
        if not npz_path and file_id:
            npz_path = npz_index.get(file_id)
        if not npz_path:
            npz_path = npz_index.get(run_id)
        if not npz_path:
            # Fallback: match by date prefix (for masters without time-of-day)
            date_prefix = run_date.strftime('%Y-%m-%d')
            candidates = npz_by_date.get(date_prefix, [])
            if len(candidates) == 1:
                npz_path = candidates[0]
        if npz_path:
            try:
                npz_data = np.load(npz_path, allow_pickle=True)
                pw_arr = npz_data['power_w'].copy()
                hr_arr = npz_data['hr_bpm']
                spd_arr = npz_data['speed_mps']
                grd_arr = npz_data['grade']
                
                # COT power cleaning now happens in rebuild (v51.8) — NPZ power_w
                # already has implausible sections set to NaN.  For zone calcs,
                # treat NaN power as 0 so _time_in_zones counts them correctly.
                pw_arr = np.where(np.isnan(pw_arr), 0.0, pw_arr)
                
                run_entry['hz'] = _time_in_zones(hr_arr, hr_bounds, hr_znames)
                run_entry['pz'] = _time_in_zones(pw_arr, pw_bounds, pw_znames)
                run_entry['rpz'] = _time_in_zones(pw_arr, race_pw_bounds, race_pw_names)
                run_entry['rhz'] = _time_in_zones(hr_arr, race_hr_bounds, race_hr_names)
                
                # GAP pace zones: compute per-second GAP pace from speed + grade
                try:
                    from gap_power import compute_gap_for_run
                    grd_safe_gap = np.where(np.isnan(grd_arr), 0, grd_arr)
                    gap_speed = compute_gap_for_run(spd_arr, grd_safe_gap)
                    # Convert to pace (sec/km); zero/stopped seconds → NaN
                    gap_pace_skm = np.where(gap_speed > 0.5, 1000.0 / gap_speed, np.nan)
                    run_entry['rpz_gap'] = _time_in_pace_zones(
                        gap_pace_skm, race_pace_bounds_fast_to_slow, race_pace_names)
                except Exception:
                    pass  # GAP zones not available — JS will use heuristic fallback
                
                npz_hits += 1
            except Exception as e:
                pass  # Fall back to no zone data
        
        runs.append(run_entry)
    
    print(f"  Zone data: {len(runs)} runs, {npz_hits} with NPZ, CP={current_cp}W (RFL={current_rfl:.4f}), RE_p90={re_p90}")
    
    # RFL projection slope for race readiness cards
    _rfl_proj_per_day = 0.0
    _rfl_proj_col = {'gap': 'RFL_gap_Trend', 'sim': 'RFL_sim_Trend'}.get(_cfg_power_mode, 'RFL_Trend')
    if _rfl_proj_col not in df.columns or df[_rfl_proj_col].dropna().empty:
        _rfl_proj_col = 'RFL_Trend'
    try:
        _rfl_col = df[_rfl_proj_col].dropna()
        if len(_rfl_col) >= 10:
            _recent_28 = _rfl_col.tail(28)
            if len(_recent_28) >= 5:
                _x = np.arange(len(_recent_28), dtype=float)
                _y = _recent_28.values.astype(float)
                _slope, _ = np.polyfit(_x, _y, 1)
                _days_span = (df['date'].iloc[-1] - df['date'].iloc[-len(_recent_28)]).days
                if _days_span > 0:
                    _rfl_proj_per_day = _slope * len(_recent_28) / _days_span
    except Exception:
        pass
    
    return {'runs': runs, 'current_cp': current_cp, 'current_rfl': round(current_rfl, 4), 're_p90': re_p90, 'gap_target_paces': gap_target_paces_local, 'rfl_proj_per_day': _rfl_proj_per_day}


# ============================================================================
# v51.7: ZONE HTML GENERATOR
# ============================================================================
def _generate_zone_html(zone_data, stats=None):
    """Generate the Training Zones, Race Readiness, and Weekly Zone Volume HTML."""
    if zone_data is None:
        return '<!-- zones: no data -->'
    
    cp = zone_data['current_cp']
    re_p90 = zone_data['re_p90']
    gap_paces = zone_data.get('gap_target_paces', {})
    # Default paces if not available (sec/km)
    pace_5k = gap_paces.get('5k', 240)
    pace_10k = gap_paces.get('10k', 252)
    pace_hm = gap_paces.get('hm', 265)
    pace_mara = gap_paces.get('marathon', 282)
    
    # Combined zone table rows (single table, 5 columns)
    combined_rows = ''
    zone_names = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
    zone_labels = ['Easy', 'Aerobic', 'Tempo', 'Threshold', 'Max']
    zone_colors = ['#3b82f6', '#22c55e', '#eab308', '#f97316', '#ef4444']
    # Power zones: RF×HR anchored (using module-level zone boundaries)
    rf_thr = cp / LTHR_DASH
    pw_z12 = round(rf_thr * _z12_hr)
    pw_z23 = round(rf_thr * _z23_hr)
    pw_z34 = round(rf_thr * _z34_hr)
    pw_z45 = cp  # RF×LTHR = CP by definition
    hr_ranges =   [f'<{_z12_hr}', f'{_z12_hr}–{_z23_hr}', f'{_z23_hr}–{_z34_hr}', f'{_z34_hr}–{_z45_hr}', f'>{_z45_hr}']
    pw_strs =     [f'<{pw_z12}', f'{pw_z12}–{pw_z23}', f'{pw_z23}–{pw_z34}', f'{pw_z34}–{pw_z45}', f'>{pw_z45}']
    pct_strs =    [f'<{round(pw_z12/cp*100)}', f'{round(pw_z12/cp*100)}–{round(pw_z23/cp*100)}', f'{round(pw_z23/cp*100)}–{round(pw_z34/cp*100)}', f'{round(pw_z34/cp*100)}–{round(pw_z45/cp*100)}', f'>{round(pw_z45/cp*100)}']
    effort_hints = ['', 'easy–Mara', 'Mara–10K', '10K–5K', '>5K']
    combined_rows = ''
    for i in range(5):
        combined_rows += f'<tr><td><span class="zd" style="background:{zone_colors[i]}"></span><strong>{zone_names[i]}</strong></td><td style="color:var(--text-dim);font-family:DM Sans">{zone_labels[i]}</td><td>{hr_ranges[i]}</td><td class="power-only">{pw_strs[i]}W</td><td class="power-only pct-col" style="color:var(--text-dim)">{pct_strs[i]}%</td></tr>'
    
    # Race readiness cards
    race_cards = ''
    from datetime import date as dt_date
    
    # Get current RFL trend and recent trajectory for projection
    _rfl_current = zone_data.get('current_rfl', 0.90)
    # RFL projection slope from zone_data (computed in get_zone_data where df is available)
    _rfl_proj_per_day = zone_data.get('rfl_proj_per_day', 0.0)
    
    _priority_colors = {'A': '#f87171', 'B': '#fbbf24', 'C': '#8b90a0'}
    _priority_labels = {'A': 'A RACE', 'B': 'B RACE', 'C': 'C RACE'}
    _taper_days = {'A': 14, 'B': 7, 'C': 0}
    
    # Surface-specific multipliers
    # power_mult: sustainable fraction of road CP factor on this surface
    # re_mult: running economy relative to road (speed per unit power)
    _surface_factors = {
        'indoor_track': {'power_mult': 1.00, 're_mult': 1.04},
        'track':        {'power_mult': 1.00, 're_mult': 1.02},
        'road':         {'power_mult': 1.00, 're_mult': 1.00},
        'trail':        {'power_mult': 0.95, 're_mult': 0.97},
        'undulating_trail': {'power_mult': 0.90, 're_mult': 0.95}
    }
    # Continuous power-duration curve: piecewise linear in log-distance space
    # Exact at anchor points, interpolated between them
    import math as _math
    _pd_anchors = [(3.0, 1.07), (5.0, 1.05), (10.0, 1.00), (21.097, 0.95), (42.195, 0.90)]
    def _road_cp_factor(dist_km):
        d = max(dist_km, 1.0)
        ld = _math.log(d)
        if ld <= _math.log(_pd_anchors[0][0]): return _pd_anchors[0][1]
        if ld >= _math.log(_pd_anchors[-1][0]): return _pd_anchors[-1][1]
        for i in range(len(_pd_anchors) - 1):
            l0, l1 = _math.log(_pd_anchors[i][0]), _math.log(_pd_anchors[i+1][0])
            if l0 <= ld <= l1:
                frac = (ld - l0) / (l1 - l0)
                return _pd_anchors[i][1] + frac * (_pd_anchors[i+1][1] - _pd_anchors[i][1])
        return 1.0
    
    # Pre-compute specificity minutes for race cards (server-side for no-JS)
    from datetime import date as _date_cls, timedelta as _td
    from datetime import datetime as _dt
    _today = _date_cls.today()
    _c14 = _today - _td(days=14)
    _c28 = _today - _td(days=28)
    
    _spec_zones_map = {
        'Sub-5K': ['Sub-5K'],
        '5K': ['Sub-5K', '5K'],
        '10K': ['10K'],
        'HM': ['HM'],
        'Mara': ['Mara'],
    }
    
    # Build race effort zones based on mode
    # For Stryd: use power zones (npower field)
    # For GAP: use pace zones (avg_pace_skm field, lower = faster)
    # Fallback: HR zones
    _use_power = (_cfg_power_mode == 'stryd' and cp > 0)
    _use_pace = (_cfg_power_mode == 'gap')
    
    if _use_power:
        # Power race zones: boundaries between adjacent race powers
        _m_pw = round(cp * 0.90)  # marathon
        _h_pw = round(cp * 0.95)  # HM
        _t_pw = round(cp * 1.00)  # 10K
        _f_pw = round(cp * 1.05)  # 5K
        _above = round(_f_pw * 1.05)  # Sub-5K
        # Zones: (id, lo_power, hi_power) — higher power = faster
        _mh = round((_m_pw + _h_pw) / 2)
        _ht = round((_h_pw + _t_pw) / 2)
        _tf = round((_t_pw + _f_pw) / 2)
        _spec_race_zones = [
            ('Sub-5K', _above, 9999),
            ('5K', _tf, _above),
            ('10K', _ht, _tf),
            ('HM', _mh, _ht),
            ('Mara', round(_m_pw * 0.93), _mh),
            ('Other', 0, round(_m_pw * 0.93)),
        ]
    elif _use_pace:
        # Pace race zones (sec/km) — lower pace = faster
        _p5 = gap_paces.get('5k', 240)
        _p10 = gap_paces.get('10k', 252)
        _ph = gap_paces.get('hm', 265)
        _pm = gap_paces.get('marathon', 282)
        _s5 = round(_p5 * 0.97)
        _m5 = round((_p5 + _p10) / 2)
        _mt = round((_p10 + _ph) / 2)
        _mh_p = round((_ph + _pm) / 2)
        _slow = round(_pm * 1.07)
        # Pace zones: (id, lo_pace, hi_pace) — INVERTED: fast=low number
        _spec_race_zones = [
            ('Sub-5K', 0, _s5),
            ('5K', _s5, _m5),
            ('10K', _m5, _mt),
            ('HM', _mt, _mh_p),
            ('Mara', _mh_p, _slow),
            ('Other', _slow, 9999),
        ]
    else:
        # HR fallback
        _spec_race_zones = [
            ('Sub-5K', _rhr_sub5k, 9999),
            ('5K', _rhr_5k, _rhr_sub5k),
            ('10K', _rhr_10k, _rhr_5k),
            ('HM', _rhr_hm, _rhr_10k),
            ('Mara', _rhr_mara, _rhr_hm),
            ('Other', 0, _rhr_mara),
        ]
    
    _spec_values = {}
    _runs_for_spec = zone_data['runs'] if zone_data else []
    for race_idx_s, race_s in enumerate(PLANNED_RACES_DASH):
        dk = race_s['distance_key']
        at_or_above = _spec_zones_map.get(dk, [dk])
        m14 = 0.0
        m28 = 0.0
        for r in _runs_for_spec:
            rd = _dt.strptime(r['date'], '%Y-%m-%d').date()
            if rd < _c28:
                continue
            mins = r.get('duration_min', 0) or 0
            if mins <= 0:
                continue
            
            if _use_power:
                val = r.get('npower', 0) or 0
                if val <= 0:
                    continue
                # Estimate: 15% at 110% power, 60% at avg, 25% at 88%
                fracs = [(1.10, 0.15), (1.0, 0.60), (0.88, 0.25)]
            elif _use_pace:
                val = r.get('avg_pace_skm', 0) or 0
                if val <= 0:
                    continue
                # Estimate: 15% at 92% pace (faster), 60% at avg, 25% at 110% (slower)
                fracs = [(0.92, 0.15), (1.0, 0.60), (1.10, 0.25)]
            else:
                val = r.get('avg_hr', 0) or 0
                if val <= 0:
                    continue
                fracs = [(1.06, 0.15), (1.0, 0.60), (0.92, 0.25)]
            
            for frac, min_frac in fracs:
                test_val = val * frac
                assigned = 'Other'
                for zid, lo, hi in _spec_race_zones:
                    if zid == 'Other':
                        continue
                    if _use_pace:
                        # Pace: lower is faster. Zone = [lo, hi) where lo < hi
                        if lo <= test_val < hi:
                            assigned = zid
                            break
                    else:
                        # Power/HR: higher is harder. Zone = [lo, hi)
                        if lo <= test_val < hi:
                            assigned = zid
                            break
                if assigned in at_or_above:
                    contrib = mins * min_frac
                    if rd >= _c14:
                        m14 += contrib
                    m28 += contrib
        _spec_values[race_idx_s] = (round(m14), round(m28))
    
    for race_idx, race in enumerate(PLANNED_RACES_DASH):
        # Skip past races
        try:
            _race_dt = datetime.strptime(race['date'], '%Y-%m-%d').date()
            if _race_dt < dt_date.today():
                continue
        except (ValueError, KeyError):
            pass
        key = race['distance_key']
        priority = race.get('priority', 'B')
        surface = race.get('surface', 'road')
        
        # Use actual race distance for continuous power-duration curve
        dist_km = race.get('distance_km', RACE_DISTANCES_KM_DASH.get(key, 5.0))
        road_factor = _road_cp_factor(dist_km)
        
        # Surface adjustments
        sf = _surface_factors.get(surface, _surface_factors['road'])
        factor = road_factor * sf['power_mult']
        surface_re = re_p90 * sf['re_mult']
        
        pw = round(cp * factor)
        dist_m = dist_km * 1000
        band = round(pw * 0.03)
        
        # For standard road distances, use StepB predictions (matches stats grid exactly)
        # Use mode-appropriate predictions: GAP mode → _mode_gap, Sim → _mode_sim
        _stepb_key_map = {'5K': '5k_raw', '10K': '10k_raw', 'HM': 'hm_raw', 'Mara': 'marathon_raw'}
        _rp = stats.get('race_predictions', {}) if stats else {}
        # Pick mode-appropriate prediction source
        if _cfg_power_mode in ('gap', 'sim'):
            _mode_preds = _rp.get(f'_mode_{_cfg_power_mode}', {})
            _stepb_raw = _mode_preds.get(_stepb_key_map.get(key, ''), 0) if surface == 'road' else 0
        else:
            _stepb_raw = _rp.get(_stepb_key_map.get(key, ''), 0) if surface == 'road' else 0
        if _stepb_raw and _stepb_raw > 0:
            t = _stepb_raw
        else:
            speed = (pw / ATHLETE_MASS_KG_DASH) * surface_re
            t = round(dist_m / speed) if speed > 0 else 0
        
        mins = t // 60
        secs = t % 60
        hrs = mins // 60
        t_str = f"{hrs}:{mins%60:02d}:{secs:02d}" if hrs > 0 else f"{mins}:{secs:02d}"
        pace = t / dist_km if dist_km > 0 and t > 0 else 0
        p_min = int(pace // 60)
        p_sec = int(pace % 60)
        pace_str = f"{p_min}:{p_sec:02d}/km" if pace > 0 else '-'
        race_dt = datetime.strptime(race['date'], '%Y-%m-%d').date()
        days_to = (race_dt - dt_date.today()).days
        if days_to < 0:
            days_str = f"{abs(days_to)}d ago"
        elif days_to == 0:
            days_str = "TODAY"
        else:
            days_str = f"{days_to}d"
        
        # RFL projection at race day
        proj_rfl = _rfl_current + (_rfl_proj_per_day * max(days_to, 0))
        proj_rfl = max(0, min(1.05, proj_rfl))  # sanity clamp
        proj_show = days_to <= 90 and days_to > 0
        proj_rfl_pct = f"{proj_rfl*100:.1f}%" if proj_show else "—"
        proj_direction = ("↗" if _rfl_proj_per_day > 0.0002 else ("↘" if _rfl_proj_per_day < -0.0002 else "→")) if proj_show else ""
        
        # Taper guidance
        taper_d = _taper_days.get(priority, 7)
        taper_html = ''
        if taper_d > 0 and days_to > 0:
            taper_start = days_to - taper_d
            if taper_start <= 0:
                taper_html = f'<span style="color:#4ade80;font-size:0.72rem">🟢 In taper window</span>'
            elif taper_start <= 7:
                taper_html = f'<span style="color:#fbbf24;font-size:0.72rem">⏳ Taper starts in {taper_start}d</span>'
            else:
                taper_html = f'<span style="color:var(--text-dim);font-size:0.72rem">Taper in {taper_start}d</span>'
        
        p_color = _priority_colors.get(priority, '#8b90a0')
        p_label = _priority_labels.get(priority, priority)
        _surface_labels = {'indoor_track': '🏟️ Indoor', 'track': '🏟️ Track', 'road': '🛣️ Road', 'trail': '🌲 Trail', 'undulating_trail': '⛰️ Trail'}
        s_label = _surface_labels.get(surface, '')
        
        race_cards += f'''<div class="rc">
            <div class="rh">
                <span class="rn">{race['name']} <span style="font-size:0.65rem;padding:2px 6px;border-radius:4px;background:{p_color}22;color:{p_color};font-weight:600;margin-left:6px;vertical-align:middle">{p_label}</span>{f' <span style="font-size:0.65rem;color:var(--text-dim)">{s_label}</span>' if surface != 'road' else ''}</span>
                <span class="rd">{race['date']} · {days_str}</span>
            </div>
            <div class="rs">
                <div class="power-only"><div class="rv" style="color:var(--accent)" id="race-pw-{race_idx}">{pw}W</div><div class="rl">Target</div><div class="rx">±{band}W</div></div>
                <div class="pace-target" style="display:none"><div class="rv" style="color:#4ade80" id="race-pace-{race_idx}">{pace_str}</div><div class="rl">Target pace</div></div>
                <div><div class="rv" id="race-pred-{race_idx}">{t_str}</div><div class="rl">Predicted</div><div class="rx power-only">{pace_str}</div></div>
                <div><div class="rv" id="spec14_{race_idx}">{_spec_values.get(race_idx, (0, 0))[0]}<span style="font-size:0.75rem;color:var(--text-dim)">min</span></div><div class="rl">14d at effort</div></div>
                <div><div class="rv" id="spec28_{race_idx}">{_spec_values.get(race_idx, (0, 0))[1]}<span style="font-size:0.75rem;color:var(--text-dim)">min</span></div><div class="rl">28d at effort</div></div>
            </div>
            <div style="margin-top:6px">{taper_html}</div>
        </div>'''
    
    # Zone run data as JSON for JS
    import json as _json
    zone_runs_json = _json.dumps(zone_data['runs'])
    
    # ── Server-side static zone bars (no-JS fallback) ──────────────────
    # Pre-render weekly HR zone bars and per-run bars as static HTML
    # JS will overwrite these when it runs (for mode switching)
    _hr_zones = [
        ('Z1', 'Easy', 0, _z12_hr, '#3b82f6'),
        ('Z2', 'Aerobic', _z12_hr, _z23_hr, '#22c55e'),
        ('Z3', 'Tempo', _z23_hr, _z34_hr, '#eab308'),
        ('Z4', 'Threshold', _z34_hr, _z45_hr, '#f97316'),
        ('Z5', 'Max', _z45_hr, 9999, '#ef4444'),
    ]
    
    def _assign_hr_zone(hr):
        if not hr or hr <= 0:
            return 'Z5'
        for zid, _, lo, hi, _ in _hr_zones:
            if hr < hi:
                return zid
        return 'Z5'
    
    _runs = zone_data['runs']
    
    # Weekly aggregation (last 8 weeks)
    from collections import defaultdict
    from datetime import datetime as _dt
    _weekly = defaultdict(lambda: {z[0]: 0.0 for z in _hr_zones})
    _weekly_total = defaultdict(float)
    _week_labels = {}  # iso_key -> display label
    for r in _runs:
        d = _dt.strptime(r['date'], '%Y-%m-%d')
        # Use Monday of the week as sort key (matches JS weekKey)
        day_of_week = d.weekday()  # 0=Monday
        monday = d - _td(days=day_of_week)
        iso_key = monday.strftime('%Y-%m-%d')
        iso = d.isocalendar()
        _week_labels[iso_key] = f"W{iso[1]:02d}/{str(iso[0])[2:]}"
        mins = r.get('duration_min', 0) or 0
        zid = _assign_hr_zone(r.get('avg_hr', 0))
        _weekly[iso_key][zid] += mins
        _weekly_total[iso_key] += mins
    
    _sorted_weeks = sorted(_weekly.keys())[-8:]
    _max_total = max((_weekly_total.get(w, 1) for w in _sorted_weeks), default=1)
    
    _static_wk_rows = ''
    for wk in _sorted_weeks:
        zm = _weekly[wk]
        total = _weekly_total[wk]
        label = _week_labels.get(wk, wk)
        _static_wk_rows += f'<div class="wr"><div class="wl">{label}</div><div class="wb">'
        for zid, zname, _, _, zcol in _hr_zones:
            if zm[zid] > 0:
                pct = (zm[zid] / _max_total) * 100
                _static_wk_rows += f'<div class="ws" style="width:{pct:.1f}%;background:{zcol}"><div class="tip">{zname}: {round(zm[zid])} min</div></div>'
        _static_wk_rows += f'</div><div class="wt">{round(total)} min</div></div>'
    
    _static_wk_legend = ''.join(
        f'<div class="lg"><div class="lsw" style="background:{zcol}"></div>{zname}</div>'
        for _, zname, _, _, zcol in _hr_zones
    )
    
    # Per-run static bars (last 30, stacked horizontal bars via CSS)
    _last30 = _runs[-30:]
    _max_run_mins = max((r.get('duration_min', 0) or 0 for r in _last30), default= 1)
    _static_pr_rows = ''
    for r in _last30:
        d = _dt.strptime(r['date'], '%Y-%m-%d')
        label = f"{d.day}/{d.month}"
        mins = r.get('duration_min', 0) or 0
        hr = r.get('avg_hr', 0) or 0
        zid = _assign_hr_zone(hr)
        zcol = next((c for z, _, _, _, c in _hr_zones if z == zid), '#3b82f6')
        zname = next((n for z, n, _, _, _ in _hr_zones if z == zid), 'Easy')
        pct = (mins / _max_run_mins) * 100 if _max_run_mins > 0 else 0
        _static_pr_rows += f'<div class="wr"><div class="wl">{label}</div><div class="wb">'
        _static_pr_rows += f'<div class="ws" style="width:{pct:.1f}%;background:{zcol}"><div class="tip">{r.get("name","")[:30]}: {round(mins)} min {zname}</div></div>'
        _static_pr_rows += f'</div><div class="wt">{round(mins)} min</div></div>'
    
    # Specificity: compute 14d and 28d minutes at race effort for each planned race
    
    _planned_races_zone_json = _json.dumps([
        {"name": r["name"], "date": r["date"], "priority": r.get("priority", "C"),
         "distance_key": r["distance_key"], "distance_km": r["distance_km"],
         "surface": r.get("surface", "road")} for r in PLANNED_RACES_DASH
    ])
    
    return f'''
    <div class="card" id="zone-table-card">
        <h2>Training Zones <span class="badge" id="zone-badge">CP {cp}W · LTHR {LTHR_DASH} · Max ~{MAX_HR_DASH}</span></h2>
        <table class="zt"><thead><tr><th>Zone</th><th></th><th>HR</th><th class="power-only">Power</th><th class="power-only pct-col">%CP</th></tr></thead><tbody>{combined_rows}</tbody></table>
    </div>

    <div class="card" id="race-readiness-card">
        <h2>🎯 Race Readiness <span class="badge">±3% of target</span></h2>
        {race_cards}
    </div>

    <div class="card">
        <h2>📊 Weekly Zone Volume</h2>
        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px">
            <div class="chart-toggle" id="wk-mode">
                <button class="active" onclick="setWM('hr',this)">HR Zone</button>
                <button class="power-only" onclick="setWM('power',this)">Power Zone</button>
                <button class="power-only" onclick="setWM('race',this)">Race (W)</button>
                <button class="gap-only" onclick="setWM('race',this)">Race (Pace)</button>
                <button onclick="setWM('racehr',this)">Race (HR)</button>
            </div>
            <div class="chart-toggle" id="wk-period">
                <button class="active" onclick="setWP(8,this)">8w</button>
                <button onclick="setWP(12,this)">12w</button>
                <button onclick="setWP(16,this)">16w</button>
            </div>
        </div>
        <div id="wk-bars">{_static_wk_rows}</div>
        <div id="wk-leg" class="legend">{_static_wk_legend}</div>
    </div>

    <div class="card">
        <h2>📊 Per-Run Distribution <span class="badge">last 30</span></h2>
        <div style="margin-bottom:10px">
            <div class="chart-toggle" id="pr-mode">
                <button class="active" onclick="setPR('hr',this)">HR Zone</button>
                <button class="power-only" onclick="setPR('power',this)">Power Zone</button>
                <button class="power-only" onclick="setPR('race',this)">Race (W)</button>
                <button class="gap-only" onclick="setPR('race',this)">Race (Pace)</button>
                <button onclick="setPR('racehr',this)">Race (HR)</button>
            </div>
        </div>
        <div id="pr-static">{_static_pr_rows}</div>
        <div class="chart-wrapper" id="pr-canvas-wrap" style="display:none"><canvas id="prChart"></canvas></div>
        <div id="pr-leg" class="legend"></div>
        <div class="note">Zone split from 30s rolling average of per-second power/HR data where available.</div>
    </div>

    <script>
    const ZONE_RUNS=''' + zone_runs_json + ''';
    const PLANNED_RACES=''' + _planned_races_zone_json + f''';
    const ZONE_CP={cp};const ZONE_PEAK_CP={PEAK_CP_WATTS_DASH};const ZONE_MASS={ATHLETE_MASS_KG_DASH};const ZONE_RE={re_p90};const ZONE_LTHR={LTHR_DASH};const ZONE_MAXHR={MAX_HR_DASH};
    const HR_Z=[
      {{id:'Z1',name:'Easy',lo:0,hi:{_z12_hr},c:'#3b82f6'}},
      {{id:'Z2',name:'Aerobic',lo:{_z12_hr},hi:{_z23_hr},c:'#22c55e'}},
      {{id:'Z3',name:'Tempo',lo:{_z23_hr},hi:{_z34_hr},c:'#eab308'}},
      {{id:'Z4',name:'Threshold',lo:{_z34_hr},hi:{_z45_hr},c:'#f97316'}},
      {{id:'Z5',name:'Max',lo:{_z45_hr},hi:9999,c:'#ef4444'}}
    ];
    const RF_THR=ZONE_CP/ZONE_LTHR;
    const PW_Z12=Math.round(RF_THR*{_z12_hr});
    const PW_Z23=Math.round(RF_THR*{_z23_hr});
    const PW_Z34=Math.round(RF_THR*{_z34_hr});
    const PW_Z45=ZONE_CP;
    const PW_Z=[
      {{id:'Z1',name:'Easy',lo:0,hi:PW_Z12,c:'#3b82f6'}},
      {{id:'Z2',name:'Aerobic',lo:PW_Z12,hi:PW_Z23,c:'#22c55e'}},
      {{id:'Z3',name:'Tempo',lo:PW_Z23,hi:PW_Z34,c:'#eab308'}},
      {{id:'Z4',name:'Threshold',lo:PW_Z34,hi:PW_Z45,c:'#f97316'}},
      {{id:'Z5',name:'Max',lo:PW_Z45,hi:9999,c:'#ef4444'}}
    ];
    const RACE_CFG={{}};
    ['Sub-5K','5K','10K','HM','Mara'].forEach(k=>{{
      const f={{'Sub-5K':1.07,'5K':1.05,'10K':1.00,'HM':0.95,'Mara':0.90}}[k],pw=Math.round(ZONE_CP*f);
      const dists={{'Sub-5K':3,'5K':5,'10K':10,'HM':21.097,'Mara':42.195}};
      const cols={{'Sub-5K':'#4ade80','5K':'#f472b6','10K':'#fb923c','HM':'#a78bfa','Mara':'#38bdf8'}};
      RACE_CFG[k]={{pw,dist:dists[k],color:cols[k]}};
    }});
    function makeRacePwZones(){{const pw=RACE_CFG,m=pw['Mara'].pw,h=pw['HM'].pw,t=pw['10K'].pw,f=pw['5K'].pw;const mh=Math.round((m+h)/2),ht=Math.round((h+t)/2),tf=Math.round((t+f)/2),above=Math.round(f*1.05);return[{{id:'Sub-5K',name:'Sub-5K',lo:above,hi:9999,c:'#4ade80'}},{{id:'5K',name:'5K',lo:tf,hi:above,c:'#f472b6'}},{{id:'10K',name:'10K',lo:ht,hi:tf,c:'#fb923c'}},{{id:'HM',name:'HM',lo:mh,hi:ht,c:'#a78bfa'}},{{id:'Mara',name:'Mara',lo:Math.round(m*0.93),hi:mh,c:'#38bdf8'}},{{id:'Other',name:'Other',lo:0,hi:Math.round(m*0.93),c:'#4b5563'}}];}}
    function makeRaceHrZones(){{return[{{id:'Sub-5K',name:'Sub-5K',lo:{_rhr_sub5k},hi:9999,c:'#4ade80'}},{{id:'5K',name:'5K',lo:{_rhr_5k},hi:{_rhr_sub5k},c:'#f472b6'}},{{id:'10K',name:'10K',lo:{_rhr_10k},hi:{_rhr_5k},c:'#fb923c'}},{{id:'HM',name:'HM',lo:{_rhr_hm},hi:{_rhr_10k},c:'#a78bfa'}},{{id:'Mara',name:'Mara',lo:{_rhr_mara},hi:{_rhr_hm},c:'#38bdf8'}},{{id:'Other',name:'Other',lo:0,hi:{_rhr_mara},c:'#4b5563'}}];}}
    function makeRacePaceZones(){{const p5={pace_5k},p10={pace_10k},ph={pace_hm},pm={pace_mara};const s5=Math.round(p5*0.97),m5=Math.round((p5+p10)/2),mt=Math.round((p10+ph)/2),mh=Math.round((ph+pm)/2),slow=Math.round(pm*1.07);return[{{id:'Sub-5K',name:'Sub-5K',lo:0,hi:s5,c:'#4ade80'}},{{id:'5K',name:'5K',lo:s5,hi:m5,c:'#f472b6'}},{{id:'10K',name:'10K',lo:m5,hi:mt,c:'#fb923c'}},{{id:'HM',name:'HM',lo:mt,hi:mh,c:'#a78bfa'}},{{id:'Mara',name:'Mara',lo:mh,hi:slow,c:'#38bdf8'}},{{id:'Other',name:'Other',lo:slow,hi:9999,c:'#4b5563'}}];}}
    const RACE_PW_Z=makeRacePwZones(),RACE_HR_Z=makeRaceHrZones(),RACE_PACE_Z=makeRacePaceZones();
    function zonesFor(m){{if(m==='hr')return HR_Z;if(m==='power')return PW_Z;if(m==='race'){{if(typeof currentMode!=='undefined'&&currentMode==='gap')return RACE_PACE_Z;return RACE_PW_Z;}}if(m==='racehr')return RACE_HR_Z;return HR_Z;}}
    function valFor(r,m){{if(m==='hr'||m==='racehr')return r.avg_hr;if(m==='race'&&typeof currentMode!=='undefined'&&currentMode==='gap')return r.avg_pace_skm;return r.npower;}}
    function isRaceMode(m){{return m==='race'||m==='racehr';}}
    function assignZ(val,zones,rm){{if(!val||val<=0)return zones[zones.length-1].id;if(rm){{for(const z of zones){{if(z.id==='Other')continue;if(val>=z.lo&&val<=z.hi)return z.id;}}return'Other';}}for(const z of zones){{if(val<z.hi)return z.id;}}return zones[zones.length-1].id;}}
    function assignToZone(val,zones,result,mins){{for(const z of zones){{if(z.id==='Other')continue;if(val>=z.lo){{result[z.id]+=mins;return;}}}}result['Other']+=mins;}}
    function assignToZonePace(val,zones,result,mins){{for(const z of zones){{if(z.id==='Other')continue;if(val>=z.lo&&val<z.hi){{result[z.id]+=mins;return;}}}}result['Other']+=mins;}}
    function estimateRaceEffortMins(r,zones){{const result={{}};zones.forEach(z=>result[z.id]=0);const mins=r.duration_min||0,pw=r.npower||0;if(!pw||!mins){{result['Other']=mins;return result;}}assignToZone(pw*1.10,zones,result,mins*0.15);assignToZone(pw,zones,result,mins*0.60);assignToZone(pw*0.88,zones,result,mins*0.25);return result;}}
    function estimateRaceEffortMinsPace(r,zones){{const result={{}};zones.forEach(z=>result[z.id]=0);const mins=r.duration_min||0,p=r.avg_pace_skm||0;if(!p||!mins){{result['Other']=mins;return result;}}assignToZonePace(Math.round(p*0.92),zones,result,mins*0.15);assignToZonePace(p,zones,result,mins*0.60);assignToZonePace(Math.round(p*1.10),zones,result,mins*0.25);return result;}}
    function estimateRaceEffortMinsHR(r,zones){{const result={{}};zones.forEach(z=>result[z.id]=0);const mins=r.duration_min||0,hr=r.avg_hr||0;if(!hr||!mins){{result['Other']=mins;return result;}}assignToZone(hr*1.06,zones,result,mins*0.15);assignToZone(hr,zones,result,mins*0.60);assignToZone(hr*0.92,zones,result,mins*0.25);return result;}}
    function getZoneMins(r,mode,zones){{
      // Use pre-computed NPZ zone data if available
      const isGapMode=(typeof currentMode!=='undefined'&&currentMode==='gap');
      // In GAP mode for race zones, prefer rpz_gap (per-second GAP pace data)
      if(isGapMode&&mode==='race'&&r.rpz_gap){{const result={{}};zones.forEach(z=>result[z.id]=0);Object.keys(r.rpz_gap).forEach(k=>{{if(result[k]!==undefined)result[k]=r.rpz_gap[k];}});return result;}}
      // Skip power-based race/power zones in GAP mode
      const key={{hr:'hz',power:'pz',race:'rpz',racehr:'rhz'}}[mode];
      if(r[key]&&!(isGapMode&&(key==='rpz'||key==='pz'))){{const result={{}};zones.forEach(z=>result[z.id]=0);Object.keys(r[key]).forEach(k=>{{if(result[k]!==undefined)result[k]=r[key][k];}});return result;}}
      // Fallback to heuristic
      if(mode==='race'){{if(isGapMode)return estimateRaceEffortMinsPace(r,zones);return estimateRaceEffortMins(r,zones);}}
      if(mode==='racehr')return estimateRaceEffortMinsHR(r,zones);
      // HR/Power zone fallback: assign all time to primary zone
      const result={{}};zones.forEach(z=>result[z.id]=0);const mins=r.duration_min||0,v=valFor(r,mode),zid=assignZ(v,zones,false);if(result[zid]!==undefined)result[zid]=mins;return result;
    }}
    const SPEC_ZONES={{'Sub-5K':['Sub-5K'],'5K':['Sub-5K','5K'],'10K':['10K'],'HM':['HM'],'Mara':['Mara']}};
    function calcSpecificity(){{const today=new Date();const c14=new Date(today);c14.setDate(c14.getDate()-14);const c28=new Date(today);c28.setDate(c28.getDate()-28);const useGAP=(typeof currentMode!=='undefined'&&currentMode==='gap');const zones=useGAP?RACE_PACE_Z:RACE_PW_Z;const modeKey='race';PLANNED_RACES.forEach((tgt,idx)=>{{let m14=0,m28=0;const atOrAbove=SPEC_ZONES[tgt.distance_key]||[tgt.distance_key];ZONE_RUNS.forEach(r=>{{const d=new Date(r.date);if(d<c28)return;const est=getZoneMins(r,modeKey,zones);let mins=0;atOrAbove.forEach(k=>{{mins+=(est[k]||0);}});if(d>=c14)m14+=mins;m28+=mins;}});const e14=document.getElementById('spec14_'+idx),e28=document.getElementById('spec28_'+idx);if(e14)e14.innerHTML=Math.round(m14)+'<span style="font-size:0.75rem;color:var(--text-dim)">min</span>';if(e28)e28.innerHTML=Math.round(m28)+'<span style="font-size:0.75rem;color:var(--text-dim)">min</span>';}});}}calcSpecificity();
    // Weekly zone bars
    function weekKey(ds){{const d=new Date(ds),day=d.getDay(),m=new Date(d);m.setDate(d.getDate()-((day+6)%7));return m.toISOString().slice(0,10);}}
    function fmtWk(s){{const d=new Date(s),tmp=new Date(d.valueOf());tmp.setDate(tmp.getDate()+3-(tmp.getDay()+6)%7);const w1=new Date(tmp.getFullYear(),0,4);const wk=1+Math.round(((tmp-w1)/864e5-3+(w1.getDay()+6)%7)/7);return'W'+String(wk).padStart(2,'0')+'/'+String(tmp.getFullYear()).slice(-2);}}
    var wkMode='hr',wkN=8;
    function renderWk(){{const el=document.getElementById('wk-bars');el.innerHTML='';const zones=zonesFor(wkMode),rm=isRaceMode(wkMode);const weeks={{}};ZONE_RUNS.forEach(r=>{{const w=weekKey(r.date);if(!weeks[w])weeks[w]=[];weeks[w].push(r);}});const sorted=Object.keys(weeks).sort().slice(-wkN);let maxT=0;const wd=sorted.map(wk=>{{const zm={{}};zones.forEach(z=>zm[z.id]=0);let t=0;weeks[wk].forEach(r=>{{const mins=r.duration_min||0;const est=getZoneMins(r,wkMode,zones);Object.keys(est).forEach(zid=>{{if(zm[zid]!==undefined)zm[zid]+=est[zid];}});t+=mins;}});if(t>maxT)maxT=t;return{{week:wk,zm,t}};}});wd.forEach(w=>{{const row=document.createElement('div');row.className='wr';const lbl=document.createElement('div');lbl.className='wl';lbl.textContent=fmtWk(w.week);row.appendChild(lbl);const bar=document.createElement('div');bar.className='wb';zones.forEach(z=>{{if(w.zm[z.id]>0){{const pct=(w.zm[z.id]/maxT)*100,seg=document.createElement('div');seg.className='ws';seg.style.width=pct+'%';seg.style.background=z.c;seg.innerHTML='<div class="tip">'+z.name+': '+Math.round(w.zm[z.id])+' min</div>';bar.appendChild(seg);}}}});row.appendChild(bar);const tot=document.createElement('div');tot.className='wt';tot.textContent=Math.round(w.t)+' min';row.appendChild(tot);el.appendChild(row);}});document.getElementById('wk-leg').innerHTML=zones.map(z=>'<div class="lg"><div class="lsw" style="background:'+z.c+'"></div>'+z.name+'</div>').join('');}}
    function setWM(m,btn){{wkMode=m;document.querySelectorAll('#wk-mode button').forEach(b=>b.classList.remove('active'));btn.classList.add('active');renderWk();}}
    function setWP(n,btn){{wkN=n;document.querySelectorAll('#wk-period button').forEach(b=>b.classList.remove('active'));btn.classList.add('active');renderWk();}}
    renderWk();
    // Per-run chart
    var prMode='hr',prChart=null;
    function renderPR(){{const ctx=document.getElementById('prChart').getContext('2d');const last30=ZONE_RUNS.slice(-30),zones=zonesFor(prMode),rm=isRaceMode(prMode);const labels=last30.map(r=>{{const d=new Date(r.date);return d.getDate()+'/'+(d.getMonth()+1);}});const datasets=zones.map((z,zi)=>({{label:z.name,data:last30.map(r=>{{const mins=r.duration_min||0;const est=getZoneMins(r,prMode,zones);return Math.round(est[z.id]||0);}}),backgroundColor:z.c+'cc',borderWidth:0,borderRadius:2}}));if(prChart)prChart.destroy();prChart=new Chart(ctx,{{type:'bar',data:{{labels,datasets}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{title:items=>{{const i=items[0].dataIndex;return last30[i].name;}},label:item=>item.dataset.label+': '+item.raw+' min'}}}}}},scales:{{x:{{stacked:true,grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:'#8b90a0',font:{{size:10,family:"'JetBrains Mono'"}}}}}},y:{{stacked:true,grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:'#8b90a0',font:{{size:10,family:"'JetBrains Mono'"}},callback:v=>v+' min'}}}}}}}}}});document.getElementById('pr-leg').innerHTML=zones.map(z=>'<div class="lg"><div class="lsw" style="background:'+z.c+'"></div>'+z.name+'</div>').join('');}}
    function setPR(m,btn){{prMode=m;document.querySelectorAll('#pr-mode button').forEach(b=>b.classList.remove('active'));btn.classList.add('active');try{{renderPR();}}catch(e){{}}}}
    try{{renderPR();var _ps=document.getElementById('pr-static');if(_ps)_ps.style.display='none';var _pc=document.getElementById('pr-canvas-wrap');if(_pc)_pc.style.display='';}}catch(e){{console.warn('Chart.js required for Per-Run chart:',e);}}
    </script>
'''


# ============================================================================
# v51: ALERT BANNER GENERATOR
# ============================================================================
def _generate_alert_banner(alert_data, critical_power=None):
    """Generate dark-themed alert banner with per-mode switching."""
    import json as _json
    
    # alert_data is now {stryd: [...], gap: [...], sim: [...]}
    if not alert_data:
        alert_data = {'stryd': [], 'gap': [], 'sim': []}
    alert_json = _json.dumps({
        mode: [{'name': a['name'], 'level': a['level'], 'icon': a['icon'], 'detail': a.get('detail', '')} for a in alerts]
        for mode, alerts in alert_data.items()
    })
    
    cp_html = f'<span class="power-only" id="banner-cp" style="font-size:0.82em;color:#8b90a0;">⚡ CP {critical_power}W</span>' if critical_power else ''
    
    return f'''<div id="alert-banner"></div>
    <script>
    const _alertData = {alert_json};
    function renderAlertBanner(mode) {{
        const el = document.getElementById('alert-banner');
        const alerts = _alertData[mode] || [];
        const ms = (typeof modeStats !== 'undefined') ? modeStats[mode] : null;
        const cpVal = ms ? ms.cp : {critical_power or 0};
        const cpHtml = (mode !== 'gap' && cpVal > 0) ? '<span class="power-only" style="font-size:0.82em;color:#8b90a0;">⚡ CP ' + cpVal + 'W</span>' : '';
        if (alerts.length === 0) {{
            el.innerHTML = '<div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.3);border-radius:10px;padding:10px 16px;margin-bottom:14px;display:flex;justify-content:center;align-items:center;gap:8px;"><span style="font-size:1.1em;">🟢</span> <strong style="color:#4ade80;">All clear</strong> <span style="color:#8b90a0;">— no alerts</span>' + cpHtml + '</div>';
            return;
        }}
        const hasConcern = alerts.some(a => a.level === 'concern');
        const bg = hasConcern ? 'rgba(239,68,68,0.08)' : 'rgba(234,179,8,0.08)';
        const brd = hasConcern ? 'rgba(239,68,68,0.3)' : 'rgba(234,179,8,0.3)';
        const icon = hasConcern ? '🔴' : '🟡';
        const lc = hasConcern ? '#f87171' : '#fbbf24';
        const n = alerts.length;
        const s = n > 1 ? 's' : '';
        const items = alerts.map(a => '<div style="margin:4px 0;font-size:0.85em;color:#e4e7ef;">' + a.icon + ' <strong>' + a.name + '</strong><span style="color:#8b90a0"> ' + a.detail + '</span></div>').join('');
        el.innerHTML = '<div style="background:'+bg+';border:1px solid '+brd+';border-radius:10px;padding:10px 16px;margin-bottom:14px;"><div style="text-align:center;margin-bottom:4px;"><span style="font-size:1.1em;">'+icon+'</span> <strong style="color:'+lc+';">'+n+' alert'+s+' active</strong>' + cpHtml + '</div>'+items+'</div>';
    }}
    renderAlertBanner(typeof currentMode !== 'undefined' ? currentMode : 'stryd');
    </script>'''

    # Note: renderAlertBanner references modeStats which is defined later in generate_html.
    # The script tag is inline and runs immediately, but modeStats won't exist yet.
    # We handle this by having the function check for modeStats existence (with fallback),
    # and we'll call renderAlertBanner again after modeStats is defined.
    # The initial call uses the fallback CP value.


def generate_html(stats, rf_data, volume_data, ctl_atl_data, ctl_atl_lookup, rfl_trend_dates, rfl_trend_values, rfl_trendline, rfl_projection, rfl_ci_upper, rfl_ci_lower, alltime_rfl_dates, alltime_rfl_values, recent_runs, top_races, alert_data=None, weight_data=None, prediction_data=None, ag_data=None, zone_data=None, rfl_trend_gap=None, rfl_trend_sim=None):
    """Generate the HTML dashboard."""
    
    # --- Helper for None-safe delta formatting ---
    def _fmt_delta(v):
        if v is None:
            return "-"
        return f"{'+' if v > 0 else ''}{v}%"

    _gap_rfl = stats.get("latest_rfl_gap") or stats.get("latest_rfl")
    _gap_rfl_num = float(_gap_rfl) if isinstance(_gap_rfl, (int, float)) else 0
    _sim_rfl = stats.get("latest_rfl_sim") or stats.get("latest_rfl")
    _sim_rfl_num = float(_sim_rfl) if isinstance(_sim_rfl, (int, float)) else 0
    _gap_preds = stats.get("race_predictions", {}).get("_mode_gap") or stats.get("race_predictions", {})
    _sim_preds = stats.get("race_predictions", {}).get("_mode_sim") or stats.get("race_predictions", {})

    # Compute initial display values based on configured power mode
    # This ensures GAP athletes see correct values even before JS runs (mobile, slow load)
    _init_mode = _cfg_power_mode if _cfg_power_mode in ('stryd', 'gap', 'sim') else 'stryd'
    if _init_mode == 'gap':
        _init_rfl = stats.get('latest_rfl_gap') or stats.get('latest_rfl')
        _init_delta = stats.get('rfl_14d_delta_gap') or stats.get('rfl_14d_delta')
        _init_rfl_label = 'RFL (GAP)'
        _init_preds = stats['race_predictions'].get('_mode_gap') or stats['race_predictions']
        _init_easy = stats.get('easy_rfl_gap_gap') if stats.get('easy_rfl_gap_gap') is not None else stats.get('easy_rfl_gap')
        _init_ag = stats['race_predictions'].get('_ag_gap') or stats.get('age_grade')
    elif _init_mode == 'sim':
        _init_rfl = stats.get('latest_rfl_sim') or stats.get('latest_rfl')
        _init_delta = stats.get('rfl_14d_delta_sim') or stats.get('rfl_14d_delta')
        _init_rfl_label = 'RFL (Sim)'
        _init_preds = stats['race_predictions'].get('_mode_sim', dict())
        _init_easy = stats.get('easy_rfl_gap_sim')
        _init_ag = stats['race_predictions'].get('_ag_sim')
    else:
        _init_rfl = stats.get('latest_rfl')
        _init_delta = stats.get('rfl_14d_delta')
        _init_rfl_label = 'RFL Trend'
        _init_preds = stats['race_predictions']
        _init_easy = stats.get('easy_rfl_gap')
        _init_ag = stats.get('age_grade')
    
    # Body class for initial mode
    _body_class = 'gap-mode' if _init_mode == 'gap' else ('sim-mode' if _init_mode == 'sim' else '')
    
    # Extract data for each time range - RF (v51: now includes easy_rfl and race_flags)
    rf_dates_90, rf_values_90, rf_trend_90, rf_easy_90, rf_races_90, rf_gap_trend_90, rf_sim_trend_90, rf_gap_values_90, rf_sim_values_90 = rf_data['90']
    rf_dates_180, rf_values_180, rf_trend_180, rf_easy_180, rf_races_180, rf_gap_trend_180, rf_sim_trend_180, rf_gap_values_180, rf_sim_values_180 = rf_data['180']
    rf_dates_365, rf_values_365, rf_trend_365, rf_easy_365, rf_races_365, rf_gap_trend_365, rf_sim_trend_365, rf_gap_values_365, rf_sim_values_365 = rf_data['365']
    rf_dates_730, rf_values_730, rf_trend_730, rf_easy_730, rf_races_730, rf_gap_trend_730, rf_sim_trend_730, rf_gap_values_730, rf_sim_values_730 = rf_data['730']
    rf_dates_1095, rf_values_1095, rf_trend_1095, rf_easy_1095, rf_races_1095, rf_gap_trend_1095, rf_sim_trend_1095, rf_gap_values_1095, rf_sim_values_1095 = rf_data['1095']
    rf_dates_1825, rf_values_1825, rf_trend_1825, rf_easy_1825, rf_races_1825, rf_gap_trend_1825, rf_sim_trend_1825, rf_gap_values_1825, rf_sim_values_1825 = rf_data['1825']
    rf_dates_all, rf_values_all, rf_trend_all, rf_easy_all, rf_races_all, rf_gap_trend_all, rf_sim_trend_all, rf_gap_values_all, rf_sim_values_all = rf_data['all']
    
    # Volume data - weeks
    week_labels_12, week_distances_12, week_runs_12 = volume_data['W']['12']
    week_labels_26, week_distances_26, week_runs_26 = volume_data['W']['26']
    week_labels_52, week_distances_52, week_runs_52 = volume_data['W']['52']
    # Volume data - months
    month_labels_6, month_distances_6, month_runs_6 = volume_data['M']['6']
    month_labels_12, month_distances_12, month_runs_12 = volume_data['M']['12']
    month_labels_24, month_distances_24, month_runs_24 = volume_data['M']['24']
    month_labels_all, month_distances_all, month_runs_all = volume_data['M']['All']
    # Volume data - years
    year_labels_3, year_distances_3, year_runs_3 = volume_data['Y']['3']
    year_labels_5, year_distances_5, year_runs_5 = volume_data['Y']['5']
    year_labels_all, year_distances_all, year_runs_all = volume_data['Y']['All']
    
    # CTL/ATL data for each time range (now includes projection lines)
    ctl_atl_90 = ctl_atl_data['90']
    ctl_atl_180 = ctl_atl_data['180']
    ctl_atl_365 = ctl_atl_data['365']
    ctl_atl_730 = ctl_atl_data['730']
    ctl_atl_1095 = ctl_atl_data['1095']
    ctl_atl_1825 = ctl_atl_data['1825']
    
    # Helper to safely unpack (handles both old 3-tuple and new 5-tuple format)
    def unpack_ctl_atl(data):
        if len(data) == 5:
            return data  # dates, ctl, atl, ctl_proj, atl_proj
        elif len(data) == 3:
            return data[0], data[1], data[2], [], []  # dates, ctl, atl, empty proj
        else:
            return [], [], [], [], []
    
    ctl_atl_dates_90, ctl_values_90, atl_values_90, ctl_proj_90, atl_proj_90 = unpack_ctl_atl(ctl_atl_90)
    ctl_atl_dates_180, ctl_values_180, atl_values_180, ctl_proj_180, atl_proj_180 = unpack_ctl_atl(ctl_atl_180)
    ctl_atl_dates_365, ctl_values_365, atl_values_365, ctl_proj_365, atl_proj_365 = unpack_ctl_atl(ctl_atl_365)
    ctl_atl_dates_730, ctl_values_730, atl_values_730, ctl_proj_730, atl_proj_730 = unpack_ctl_atl(ctl_atl_730)
    ctl_atl_dates_1095, ctl_values_1095, atl_values_1095, ctl_proj_1095, atl_proj_1095 = unpack_ctl_atl(ctl_atl_1095)
    ctl_atl_dates_1825, ctl_values_1825, atl_values_1825, ctl_proj_1825, atl_proj_1825 = unpack_ctl_atl(ctl_atl_1825)
    
    # Build planned races JSON for JS injection (outside f-string to avoid dict/brace conflicts)
    _planned_races_json = json.dumps([
        {'name': r['name'], 'date': r['date'], 'priority': r.get('priority', 'B'),
         'distance_key': r['distance_key'], 'distance_km': r.get('distance_km', 5.0),
         'surface': r.get('surface', 'road')}
        for r in PLANNED_RACES_DASH
    ])
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>{_cfg_name} Fitness Dashboard</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏃</text></svg>">
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3"></script>
    <style>
        :root {{
            --bg: #0f1117;
            --surface: #1a1d27;
            --surface2: #232733;
            --border: #2e3340;
            --text: #e4e7ef;
            --text-dim: #8b90a0;
            --text-muted: #565b6b;
            --accent: #818cf8;
            --z1: #3b82f6;
            --z2: #22c55e;
            --z3: #eab308;
            --z4: #f97316;
            --z5: #ef4444;
            --grid: rgba(255,255,255,0.04);
        }}
        * {{
            box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            padding: 14px;
            background: var(--bg);
            color: var(--text);
            font-size: 14px;
        }}
        
        h1 {{
            font-size: 1.4rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0 0 4px 0;
            text-align: center;
        }}
        .dash-sub {{
            text-align: center;
            color: var(--text-dim);
            font-size: 0.82rem;
            margin-bottom: 14px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin-bottom: 10px;
        }}
        
        .stat-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 1.45em;
            font-weight: 700;
            color: var(--accent);
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .stat-label {{
            font-size: 0.78em;
            color: var(--text-dim);
            margin-top: 2px;
            font-weight: 500;
        }}
        
        .stat-sub {{
            font-size: 0.7em;
            color: var(--text-muted);
        }}
        
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 12px;
        }}
        
        .card h2 {{
            font-size: 0.9rem;
            font-weight: 600;
            margin: 0 0 10px 0;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .card-header {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 6px;
            margin-bottom: 8px;
        }}
        
        .card-description {{
            font-size: 0.75em;
            color: var(--text-dim);
            margin-bottom: 10px;
            line-height: 1.3;
        }}
        
        .badge {{
            font-size: 0.68rem;
            font-weight: 500;
            background: var(--surface2);
            border: 1px solid var(--border);
            padding: 2px 8px;
            border-radius: 4px;
            color: var(--text-dim);
        }}

        .chart-wrapper {{
            position: relative;
            height: 220px;
            width: 100%;
        }}
        
        /* Zone tables */
        .zt {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
            min-width: 0;
            table-layout: auto;
        }}
        .zt th {{
            text-align: left;
            padding: 5px 6px;
            color: var(--text-dim);
            font-weight: 500;
            border-bottom: 1px solid var(--border);
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .zt td {{
            padding: 5px 6px;
            border-bottom: 1px solid var(--border);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
        }}
        .zt tr:last-child td {{ border-bottom: none; }}
        .zd {{
            width: 7px; height: 7px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 4px;
            vertical-align: middle;
        }}
        
        /* Weekly zone bars */
        .wr {{ display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }}
        .wl {{ font-size: 0.72rem; font-family: 'JetBrains Mono'; color: var(--text-dim); width: 52px; flex-shrink: 0; }}
        .wb {{ flex: 1; height: 20px; display: flex; border-radius: 3px; overflow: visible; background: rgba(255,255,255,0.02); }}
        .ws {{ height: 100%; transition: width 0.3s; position: relative; cursor: pointer; }}
        .ws:first-child {{ border-radius: 3px 0 0 3px; }}
        .ws:last-child {{ border-radius: 0 3px 3px 0; }}
        .ws:only-child {{ border-radius: 3px; }}
        .ws:hover {{ filter: brightness(1.15); }}
        .ws .tip {{ display: none; position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%); background: var(--surface2); border: 1px solid var(--border); padding: 3px 7px; border-radius: 4px; font-size: 0.68rem; white-space: nowrap; z-index: 10; color: var(--text); pointer-events: none; margin-bottom: 4px; }}
        .ws:hover .tip {{ display: block; }}
        .wt {{ font-size: 0.72rem; font-family: 'JetBrains Mono'; color: var(--text-dim); width: 48px; text-align: right; flex-shrink: 0; }}
        
        /* Race readiness cards */
        .rc {{ background: var(--surface2); border-radius: 8px; padding: 12px; margin-bottom: 8px; }}
        .rc:last-child {{ margin-bottom: 0; }}
        .rh {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
        .rn {{ font-weight: 600; font-size: 0.88rem; }}
        .rd {{ font-size: 0.72rem; color: var(--text-dim); font-family: 'JetBrains Mono'; }}
        .rs {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }}
        .rv {{ font-size: 1.1rem; font-weight: 700; font-family: 'JetBrains Mono'; text-align: center; }}
        .rl {{ font-size: 0.68rem; color: var(--text-dim); text-align: center; margin-top: 1px; }}
        .rx {{ font-size: 0.64rem; color: var(--text-muted); text-align: center; }}
        
        /* Legend */
        .legend {{ display: flex; gap: 10px; margin-top: 8px; flex-wrap: wrap; }}
        .lg {{ display: flex; align-items: center; gap: 4px; font-size: 0.7rem; color: var(--text-dim); }}
        .lsw {{ width: 9px; height: 9px; border-radius: 2px; flex-shrink: 0; }}
        
        .note {{
            font-size: 0.72rem;
            color: var(--text-muted);
            margin-top: 10px;
            padding: 8px 10px;
            background: var(--surface2);
            border-radius: 6px;
            border-left: 3px solid var(--accent);
        }}
        
        .table-wrapper {{
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            margin: 0 -14px;
            padding: 0 14px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.78em;
            table-layout: fixed;
            min-width: 500px;
        }}
        
        th, td {{
            padding: 5px 3px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        td:nth-child(2), td:nth-child(3) {{
            white-space: normal;
            word-wrap: break-word;
            line-height: 1.2;
        }}
        
        th {{
            font-weight: 500;
            color: var(--text-dim);
            font-size: 0.7em;
            text-transform: uppercase;
            white-space: nowrap;
            letter-spacing: 0.03em;
        }}
        
        .race-badge {{
            background: #fbbf24;
            color: #000;
            padding: 1px 4px;
            border-radius: 3px;
            font-size: 0.65em;
            font-weight: bold;
        }}
        
        .footer {{
            text-align: center;
            font-size: 0.72em;
            color: var(--text-muted);
            margin-top: 16px;
            padding-top: 12px;
            border-top: 1px solid var(--border);
        }}
        
        @media (min-width: 600px) {{
            body {{
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
            }}
            .stats-grid {{
                grid-template-columns: repeat(4, 1fr);
            }}
            .chart-wrapper {{
                height: 280px;
            }}
            .zgrid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
            }}
        }}
        @media (max-width: 599px) {{
            .rs {{ grid-template-columns: repeat(2, 1fr); }}
        }}
        
        /* Legacy chart classes — dark theme */
        .chart-container {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 12px;
        }}
        .chart-header {{
            display: flex;
            flex-direction: column;
            gap: 4px;
            margin-bottom: 10px;
        }}
        .chart-title {{
            font-size: 0.9rem;
            font-weight: 600;
            font-family: 'DM Sans', -apple-system, sans-serif;
            margin-bottom: 2px;
        }}
        .chart-title-row {{
            display: flex;
            flex-direction: column;
            gap: 6px;
            margin-bottom: 8px;
        }}
        .chart-title-row .chart-title {{
            margin-bottom: 0;
        }}
        .chart-toggle {{
            display: inline-flex;
            gap: 2px;
            background: var(--surface2);
            border-radius: 6px;
            padding: 2px;
            align-self: flex-start;
        }}
        .chart-toggle button {{
            font-family: 'DM Sans', sans-serif;
            font-size: 0.72rem;
            padding: 4px 10px;
            border-radius: 5px;
            border: none;
            background: transparent;
            color: var(--text-dim);
            cursor: pointer;
            transition: all 0.15s;
        }}
        .chart-toggle button:hover {{
            color: var(--text);
        }}
        .chart-toggle button.active {{
            background: var(--accent);
            color: #fff;
        }}
        .chart-desc {{
            font-size: 0.75em;
            color: var(--text-dim);
            margin-bottom: 12px;
            line-height: 1.4;
        }}
        
        /* Zone tables: single-card layout */
        .zone-combined {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}
        @media (max-width: 599px) {{
            .zone-combined {{ grid-template-columns: 1fr; gap: 16px; }}
            .zt {{ font-size: 0.72rem; }}
            .zt th, .zt td {{ padding: 4px 4px; }}
            .zt .pct-col {{ display: none; }}
            .race-card {{ grid-template-columns: 1fr 1fr !important; }}
        }}
    </style>
</head>
<body class="{_body_class}">
<script>try{{Chart.defaults.color="#8b90a0";Chart.defaults.borderColor="rgba(255,255,255,0.04)";Chart.defaults.font.family="'DM Sans',sans-serif";Chart.defaults.plugins.legend.labels.padding=10;
const _today = new Date(); _today.setHours(0,0,0,0);
const _14dFromNow = new Date(_today.getTime() + 14*86400000);
// Format ISO date "2026-02-21" to match chart label formats
function _isoToShort(iso) {{
    // -> "21 Feb" (matches RFL 14-day chart '%d %b')
    const d = new Date(iso + 'T00:00:00');
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    return String(d.getDate()).padStart(2,'0') + ' ' + months[d.getMonth()];
}}
function _isoToMedium(iso) {{
    // -> "21 Feb 26" (matches CTL chart '%d %b %y')
    const d = new Date(iso + 'T00:00:00');
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    return String(d.getDate()).padStart(2,'0') + ' ' + months[d.getMonth()] + ' ' + String(d.getFullYear()).slice(2);
}}
function raceAnnotations(dates) {{
    const pColors = {{'A': '#f87171', 'B': '#fbbf24', 'C': '#6b7280'}};
    const pDash = {{'A': [], 'B': [6,3], 'C': [3,3]}};
    const annots = {{}};
    if (!dates || dates.length === 0) return annots;
    // Find only the NEXT upcoming race
    const upcoming = PLANNED_RACES.filter(r => {{
        const rd = new Date(r.date + 'T00:00:00');
        return rd >= _today;
    }}).sort((a,b) => a.date.localeCompare(b.date));
    if (upcoming.length === 0) return annots;
    const r = upcoming[0];
    const rd = new Date(r.date + 'T00:00:00');
    if (rd > _14dFromNow) return annots;
    const short = _isoToShort(r.date);
    const medium = _isoToMedium(r.date);
    let matchLabel = null;
    if (dates.includes(short)) matchLabel = short;
    else if (dates.includes(medium)) matchLabel = medium;
    if (matchLabel) {{
        annots['race_next'] = {{
            type: 'line', xMin: matchLabel, xMax: matchLabel,
            borderColor: pColors[r.priority] || '#fbbf24',
            borderWidth: r.priority === 'A' ? 2 : 1.5,
            borderDash: pDash[r.priority] || [6,3],
            label: {{ display: true, content: r.name, position: 'start',
                backgroundColor: 'rgba(15,17,23,0.85)',
                color: pColors[r.priority] || '#fbbf24',
                font: {{ size: 10, family: "'DM Sans'" }},
                padding: {{x:4,y:2}}, borderRadius: 3
            }}
        }};
    }}
    return annots;
}}


}}catch(e){{console.warn('Chart.js not loaded:',e);}}
</script>
    <h1>🏃 {_cfg_name}</h1>
    <div class="dash-sub">{datetime.now().strftime("%A %d %B %Y, %H:%M")}</div>
    
    <!-- Phase 2: Mode Toggle -->
    <div class="mode-toggle" style="display:flex;gap:6px;margin:10px 0 8px;align-items:center;{'display:none;' if _cfg_power_mode == 'gap' else ''}">
        <span style="font-size:0.72rem;color:var(--text-dim);margin-right:4px;">Model:</span>
        <button class="mode-btn{' active' if _cfg_power_mode == 'stryd' else ''}" onclick="setMode('stryd')" id="mode-stryd">⚡ Stryd</button>
        <button class="mode-btn{' active' if _cfg_power_mode == 'gap' else ''}" onclick="setMode('gap')" id="mode-gap">🏃 GAP</button>
        <button class="mode-btn{' active' if _cfg_power_mode == 'sim' else ''}" onclick="setMode('sim')" id="mode-sim">🔬 Sim</button>
    </div>
    <style>
        .mode-btn {{ font-family:'DM Sans',sans-serif; font-size:0.75rem; padding:5px 12px; border-radius:16px;
            border:1px solid var(--border); background:var(--surface); color:var(--text-dim); cursor:pointer; transition:all 0.15s; }}
        .mode-btn:hover {{ border-color:var(--accent); color:var(--text); }}
        .mode-btn.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
        body.gap-mode .power-only {{ display:none !important; }}
        body.sim-mode .stryd-only {{ display:none !important; }}
        body.gap-mode .stryd-only {{ display:none !important; }}
        body.gap-mode .pace-target {{ display:block !important; }}
        .gap-only {{ display:none !important; }}
        body.gap-mode .gap-only {{ display:inline-block !important; }}
    </style>
    
    <!-- v51: Health Check Banner -->
    {_generate_alert_banner(alert_data, critical_power=stats.get('critical_power'))}
    
    <!-- Stats Cards -->
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value" id="ctl-value">{stats['ctl'] if stats['ctl'] else '-'}</div>
            <div class="stat-label">CTL</div>
            <div class="stat-sub">fitness</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="atl-value">{stats['atl'] if stats['atl'] else '-'}</div>
            <div class="stat-label">ATL</div>
            <div class="stat-sub">fatigue</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="tsb-value">{stats['tsb'] if stats['tsb'] else '-'}</div>
            <div class="stat-label">TSB</div>
            <div class="stat-sub">form</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{f"{stats['weight']}kg" if stats['weight'] else '-'}</div>
            <div class="stat-label">Weight</div>
            <div class="stat-sub">7d average</div>
        </div>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value" id="rfl-value">{_init_rfl}%</div>
            <div class="stat-label" id="rfl-label">{_init_rfl_label}</div>
            <div class="stat-sub">vs peak</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="rfl-delta">{f"{'+' if _init_delta > 0 else ''}{_init_delta}%" if _init_delta is not None else '-'}</div>
            <div class="stat-label">RFL 14d</div>
            <div class="stat-sub">change</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="easy-rf-gap-value"
                 data-stryd="{f"{'+' if stats['easy_rfl_gap'] > 0 else ''}{stats['easy_rfl_gap']}%" if stats['easy_rfl_gap'] is not None else '-'}"
                 data-gap="{f"{'+' if stats['easy_rfl_gap_gap'] > 0 else ''}{stats['easy_rfl_gap_gap']}%" if stats['easy_rfl_gap_gap'] is not None else '-'}"
                 data-sim="{f"{'+' if stats['easy_rfl_gap_sim'] > 0 else ''}{stats['easy_rfl_gap_sim']}%" if stats['easy_rfl_gap_sim'] is not None else '-'}"
            >{f"{'+' if _init_easy > 0 else ''}{_init_easy}%" if _init_easy is not None else '-'}</div>
            <div class="stat-label">Easy RF Gap</div>
            <div class="stat-sub">vs trend</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="ag-value">{_init_ag if _init_ag else '-'}%</div>
            <div class="stat-label">Age Grade</div>
            <div class="stat-sub">5k estimate</div>
        </div>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value" id="pred-5k">{format_race_time(_init_preds.get('5k', '-'))}</div>
            <div class="stat-label">5k</div>
            <div class="stat-sub">predicted</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="pred-10k">{format_race_time(_init_preds.get('10k', '-'))}</div>
            <div class="stat-label">10k</div>
            <div class="stat-sub">predicted</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="pred-hm">{format_race_time(_init_preds.get('Half Marathon', '-'))}</div>
            <div class="stat-label">Half</div>
            <div class="stat-sub">predicted</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="pred-mara">{format_race_time(_init_preds.get('Marathon', '-'))}</div>
            <div class="stat-label">Marathon</div>
            <div class="stat-sub">predicted</div>
        </div>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{stats['week_km']}</div>
            <div class="stat-label">Last 7 Days</div>
            <div class="stat-sub">km</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['month_km']}</div>
            <div class="stat-label">Last 30 Days</div>
            <div class="stat-sub">km</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{int(stats['year_km']):,}</div>
            <div class="stat-label">Last 12 Months</div>
            <div class="stat-sub">km</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['total_runs']:,}</div>
            <div class="stat-label">Total Runs</div>
            <div class="stat-sub">since {stats['first_year']}</div>
        </div>
    </div>
    
    <!-- RF Trend Chart -->
    
    <!-- Training Zones Section -->
    {_generate_zone_html(zone_data, stats)}
    
    <!-- RF Trend Chart -->
    <div class="chart-container">
        <div class="chart-header">
            <div class="chart-title">📈 Relative Fitness Level</div>
            <div class="chart-toggle" id="rfToggle">
                <button data-range="90">90d</button>
                <button data-range="180">6m</button>
                <button data-range="365">1yr</button>
                <button data-range="730">2yr</button>
                <button data-range="1095">3yr</button>
                <button data-range="1825">5yr</button>
                <button class="active" data-range="all">All</button>
            </div>
        </div>
        <div class="chart-desc">Blue dots = training, red dots = races, green dashed = easy-run signal.</div>
        <div class="chart-wrapper">
            <canvas id="rfChart"></canvas>
        </div>
    </div>
    
    <!-- RFL 14-day Trend Chart -->
    <div class="chart-container">
        <div class="chart-title">📈 Relative Fitness Level (14 days)</div>
        <div class="chart-desc">Current fitness as % of personal best. Trendline projects 7 days ahead with 95% confidence interval.</div>
        <div class="chart-wrapper" style="height: 180px;">
            <canvas id="rflTrendChart"></canvas>
        </div>
    </div>
    
    <!-- CTL/ATL Chart -->
    <div class="chart-container">
        <div class="chart-header">
            <div class="chart-title">💪 Training Load</div>
            <div class="chart-toggle" id="ctlAtlToggle">
                <button class="active" data-range="90">90d</button>
                <button data-range="180">6m</button>
                <button data-range="365">1yr</button>
                <button data-range="730">2yr</button>
                <button data-range="1095">3yr</button>
                <button data-range="1825">5yr</button>
            </div>
        </div>
        <div class="chart-desc">CTL (blue) = fitness. ATL (red) = fatigue. Dashed lines show 14-day projection if no training.</div>
        <div class="chart-wrapper">
            <canvas id="ctlAtlChart"></canvas>
        </div>
    </div>
    
    <!-- Volume Chart -->
    <div class="chart-container">
        <div class="chart-header">
            <div class="chart-title">📅 Volume</div>
            <div class="chart-toggle" id="volumeGranularityToggle">
                <button class="active" data-granularity="W">W</button>
                <button data-granularity="M">M</button>
                <button data-granularity="Y">Y</button>
            </div>
            <div class="chart-toggle" id="volumeRangeToggle">
                <button class="active" data-range="12">12w</button>
                <button data-range="26">6m</button>
                <button data-range="52">1yr</button>
            </div>
        </div>
        <div class="chart-desc">Distance over time. W=weeks, M=months, Y=years. Consistency matters more than big weeks.</div>
        <div class="chart-wrapper">
            <canvas id="volumeChart"></canvas>
        </div>
    </div>
    
    <!-- v51: Weight Chart (hidden if no weight data) -->
    {''.join(['''
    <div class="chart-container">
        <div class="chart-header">
            <div class="chart-title">&#9878;&#65039; Weight</div>
            <div class="chart-toggle" id="weightToggle">
                <button class="active" data-months="6">6m</button>
                <button data-months="12">1yr</button>
                <button data-months="24">2yr</button>
                <button data-months="36">3yr</button>
                <button data-months="60">5yr</button>
                <button data-months="999">All</button>
            </div>
        </div>
        <div class="chart-desc">Weekly average weight (kg) from daily measurements.</div>
        <div class="chart-wrapper" style="height: 180px;">
            <canvas id="weightChart"></canvas>
        </div>
    </div>
    ''']) if weight_data and weight_data[0] else ''}
    
    <!-- v51: Race Prediction Trend Chart (hidden if no race data) -->
    {''.join(['''
    <div class="chart-container">
        <div class="chart-header">
            <div class="chart-title">🎯 Race Predictions</div>
            <div class="chart-toggle" id="predToggle">
                <button class="active" data-dist="5k">5k</button>
                <button data-dist="10k">10k</button>
                <button data-dist="hm">Half</button>
                <button data-dist="marathon">Marathon</button>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="chart-desc">Solid = predicted. Dashed = adjusted for conditions. <span style="color:#ef4444">●</span> race <span style="color:#fca5a5">●</span> parkrun.</div>
            <div style="display: flex; gap: 10px; align-items: center;">
                <label style="font-size: 0.75em; color: var(--text-dim); white-space: nowrap; cursor: pointer;">
                    <input type="checkbox" id="predConditionsToggle" style="margin-right: 3px;">adjust for conditions
                </label>
                <label style="font-size: 0.75em; color: var(--text-dim); white-space: nowrap; cursor: pointer;">
                    <input type="checkbox" id="predParkrunToggle" style="margin-right: 3px;">parkruns
                </label>
            </div>
        </div>
        <div class="chart-wrapper" style="height: 220px;">
            <canvas id="predChart"></canvas>
        </div>
    </div>
    ''']) if prediction_data and any(prediction_data.get(k, {}).get('dates') for k in prediction_data) else ''}
    
    <!-- v51: Age Grade Trend Chart (hidden if no age grade data) -->
    {''.join(["""
    <div class="chart-container">
        <div class="chart-header">
            <div class="chart-title">📊 Age Grade</div>
            <div class="chart-toggle" id="agToggle">
                <button class="active" data-range="365">1yr</button>
                <button data-range="730">2yr</button>
                <button data-range="1825">5yr</button>
                <button data-range="99999">All</button>
            </div>
        </div>
        <div class="chart-desc">Dot size = distance. Dark = race, light = parkrun. Red line = trend.</div>
        <div class="chart-wrapper" style="height: 220px;">
            <canvas id="agChart"></canvas>
        </div>
    </div>
    """]) if ag_data and ag_data.get('dates') else ''}
    
    <!-- Recent Runs Table -->
    <div class="chart-container">
        <div class="chart-title-row">
            <span class="chart-title">🏃 Recent Runs</span>
            <div class="chart-toggle" id="recentRunsToggle">
                <button class="active" data-filter="all">All</button>
                <button data-filter="races">Races</button>
            </div>
        </div>
        <div class="table-wrapper">
        <table id="recentRunsTable">
            <colgroup>
                <col style="width: 13%;">
                <col style="width: 28%;">
                <col style="width: 11%;">
                <col style="width: 11%;">
                <col class="power-only" style="width: 10%;">
                <col style="width: 9%;">
                <col style="width: 9%;">
                <col style="width: 9%;">
            </colgroup>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Activity</th>
                    <th>Dist</th>
                    <th>Pace</th>
                    <th class="power-only">nPwr</th>
                    <th>HR</th>
                    <th>TSS</th>
                    <th>RFL%</th>
                </tr>
            </thead>
            <tbody id="recentRunsBody">
            </tbody>
        </table>
        </div>
    </div>
    
    <!-- Top Races Section (hidden if no race data) -->
    {''.join(["""
    <div class="chart-container">
        <div class="chart-title-row">
            <span class="chart-title">🏆 Top Race Performances</span>
            <div class="chart-toggle" id="topRacesToggle">
                <button data-period="1y">1Y</button>
                <button data-period="2y">2Y</button>
                <button data-period="3y">3Y</button>
                <button data-period="5y">5Y</button>
                <button class="active" data-period="all">All</button>
            </div>
            <div class="chart-toggle" id="topRacesDistToggle" style="margin-left:12px;">
                <button class="active" data-dist="all">All</button>
                <button data-dist="short">≤5K</button>
                <button data-dist="long">>5K</button>
            </div>
        </div>
        <div class="chart-desc" id="top-races-desc">Best races by normalised AG% (adjusted for temperature, terrain, surface, distance).</div>
        <div class="table-wrapper">
        <table id="topRacesTable">
            <colgroup>
                <col style="width: 12%;">
                <col style="width: 32%;">
                <col style="width: 10%;">
                <col style="width: 14%;">
                <col style="width: 8%;">
                <col style="width: 10%;">
                <col style="width: 14%;">
            </colgroup>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Race</th>
                    <th>Dist</th>
                    <th>Time</th>
                    <th>HR</th>
                    <th>TSS</th>
                    <th>nAG%</th>
                </tr>
            </thead>
            <tbody id="topRacesBody">
                <!-- Populated by JavaScript -->
            </tbody>
        </table>
        </div>
    </div>
    """]) if top_races and any(top_races.get(k) for k in top_races) else ''}
    
    <div class="footer">
        Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}<br>
        Data through: {stats['data_date']}
    </div>
    
    <script>
        // RFL 14-day Trend Chart with trendline, 7-day projection and 95% CI
        const rflTrendCtx = document.getElementById('rflTrendChart').getContext('2d');
        const rfl14Data = {{
            stryd: {{ values: {json.dumps(rfl_trend_values)}, trendline: {json.dumps(rfl_trendline)}, projection: {json.dumps(rfl_projection)}, ci_upper: {json.dumps(rfl_ci_upper)}, ci_lower: {json.dumps(rfl_ci_lower)} }},
            gap: {{ values: {json.dumps(rfl_trend_gap['values'] if rfl_trend_gap else rfl_trend_values)}, trendline: {json.dumps(rfl_trend_gap['trendline'] if rfl_trend_gap else rfl_trendline)}, projection: {json.dumps(rfl_trend_gap['projection'] if rfl_trend_gap else rfl_projection)}, ci_upper: {json.dumps(rfl_trend_gap['ci_upper'] if rfl_trend_gap else rfl_ci_upper)}, ci_lower: {json.dumps(rfl_trend_gap['ci_lower'] if rfl_trend_gap else rfl_ci_lower)} }},
            sim: {{ values: {json.dumps(rfl_trend_sim['values'] if rfl_trend_sim else rfl_trend_values)}, trendline: {json.dumps(rfl_trend_sim['trendline'] if rfl_trend_sim else rfl_trendline)}, projection: {json.dumps(rfl_trend_sim['projection'] if rfl_trend_sim else rfl_projection)}, ci_upper: {json.dumps(rfl_trend_sim['ci_upper'] if rfl_trend_sim else rfl_ci_upper)}, ci_lower: {json.dumps(rfl_trend_sim['ci_lower'] if rfl_trend_sim else rfl_ci_lower)} }}
        }};
        let rfl14Chart = new Chart(rflTrendCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(rfl_trend_dates)},
                datasets: [{{
                    label: 'RFL vs peak',
                    data: {json.dumps(rfl_trend_values)},
                    borderColor: 'rgba(129, 140, 248, 1)',
                    backgroundColor: 'rgba(129, 140, 248, 0.1)',
                    borderWidth: 2,
                    pointRadius: 3,
                    pointBackgroundColor: 'rgba(129, 140, 248, 1)',
                    fill: false,
                    tension: 0,
                    order: 1,
                }}, {{
                    label: 'Trendline',
                    data: {json.dumps(rfl_trendline)},
                    borderColor: 'rgba(100, 100, 100, 0.6)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                    tension: 0,
                    order: 2,
                }}, {{
                    label: 'Projection',
                    data: {json.dumps(rfl_projection)},
                    borderColor: 'rgba(239, 68, 68, 0.9)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0,
                    order: 3,
                }}, {{
                    label: '95% CI upper',
                    data: {json.dumps(rfl_ci_upper)},
                    borderColor: 'rgba(239, 68, 68, 0.5)',
                    borderWidth: 1,
                    borderDash: [2, 2],
                    pointRadius: 0,
                    fill: false,
                    tension: 0,
                    order: 4,
                }}, {{
                    label: '95% CI lower',
                    data: {json.dumps(rfl_ci_lower)},
                    borderColor: 'rgba(239, 68, 68, 0.5)',
                    borderWidth: 1,
                    borderDash: [2, 2],
                    pointRadius: 0,
                    fill: '-1',
                    backgroundColor: 'rgba(239, 68, 68, 0.12)',
                    tension: 0,
                    order: 5,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'bottom',
                        labels: {{
                            boxWidth: 10,
                            usePointStyle: true,
                            padding: 4,
                            font: {{ size: 10 }},
                            filter: function(item) {{
                                // Hide CI labels from legend
                                return !item.text.includes('CI');
                            }}
                        }}
                    }},
                    annotation: {{ annotations: raceAnnotations({json.dumps(rfl_trend_dates)}) }}
                }},
                scales: {{
                    x: {{
                        display: true,
                        ticks: {{
                            maxTicksLimit: 7,
                            font: {{ size: 10 }}
                        }}
                    }},
                    y: {{
                        display: true,
                        grace: '10%',
                        ticks: {{
                            font: {{ size: 10 }},
                            callback: function(value) {{ return value + '%'; }}
                        }},
                        suggestedMin: Math.min(...{json.dumps(rfl_trend_values)}.filter(v => v !== null)) - 5,
                        suggestedMax: Math.max(...{json.dumps(rfl_trend_values)}.filter(v => v !== null)) + 5
                    }}
                }}
            }}
        }});
        
        function updateRfl14Chart(mode) {{
            const d = rfl14Data[mode] || rfl14Data.stryd;
            const modeColors = {{
                stryd: {{ main: 'rgba(129, 140, 248, 1)', bg: 'rgba(129, 140, 248, 0.1)', point: 'rgba(129, 140, 248, 1)' }},
                gap: {{ main: 'rgba(74, 222, 128, 1)', bg: 'rgba(74, 222, 128, 0.1)', point: 'rgba(74, 222, 128, 1)' }},
                sim: {{ main: 'rgba(249, 115, 22, 1)', bg: 'rgba(249, 115, 22, 0.1)', point: 'rgba(249, 115, 22, 1)' }}
            }};
            const mc = modeColors[mode] || modeColors.stryd;
            rfl14Chart.data.datasets[0].data = d.values;
            rfl14Chart.data.datasets[0].borderColor = mc.main;
            rfl14Chart.data.datasets[0].backgroundColor = mc.bg;
            rfl14Chart.data.datasets[0].pointBackgroundColor = mc.point;
            rfl14Chart.data.datasets[1].data = d.trendline;
            rfl14Chart.data.datasets[2].data = d.projection;
            rfl14Chart.data.datasets[3].data = d.ci_upper;
            rfl14Chart.data.datasets[4].data = d.ci_lower;
            // Update y-axis range
            const vals = d.values.filter(v => v !== null);
            if (vals.length > 0) {{
                rfl14Chart.options.scales.y.suggestedMin = Math.min(...vals) - 5;
                rfl14Chart.options.scales.y.suggestedMax = Math.max(...vals) + 5;
            }}
            rfl14Chart.update();
        }}

        // RF Trend Chart - with toggle (v51: includes Easy RF EMA + race dots)
        const rfData = {{
            '90': {{ dates: {json.dumps(rf_dates_90)}, values: {json.dumps(rf_values_90)}, trend: {json.dumps(rf_trend_90)}, easy: {json.dumps(rf_easy_90)}, races: {json.dumps(rf_races_90)}, gap_trend: {json.dumps(rf_gap_trend_90)}, sim_trend: {json.dumps(rf_sim_trend_90)}, gap_values: {json.dumps(rf_gap_values_90)}, sim_values: {json.dumps(rf_sim_values_90)} }},
            '180': {{ dates: {json.dumps(rf_dates_180)}, values: {json.dumps(rf_values_180)}, trend: {json.dumps(rf_trend_180)}, easy: {json.dumps(rf_easy_180)}, races: {json.dumps(rf_races_180)}, gap_trend: {json.dumps(rf_gap_trend_180)}, sim_trend: {json.dumps(rf_sim_trend_180)}, gap_values: {json.dumps(rf_gap_values_180)}, sim_values: {json.dumps(rf_sim_values_180)} }},
            '365': {{ dates: {json.dumps(rf_dates_365)}, values: {json.dumps(rf_values_365)}, trend: {json.dumps(rf_trend_365)}, easy: {json.dumps(rf_easy_365)}, races: {json.dumps(rf_races_365)}, gap_trend: {json.dumps(rf_gap_trend_365)}, sim_trend: {json.dumps(rf_sim_trend_365)}, gap_values: {json.dumps(rf_gap_values_365)}, sim_values: {json.dumps(rf_sim_values_365)} }},
            '730': {{ dates: {json.dumps(rf_dates_730)}, values: {json.dumps(rf_values_730)}, trend: {json.dumps(rf_trend_730)}, easy: {json.dumps(rf_easy_730)}, races: {json.dumps(rf_races_730)}, gap_trend: {json.dumps(rf_gap_trend_730)}, sim_trend: {json.dumps(rf_sim_trend_730)}, gap_values: {json.dumps(rf_gap_values_730)}, sim_values: {json.dumps(rf_sim_values_730)} }},
            '1095': {{ dates: {json.dumps(rf_dates_1095)}, values: {json.dumps(rf_values_1095)}, trend: {json.dumps(rf_trend_1095)}, easy: {json.dumps(rf_easy_1095)}, races: {json.dumps(rf_races_1095)}, gap_trend: {json.dumps(rf_gap_trend_1095)}, sim_trend: {json.dumps(rf_sim_trend_1095)}, gap_values: {json.dumps(rf_gap_values_1095)}, sim_values: {json.dumps(rf_sim_values_1095)} }},
            '1825': {{ dates: {json.dumps(rf_dates_1825)}, values: {json.dumps(rf_values_1825)}, trend: {json.dumps(rf_trend_1825)}, easy: {json.dumps(rf_easy_1825)}, races: {json.dumps(rf_races_1825)}, gap_trend: {json.dumps(rf_gap_trend_1825)}, sim_trend: {json.dumps(rf_sim_trend_1825)}, gap_values: {json.dumps(rf_gap_values_1825)}, sim_values: {json.dumps(rf_sim_values_1825)} }},
            'all': {{ dates: {json.dumps(rf_dates_all)}, values: {json.dumps(rf_values_all)}, trend: {json.dumps(rf_trend_all)}, easy: {json.dumps(rf_easy_all)}, races: {json.dumps(rf_races_all)}, gap_trend: {json.dumps(rf_gap_trend_all)}, sim_trend: {json.dumps(rf_sim_trend_all)}, gap_values: {json.dumps(rf_gap_values_all)}, sim_values: {json.dumps(rf_sim_values_all)} }}
        }};
        
        // Phase 2: Mode state (must be declared before functions that reference it)
        let currentMode = '{_cfg_power_mode if _cfg_power_mode in ("stryd", "gap", "sim") else "stryd"}';
        
        // Phase 2: Mode data for stats switching
        const modeStats = {{
            stryd: {{ rfl: '{stats["latest_rfl"]}', ag: '{stats["age_grade"] or "-"}',
                rflDelta: '{f"{chr(43) if stats['rfl_14d_delta'] > 0 else ''}{stats['rfl_14d_delta']}%" if stats["rfl_14d_delta"] is not None else "-"}',
                cp: {zone_data['current_cp'] if zone_data else (round(PEAK_CP_WATTS_DASH * float(stats["latest_rfl"]) / 100) if stats["latest_rfl"] != "-" else 0)},
                pred5k: '{format_race_time(stats["race_predictions"].get("5k", "-"))}',
                pred10k: '{format_race_time(stats["race_predictions"].get("10k", "-"))}',
                predHm: '{format_race_time(stats["race_predictions"].get("Half Marathon", "-"))}',
                predMara: '{format_race_time(stats["race_predictions"].get("Marathon", "-"))}',
                pred5k_s: {stats["race_predictions"].get("5k_raw", 0)},
                pred10k_s: {stats["race_predictions"].get("10k_raw", 0)},
                predHm_s: {stats["race_predictions"].get("hm_raw", 0)},
                predMara_s: {stats["race_predictions"].get("marathon_raw", 0)} }},
            gap: {{ rfl: '{stats.get("latest_rfl_gap") or stats.get("latest_rfl", "-")}', ag: '{stats["race_predictions"].get("_ag_gap") or stats.get("age_grade") or "-"}',
                rflDelta: '{_fmt_delta(stats.get("rfl_14d_delta_gap") if stats.get("rfl_14d_delta_gap") is not None else stats.get("rfl_14d_delta"))}',
                cp: {round(PEAK_CP_WATTS_DASH * _gap_rfl_num / 100) if _gap_rfl_num else 0},
                pred5k: '{format_race_time(_gap_preds.get("5k", "-"))}',
                pred10k: '{format_race_time(_gap_preds.get("10k", "-"))}',
                predHm: '{format_race_time(_gap_preds.get("Half Marathon", "-"))}',
                predMara: '{format_race_time(_gap_preds.get("Marathon", "-"))}',
                pred5k_s: {_gap_preds.get("5k_raw", 0)},
                pred10k_s: {_gap_preds.get("10k_raw", 0)},
                predHm_s: {_gap_preds.get("hm_raw", 0)},
                predMara_s: {_gap_preds.get("marathon_raw", 0)} }},
            sim: {{ rfl: '{stats.get("latest_rfl_sim") or stats.get("latest_rfl", "-")}', ag: '{stats["race_predictions"].get("_ag_sim") or stats.get("age_grade") or "-"}',
                rflDelta: '{_fmt_delta(stats.get("rfl_14d_delta_sim") if stats.get("rfl_14d_delta_sim") is not None else stats.get("rfl_14d_delta"))}',
                cp: {round(PEAK_CP_WATTS_DASH * float(_sim_rfl_num) / 100) if _sim_rfl_num else 0},
                pred5k: '{format_race_time(_sim_preds.get("5k", "-"))}',
                pred10k: '{format_race_time(_sim_preds.get("10k", "-"))}',
                predHm: '{format_race_time(_sim_preds.get("Half Marathon", "-"))}',
                predMara: '{format_race_time(_sim_preds.get("Marathon", "-"))}',
                pred5k_s: {_sim_preds.get("5k_raw", 0)},
                pred10k_s: {_sim_preds.get("10k_raw", 0)},
                predHm_s: {_sim_preds.get("hm_raw", 0)},
                predMara_s: {_sim_preds.get("marathon_raw", 0)} }}
        }};
        
        // Re-render alert banner now that modeStats is available (for CP display)
        if (typeof renderAlertBanner === 'function') renderAlertBanner(currentMode);
        
        // v51: Generate per-point colours (red for races, blue for training)
        function racePointColors(races, baseColor, raceColor) {{
            return races.map(r => r === 1 ? raceColor : baseColor);
        }}
        
        const rfCtx = document.getElementById('rfChart').getContext('2d');
        let rfChart = new Chart(rfCtx, {{
            type: 'line',
            data: {{
                labels: rfData['all'].dates,
                datasets: [{{
                    label: 'RFL',
                    data: rfData['all'].values,
                    borderColor: 'rgba(129, 140, 248, 0.1)',
                    backgroundColor: 'rgba(129, 140, 248, 0.1)',
                    borderWidth: 1,
                    pointRadius: 0,
                    pointBackgroundColor: 'rgba(129, 140, 248, 0.15)',
                    pointBorderColor: 'rgba(129, 140, 248, 0.1)',
                    pointBorderWidth: 1,
                    fill: false,
                    tension: 0.2,
                }}, {{
                    label: 'RFL Trend',
                    data: rfData['all'].trend,
                    borderColor: 'rgba(129, 140, 248, 1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3,
                }}, {{
                    label: 'Easy RF',
                    data: rfData['all'].easy,
                    borderColor: 'rgba(34, 197, 94, 0.8)',
                    borderWidth: 1.5,
                    borderDash: [4, 3],
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{ intersect: false, mode: 'index' }},
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'bottom',
                        labels: {{ 
                            boxWidth: 12, padding: 8, font: {{ size: 10 }}, usePointStyle: true,
                            generateLabels: function(chart) {{
                                const mc = typeof currentMode !== 'undefined' ? currentMode : 'stryd';
                                const colors = {{
                                    stryd: {{ fill: 'rgba(129, 140, 248, ', trend: '#818cf8' }},
                                    gap: {{ fill: 'rgba(74, 222, 128, ', trend: '#4ade80' }},
                                    sim: {{ fill: 'rgba(249, 115, 22, ', trend: '#f97316' }}
                                }};
                                const c = colors[mc] || colors.stryd;
                                const modeLabel = mc === 'gap' ? 'RFL (GAP)' : mc === 'sim' ? 'RFL (Sim)' : 'RFL';
                                const trendLabel = mc === 'gap' ? 'GAP Trend' : mc === 'sim' ? 'Sim Trend' : 'RFL Trend';
                                const items = [
                                    {{ text: modeLabel, fillStyle: c.fill + '0.4)', strokeStyle: c.fill + '0.3)', pointStyle: 'circle', hidden: !chart.isDatasetVisible(0), datasetIndex: 0, fontColor: '#8b90a0' }},
                                    {{ text: trendLabel, fillStyle: c.trend, strokeStyle: c.trend, pointStyle: 'circle', hidden: !chart.isDatasetVisible(1), datasetIndex: 1, fontColor: '#8b90a0' }}
                                ];
                                if (mc === 'stryd') {{
                                    items.push({{ text: 'Easy RF', fillStyle: 'rgba(34, 197, 94, 0.8)', strokeStyle: 'rgba(34, 197, 94, 0.8)', pointStyle: 'circle', hidden: !chart.isDatasetVisible(2), datasetIndex: 2, fontColor: '#8b90a0' }});
                                }}
                                return items;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{ display: true, ticks: {{ maxTicksLimit: 12, maxRotation: 0, font: {{ size: 10 }} }} }},
                    y: {{ 
                        display: true, 
                        ticks: {{ 
                            font: {{ size: 10 }},
                            callback: function(value) {{ return value + '%'; }}
                        }},
                        min: 50,
                        suggestedMax: 100
                    }}
                }}
            }}
        }});
        
        let currentRfRange = 'all';
        
        function updateRfChart(range) {{
            currentRfRange = range;
            const d = rfData[range];
            // Mode-dependent data selection
            const valKey = currentMode === 'gap' ? 'gap_values' : currentMode === 'sim' ? 'sim_values' : 'values';
            const trendKey = currentMode === 'gap' ? 'gap_trend' : currentMode === 'sim' ? 'sim_trend' : 'trend';
            rfChart.data.labels = d.dates;
            rfChart.data.datasets[0].data = d[valKey] || d.values;
            rfChart.data.datasets[1].data = d[trendKey] || d.trend;
            rfChart.data.datasets[2].data = currentMode === 'stryd' ? d.easy : d.easy.map(() => null);
            
            // Adjust display based on range
            const pointSettings = {{
                '90': {{ radius: 3, borderColor: 'rgba(129, 140, 248, 0.3)', pointColor: 'rgba(129, 140, 248, 0.5)', fill: false, tension: 0, lineWidth: 1, yMin: 70, ticks: 5 }},
                '180': {{ radius: 2, borderColor: 'rgba(129, 140, 248, 0.25)', pointColor: 'rgba(129, 140, 248, 0.4)', fill: false, tension: 0, lineWidth: 1, yMin: 70, ticks: 5 }},
                '365': {{ radius: 1.5, borderColor: 'rgba(129, 140, 248, 0.2)', pointColor: 'rgba(129, 140, 248, 0.3)', fill: false, tension: 0, lineWidth: 1, yMin: 70, ticks: 8 }},
                '730': {{ radius: 1, borderColor: 'rgba(129, 140, 248, 0.15)', pointColor: 'rgba(129, 140, 248, 0.25)', fill: false, tension: 0.1, lineWidth: 1, yMin: 60, ticks: 10 }},
                '1095': {{ radius: 0.5, borderColor: 'rgba(129, 140, 248, 0.1)', pointColor: 'rgba(129, 140, 248, 0.2)', fill: false, tension: 0.1, lineWidth: 1, yMin: 50, ticks: 10 }},
                '1825': {{ radius: 0, borderColor: 'rgba(129, 140, 248, 0.1)', pointColor: 'rgba(129, 140, 248, 0.15)', fill: false, tension: 0.2, lineWidth: 1, yMin: 50, ticks: 10 }},
                'all': {{ radius: 0, borderColor: 'rgba(129, 140, 248, 0.1)', pointColor: 'rgba(129, 140, 248, 0.15)', fill: false, tension: 0.2, lineWidth: 1, yMin: 50, ticks: 12 }}
            }};
            const settings = pointSettings[range];
            const races = d.races;
            // Mode colors
            const modeColors = {{
                stryd: {{ base: 'rgba(129, 140, 248, ', trend: '#818cf8' }},
                gap: {{ base: 'rgba(74, 222, 128, ', trend: '#4ade80' }},
                sim: {{ base: 'rgba(249, 115, 22, ', trend: '#f97316' }}
            }};
            const mc = modeColors[currentMode];
            rfChart.data.datasets[0].pointRadius = settings.radius;
            rfChart.data.datasets[0].pointBackgroundColor = races.length ? racePointColors(races, mc.base + '0.5)', 'rgba(239, 68, 68, 0.9)') : mc.base + '0.5)';
            rfChart.data.datasets[0].pointBorderColor = races.length ? racePointColors(races, mc.base + '0.3)', 'rgba(239, 68, 68, 0.7)') : mc.base + '0.3)';
            rfChart.data.datasets[0].borderColor = mc.base + '0.15)';
            rfChart.data.datasets[0].borderWidth = settings.lineWidth;
            rfChart.data.datasets[0].fill = settings.fill;
            rfChart.data.datasets[0].tension = settings.tension;
            rfChart.data.datasets[1].borderColor = mc.trend;
            rfChart.options.scales.y.min = settings.yMin;
            rfChart.options.scales.x.ticks.maxTicksLimit = settings.ticks;
            
            rfChart.update();
        }}
        
        document.getElementById('rfToggle').addEventListener('click', function(e) {{
            if (e.target.tagName === 'BUTTON') {{
                const range = e.target.dataset.range;
                this.querySelectorAll('button').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                updateRfChart(range);
            }}
        }});
        
        // CTL/ATL Training Load Chart - with toggle
        const ctlAtlData = {{
            '90': {{ dates: {json.dumps(ctl_atl_dates_90)}, ctl: {json.dumps(ctl_values_90)}, atl: {json.dumps(atl_values_90)}, ctlProj: {json.dumps(ctl_proj_90)}, atlProj: {json.dumps(atl_proj_90)} }},
            '180': {{ dates: {json.dumps(ctl_atl_dates_180)}, ctl: {json.dumps(ctl_values_180)}, atl: {json.dumps(atl_values_180)}, ctlProj: {json.dumps(ctl_proj_180)}, atlProj: {json.dumps(atl_proj_180)} }},
            '365': {{ dates: {json.dumps(ctl_atl_dates_365)}, ctl: {json.dumps(ctl_values_365)}, atl: {json.dumps(atl_values_365)}, ctlProj: {json.dumps(ctl_proj_365)}, atlProj: {json.dumps(atl_proj_365)} }},
            '730': {{ dates: {json.dumps(ctl_atl_dates_730)}, ctl: {json.dumps(ctl_values_730)}, atl: {json.dumps(atl_values_730)}, ctlProj: {json.dumps(ctl_proj_730)}, atlProj: {json.dumps(atl_proj_730)} }},
            '1095': {{ dates: {json.dumps(ctl_atl_dates_1095)}, ctl: {json.dumps(ctl_values_1095)}, atl: {json.dumps(atl_values_1095)}, ctlProj: {json.dumps(ctl_proj_1095)}, atlProj: {json.dumps(atl_proj_1095)} }},
            '1825': {{ dates: {json.dumps(ctl_atl_dates_1825)}, ctl: {json.dumps(ctl_values_1825)}, atl: {json.dumps(atl_values_1825)}, ctlProj: {json.dumps(ctl_proj_1825)}, atlProj: {json.dumps(atl_proj_1825)} }}
        }};
        
        const ctlAtlCtx = document.getElementById('ctlAtlChart').getContext('2d');
        let ctlAtlChart = new Chart(ctlAtlCtx, {{
            type: 'line',
            data: {{
                labels: ctlAtlData['90'].dates,
                datasets: [{{
                    label: 'CTL (Fitness)',
                    data: ctlAtlData['90'].ctl,
                    borderColor: 'rgba(129, 140, 248, 1)',
                    backgroundColor: 'rgba(129, 140, 248, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3,
                }}, {{
                    label: 'ATL (Fatigue)',
                    data: ctlAtlData['90'].atl,
                    borderColor: 'rgba(239, 68, 68, 1)',
                    backgroundColor: 'rgba(239, 68, 68, 0.12)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3,
                }}, {{
                    label: 'CTL Projection',
                    data: ctlAtlData['90'].ctlProj,
                    borderColor: 'rgba(129, 140, 248, 0.5)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3,
                }}, {{
                    label: 'ATL Projection',
                    data: ctlAtlData['90'].atlProj,
                    borderColor: 'rgba(239, 68, 68, 0.5)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{ intersect: false, mode: 'index' }},
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'bottom',
                        labels: {{ 
                            boxWidth: 12, 
                            padding: 8, 
                            font: {{ size: 10 }},
                            usePointStyle: true,
                            filter: function(item) {{
                                // Hide projection labels from legend
                                return !item.text.includes('Projection');
                            }}
                        }}
                    }},
                    annotation: {{ annotations: raceAnnotations(ctlAtlData['90'].dates) }}
                }},
                scales: {{
                    x: {{ display: true, ticks: {{ maxTicksLimit: 5, maxRotation: 0, font: {{ size: 10 }} }} }},
                    y: {{ display: true, ticks: {{ font: {{ size: 10 }} }} }}
                }}
            }}
        }});
        
        document.getElementById('ctlAtlToggle').addEventListener('click', function(e) {{
            if (e.target.tagName === 'BUTTON') {{
                const range = e.target.dataset.range;
                this.querySelectorAll('button').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                ctlAtlChart.data.labels = ctlAtlData[range].dates;
                ctlAtlChart.data.datasets[0].data = ctlAtlData[range].ctl;
                ctlAtlChart.data.datasets[1].data = ctlAtlData[range].atl;
                ctlAtlChart.data.datasets[2].data = ctlAtlData[range].ctlProj;
                ctlAtlChart.data.datasets[3].data = ctlAtlData[range].atlProj;
                ctlAtlChart.options.plugins.annotation = {{ annotations: raceAnnotations(ctlAtlData[range].dates) }};
                ctlAtlChart.update();
            }}
        }});
        
        // Volume Chart - with two-level toggle (granularity + range)
        const volumeData = {{
            'W': {{
                '12': {{ labels: {json.dumps(week_labels_12)}, distances: {json.dumps(week_distances_12)}, runs: {json.dumps(week_runs_12)} }},
                '26': {{ labels: {json.dumps(week_labels_26)}, distances: {json.dumps(week_distances_26)}, runs: {json.dumps(week_runs_26)} }},
                '52': {{ labels: {json.dumps(week_labels_52)}, distances: {json.dumps(week_distances_52)}, runs: {json.dumps(week_runs_52)} }}
            }},
            'M': {{
                '6': {{ labels: {json.dumps(month_labels_6)}, distances: {json.dumps(month_distances_6)}, runs: {json.dumps(month_runs_6)} }},
                '12': {{ labels: {json.dumps(month_labels_12)}, distances: {json.dumps(month_distances_12)}, runs: {json.dumps(month_runs_12)} }},
                '24': {{ labels: {json.dumps(month_labels_24)}, distances: {json.dumps(month_distances_24)}, runs: {json.dumps(month_runs_24)} }},
                'All': {{ labels: {json.dumps(month_labels_all)}, distances: {json.dumps(month_distances_all)}, runs: {json.dumps(month_runs_all)} }}
            }},
            'Y': {{
                '3': {{ labels: {json.dumps(year_labels_3)}, distances: {json.dumps(year_distances_3)}, runs: {json.dumps(year_runs_3)} }},
                '5': {{ labels: {json.dumps(year_labels_5)}, distances: {json.dumps(year_distances_5)}, runs: {json.dumps(year_runs_5)} }},
                'All': {{ labels: {json.dumps(year_labels_all)}, distances: {json.dumps(year_distances_all)}, runs: {json.dumps(year_runs_all)} }}
            }}
        }};
        
        const volumeRangeOptions = {{
            'W': ['12', '26', '52'],
            'M': ['6', '12', '24', 'All'],
            'Y': ['3', '5', 'All']
        }};
        
        let currentGranularity = 'W';
        let currentRange = '12';
        
        const volCtx = document.getElementById('volumeChart').getContext('2d');
        let volChart = new Chart(volCtx, {{
            type: 'bar',
            data: {{
                labels: volumeData['W']['12'].labels,
                datasets: [{{
                    label: 'Distance (km)',
                    data: volumeData['W']['12'].distances,
                    backgroundColor: 'rgba(129, 140, 248, 0.7)',
                    borderRadius: 4,
                }}]
            }},
            plugins: [ChartDataLabels],
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            afterLabel: function(context) {{
                                const runs = volumeData[currentGranularity][currentRange].runs;
                                return runs[context.dataIndex] + ' runs';
                            }}
                        }}
                    }},
                    datalabels: {{
                        display: function(context) {{
                            return context.dataset.data.length <= 20;
                        }},
                        anchor: 'end',
                        align: 'end',
                        offset: -4,
                        color: '#c0c4d0',
                        font: {{ size: 10, weight: '500' }},
                        formatter: function(value) {{
                            return Math.round(value);
                        }}
                    }}
                }},
                scales: {{
                    x: {{ ticks: {{ font: {{ size: 10 }}, maxTicksLimit: 12 }} }},
                    y: {{ beginAtZero: true, ticks: {{ font: {{ size: 10 }} }}, grace: '5%' }}
                }}
            }}
        }});
        
        function updateVolumeRangeButtons() {{
            const rangeToggle = document.getElementById('volumeRangeToggle');
            const options = volumeRangeOptions[currentGranularity];
            rangeToggle.innerHTML = options.map((opt, idx) => 
                `<button ${{idx === 0 ? 'class="active"' : ''}} data-range="${{opt}}">${{opt}}</button>`
            ).join('');
            currentRange = options[0];
        }}
        
        function updateVolumeChart() {{
            const data = volumeData[currentGranularity][currentRange];
            volChart.data.labels = data.labels;
            volChart.data.datasets[0].data = data.distances;
            volChart.update();
        }}
        
        document.getElementById('volumeGranularityToggle').addEventListener('click', function(e) {{
            if (e.target.tagName === 'BUTTON') {{
                currentGranularity = e.target.dataset.granularity;
                this.querySelectorAll('button').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                updateVolumeRangeButtons();
                updateVolumeChart();
            }}
        }});
        
        document.getElementById('volumeRangeToggle').addEventListener('click', function(e) {{
            if (e.target.tagName === 'BUTTON') {{
                currentRange = e.target.dataset.range;
                this.querySelectorAll('button').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                updateVolumeChart();
            }}
        }});
        
        // All-time Weekly RFL Chart
        // CTL/ATL/TSB live update based on current date
        const ctlAtlLookup = {json.dumps(ctl_atl_lookup)};
        
        function updateCtlAtl() {{
            const today = new Date();
            const dateStr = today.toISOString().split('T')[0];
            
            if (ctlAtlLookup[dateStr]) {{
                const data = ctlAtlLookup[dateStr];
                if (data.ctl !== null) document.getElementById('ctl-value').textContent = data.ctl;
                if (data.atl !== null) document.getElementById('atl-value').textContent = data.atl;
                if (data.tsb !== null) document.getElementById('tsb-value').textContent = data.tsb;
            }}
        }}
        
        // Update on page load
        updateCtlAtl();
        
        // Top Races data and toggle
        const topRacesData = {json.dumps(top_races)};
        
        function updateTopRacesTable(period, distFilter) {{
            const tbody = document.getElementById('topRacesBody');
            if (!tbody) return;
            if (!distFilter) distFilter = document.querySelector('#topRacesDistToggle button.active')?.getAttribute('data-dist') || 'all';
            let races = (topRacesData[period] || []).slice();
            
            // Apply distance filter
            if (distFilter === 'short') races = races.filter(r => r.dist_group === 'short');
            else if (distFilter === 'long') races = races.filter(r => r.dist_group === 'long');
            
            if (races.length === 0) {{
                tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-dim);">No races in this period</td></tr>';
                return;
            }}
            
            // Sort by normalised AG descending, take top 10
            races.sort((a, b) => (b.nag || 0) - (a.nag || 0));
            races = races.slice(0, 10);
            
            tbody.innerHTML = races.map((race, idx) => {{
                const nagVal = race.nag ? race.nag + '%' : '-';
                return `
                <tr>
                    <td>${{race.date}}</td>
                    <td>${{race.name}}</td>
                    <td>${{race.dist ? race.dist + ' km' : '-'}}</td>
                    <td>${{race.time}}</td>
                    <td>${{race.hr || '-'}}</td>
                    <td>${{race.tss || '-'}}</td>
                    <td>${{nagVal}}</td>
                </tr>`;
            }}).join('');
        }}
        
        // Initialize with 1 year view
        if (document.getElementById('topRacesBody')) {{
            updateTopRacesTable('all', 'all');
        }}
        
        // Period toggle handler
        const topRacesToggleEl = document.getElementById('topRacesToggle');
        if (topRacesToggleEl) {{
            topRacesToggleEl.addEventListener('click', function(e) {{
            if (e.target.tagName === 'BUTTON') {{
                this.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
                e.target.classList.add('active');
                const period = e.target.getAttribute('data-period');
                updateTopRacesTable(period);
            }}
        }});
        }}
        
        // Distance toggle handler
        const topRacesDistToggleEl = document.getElementById('topRacesDistToggle');
        if (topRacesDistToggleEl) {{
            topRacesDistToggleEl.addEventListener('click', function(e) {{
            if (e.target.tagName === 'BUTTON') {{
                this.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
                e.target.classList.add('active');
                const activePeriod = document.querySelector('#topRacesToggle button.active');
                const period = activePeriod ? activePeriod.getAttribute('data-period') : '1y';
                updateTopRacesTable(period, e.target.getAttribute('data-dist'));
            }}
        }});
        }}
        
        // --- v51: Weight Chart ---
        const weightAllDates = {json.dumps(weight_data[0] if weight_data else [])};
        const weightAllValues = {json.dumps(weight_data[1] if weight_data else [])};
        
        function getWeightSlice(months) {{
            if (months >= 999) return {{ dates: weightAllDates, values: weightAllValues }};
            const n = Math.round(months * 4.33);  // ~weeks per month
            return {{
                dates: weightAllDates.slice(-n),
                values: weightAllValues.slice(-n)
            }};
        }}
        
        const weightCtx = document.getElementById('weightChart');
        let weightChart = null;
        if (weightCtx && weightAllDates.length > 0) {{
            const w6 = getWeightSlice(6);
            weightChart = new Chart(weightCtx.getContext('2d'), {{
                type: 'line',
                data: {{
                    labels: w6.dates,
                    datasets: [{{
                        label: 'Weight (kg)',
                        data: w6.values,
                        borderColor: 'rgba(168, 85, 247, 0.9)',
                        backgroundColor: 'rgba(168, 85, 247, 0.1)',
                        borderWidth: 1.5,
                        pointRadius: 1,
                        fill: true,
                        tension: 0.3,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        x: {{ display: true, ticks: {{ maxTicksLimit: 5, maxRotation: 0, font: {{ size: 10 }} }} }},
                        y: {{ 
                            display: true, 
                            ticks: {{ font: {{ size: 10 }}, callback: function(v) {{ return v + 'kg'; }} }}
                        }}
                    }}
                }}
            }});
            
            document.getElementById('weightToggle').addEventListener('click', function(e) {{
                if (e.target.tagName === 'BUTTON') {{
                    this.querySelectorAll('button').forEach(b => b.classList.remove('active'));
                    e.target.classList.add('active');
                    const months = parseInt(e.target.dataset.months);
                    const wd = getWeightSlice(months);
                    weightChart.data.labels = wd.dates;
                    weightChart.data.datasets[0].data = wd.values;
                    weightChart.update();
                }}
            }});
        }}
        
        // --- v51: Race Prediction Trend Chart ---
        const predData = {json.dumps(prediction_data if prediction_data else {})};
        let currentPredDist = '5k';
        let showParkruns = false;
        let adjustConditions = false;
        
        // Heat scaling constants (match StepB)
        const HEAT_REF_MINS = 90.0;
        const HEAT_MAX_MULT = 1.5;
        
        function formatPredTime(seconds) {{
            if (!seconds) return '-';
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            if (h > 0) return h + ':' + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
            return m + ':' + String(s).padStart(2, '0');
        }}
        
        const predCtx = document.getElementById('predChart');
        let predChart = null;
        
        function renderPredChart(distKey, includeParkruns) {{
            if (!predCtx) return;
            const d = predData[distKey];
            if (!d || !d.dates || d.dates.length === 0) return;
            
            // Filter by parkrun toggle
            const indices = [];
            for (let i = 0; i < d.dates.length; i++) {{
                if (!includeParkruns && d.is_parkrun[i] === 1) continue;
                indices.push(i);
            }}
            
            const dates = indices.map(i => d.dates[i]);
            const datesISO = indices.map(i => d.dates_iso[i]);
            // Mode-dependent prediction source
            const predKey = currentMode === 'gap' ? 'predicted_gap' : currentMode === 'sim' ? 'predicted_sim' : 'predicted';
            const predicted = indices.map(i => (d[predKey] || d.predicted)[i]);
            const actual = indices.map(i => d.actual[i]);
            const names = indices.map(i => d.names[i]);
            const temps = indices.map(i => d.temps[i]);
            const surfaces = indices.map(i => d.surfaces[i]);
            const isParkrun = indices.map(i => d.is_parkrun[i]);
            const tempAdjs = indices.map(i => (d.temp_adjs || [])[i] || 1.0);
            const surfaceAdjs = indices.map(i => (d.surface_adjs || [])[i] || 1.0);
            
            // Apply conditions adjustment to predictions if toggled
            let displayPredicted = predicted;
            if (adjustConditions) {{
                displayPredicted = predicted.map((p, i) => {{
                    if (p === null) return null;
                    // Duration-scaled heat: use predicted time as duration estimate
                    const estMins = p / 60.0;
                    const heatMult = Math.min(HEAT_MAX_MULT, estMins / HEAT_REF_MINS);
                    const tempFactor = 1.0 + (tempAdjs[i] - 1.0) * heatMult;
                    const surfFactor = surfaceAdjs[i];
                    return Math.round(p * tempFactor * surfFactor);
                }});
            }}
            
            // Colour actual dots: parkruns lighter, non-parkrun races darker
            const dotColors = isParkrun.map(p => p === 1 ? 'rgba(252, 165, 165, 0.8)' : 'rgba(239, 68, 68, 0.9)');
            
            // Build data points for time axis
            const predPoints = datesISO.map((dt, i) => displayPredicted[i] !== null ? ({{ x: dt, y: displayPredicted[i] }}) : null).filter(p => p !== null);
            const actualPoints = datesISO.map((dt, i) => ({{ x: dt, y: actual[i] }}));
            
            // Full prediction trend line (weekly smoothed, for distances with few races)
            const trendKey = currentMode === 'gap' ? 'trend_dates_iso_gap' : currentMode === 'sim' ? 'trend_dates_iso_sim' : 'trend_dates_iso';
            const trendValKey = currentMode === 'gap' ? 'trend_values_gap' : currentMode === 'sim' ? 'trend_values_sim' : 'trend_values';
            const trendDates = d[trendKey] || d.trend_dates_iso || [];
            const trendVals = d[trendValKey] || d.trend_values || [];
            const trendPoints = trendDates.map((dt, i) => ({{ x: dt, y: trendVals[i] }}));
            const fewRaces = actualPoints.length < 5;
            
            // Compute stable y-axis range covering both adjusted and unadjusted data
            const allYVals = [];
            for (let i = 0; i < actual.length; i++) {{ if (actual[i]) allYVals.push(actual[i]); }}
            for (let i = 0; i < predicted.length; i++) {{ if (predicted[i]) allYVals.push(predicted[i]); }}
            if (fewRaces) {{ for (let i = 0; i < trendVals.length; i++) {{ if (trendVals[i]) allYVals.push(trendVals[i]); }} }}
            // Also include worst-case adjusted predictions
            for (let i = 0; i < predicted.length; i++) {{
                if (predicted[i] === null) continue;
                const estMins = predicted[i] / 60.0;
                const hm = Math.min(HEAT_MAX_MULT, estMins / HEAT_REF_MINS);
                const tf = 1.0 + (tempAdjs[i] - 1.0) * hm;
                const sf = surfaceAdjs[i];
                allYVals.push(Math.round(predicted[i] * tf * sf));
            }}
            const yDataMin = Math.min(...allYVals);
            const yDataMax = Math.max(...allYVals);
            // Pick a nice round tick interval in seconds
            const ySpan = yDataMax - yDataMin;
            const niceSteps = [15, 30, 60, 120, 300, 600, 900, 1800, 3600];  // 15s to 1h
            let tickStep = 60;
            for (const s of niceSteps) {{
                if (ySpan / s >= 3 && ySpan / s <= 8) {{ tickStep = s; break; }}
            }}
            // Round min down and max up to tick boundaries, then pad one tick
            const yMin = Math.floor(yDataMin / tickStep) * tickStep - tickStep;
            const yMax = Math.ceil(yDataMax / tickStep) * tickStep + tickStep;
            
            if (predChart) predChart.destroy();
            
            predChart = new Chart(predCtx.getContext('2d'), {{
                type: 'line',
                data: {{
                    datasets: [
                    // Background trend line (only for distances with < 5 races)
                    ...(fewRaces && trendPoints.length > 0 ? [{{
                        label: 'Fitness trend',
                        data: trendPoints,
                        borderColor: (function() {{
                            const modeBase = currentMode === 'gap' ? 'rgba(74, 222, 128,' : currentMode === 'sim' ? 'rgba(249, 115, 22,' : 'rgba(129, 140, 248,';
                            return modeBase + ' 0.3)';
                        }})(),
                        backgroundColor: (function() {{
                            const modeBase = currentMode === 'gap' ? 'rgba(74, 222, 128,' : currentMode === 'sim' ? 'rgba(249, 115, 22,' : 'rgba(129, 140, 248,';
                            return modeBase + ' 0.04)';
                        }})(),
                        borderWidth: 1.5,
                        pointRadius: 0,
                        pointHitRadius: 0,
                        fill: true,
                        tension: 0.3,
                        order: 3,
                    }}] : []),
                    {{
                        label: adjustConditions ? 'Predicted (adj)' : 'Predicted',
                        data: predPoints,
                        borderColor: (function() {{
                            const modeBase = currentMode === 'gap' ? 'rgba(74, 222, 128,' : currentMode === 'sim' ? 'rgba(249, 115, 22,' : 'rgba(129, 140, 248,';
                            return adjustConditions ? modeBase + ' 1)' : modeBase + ' 0.7)';
                        }})(),
                        backgroundColor: (function() {{
                            const modeBase = currentMode === 'gap' ? 'rgba(74, 222, 128,' : currentMode === 'sim' ? 'rgba(249, 115, 22,' : 'rgba(129, 140, 248,';
                            return modeBase + ' 0.05)';
                        }})(),
                        borderDash: adjustConditions ? [4, 3] : [],
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: !fewRaces,
                        tension: 0.3,
                    }}, {{
                        label: 'Actual',
                        data: actualPoints,
                        borderColor: 'rgba(239, 68, 68, 0.2)',
                        pointBackgroundColor: dotColors,
                        borderWidth: 1,
                        pointRadius: 3,
                        fill: false,
                        tension: 0,
                        showLine: true,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            callbacks: {{
                                title: function(ctx) {{
                                    // Find original index from ISO date
                                    const isoDate = ctx[0].raw.x;
                                    const i = datesISO.indexOf(isoDate);
                                    if (i < 0) return isoDate;
                                    return dates[i] + (names[i] ? ' — ' + names[i] : '');
                                }},
                                label: function(ctx) {{
                                    return ctx.dataset.label + ': ' + formatPredTime(ctx.raw.y);
                                }},
                                afterBody: function(ctx) {{
                                    const isoDate = ctx[0].raw.x;
                                    const i = datesISO.indexOf(isoDate);
                                    if (i < 0) return [];
                                    const lines = [];
                                    // Gap vs prediction
                                    if (predicted[i] && actual[i]) {{
                                        // If conditions toggled, compare actual vs adjusted prediction
                                        let compPred = predicted[i];
                                        if (adjustConditions) {{
                                            const estMins = predicted[i] / 60.0;
                                            const heatMult = Math.min(HEAT_MAX_MULT, estMins / HEAT_REF_MINS);
                                            const tempFactor = 1.0 + (tempAdjs[i] - 1.0) * heatMult;
                                            const surfFactor = surfaceAdjs[i];
                                            compPred = Math.round(predicted[i] * tempFactor * surfFactor);
                                        }}
                                        const gap = actual[i] - compPred;
                                        if (Math.abs(gap) < 2) {{
                                            lines.push('Hit prediction exactly');
                                        }} else {{
                                            lines.push(formatPredTime(Math.abs(gap)) + (gap > 0 ? ' slower than prediction' : ' faster than prediction'));
                                        }}
                                    }}
                                    // Condition adjustments breakdown
                                    if (adjustConditions && (tempAdjs[i] !== 1.0 || surfaceAdjs[i] !== 1.0)) {{
                                        const parts = [];
                                        if (tempAdjs[i] !== 1.0 && predicted[i]) {{
                                            const estMins = predicted[i] / 60.0;
                                            const heatMult = Math.min(HEAT_MAX_MULT, estMins / HEAT_REF_MINS);
                                            const tempPct = ((tempAdjs[i] - 1.0) * heatMult * 100).toFixed(1);
                                            parts.push('temp +' + tempPct + '%');
                                        }}
                                        if (surfaceAdjs[i] !== 1.0) {{
                                            const surfPct = ((surfaceAdjs[i] - 1.0) * 100).toFixed(1);
                                            parts.push('surface +' + surfPct + '%');
                                        }}
                                        if (parts.length > 0) lines.push('Adj: ' + parts.join(', '));
                                    }}
                                    // Temperature
                                    if (temps[i] !== null) {{
                                        const t = temps[i];
                                        let icon = '';
                                        if (t >= 25) icon = ' 🥵';
                                        else if (t >= 20) icon = ' ☀️';
                                        else if (t <= 0) icon = ' 🥶';
                                        lines.push('Temp: ' + t + '°C' + icon);
                                    }}
                                    // Surface
                                    if (surfaces[i]) {{
                                        const surfaceIcons = {{ 'SNOW': '❄️', 'HEAVY_SNOW': '🌨️', 'TRAIL': '🌲', 'TRACK': '🏟️', 'INDOOR_TRACK': '🏟️' }};
                                        const icon = surfaceIcons[surfaces[i]] || '';
                                        lines.push(icon + ' ' + surfaces[i]);
                                    }}
                                    return lines;
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            type: 'time',
                            time: {{ unit: 'year', displayFormats: {{ year: 'yyyy' }} }},
                            ticks: {{ maxTicksLimit: 10, font: {{ size: 10 }} }}
                        }},
                        y: {{
                            display: true,
                            reverse: true,
                            min: yMin,
                            max: yMax,
                            ticks: {{
                                stepSize: tickStep,
                                callback: function(v) {{ return formatPredTime(v); }},
                                font: {{ size: 10 }}
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        if (predCtx && predData && predData['5k'] && predData['5k'].dates.length > 0) {{
            renderPredChart('5k', showParkruns);
            
            document.getElementById('predToggle').addEventListener('click', function(e) {{
                if (e.target.tagName === 'BUTTON') {{
                    this.querySelectorAll('button').forEach(b => b.classList.remove('active'));
                    e.target.classList.add('active');
                    currentPredDist = e.target.dataset.dist;
                    renderPredChart(currentPredDist, showParkruns);
                }}
            }});
            
            document.getElementById('predParkrunToggle').addEventListener('change', function() {{
                showParkruns = this.checked;
                renderPredChart(currentPredDist, showParkruns);
            }});
            
            document.getElementById('predConditionsToggle').addEventListener('change', function() {{
                adjustConditions = this.checked;
                renderPredChart(currentPredDist, showParkruns);
            }});
        }}
        
        // --- v51: Age Grade Trend Chart ---
        const agAllDates = {json.dumps(ag_data['dates'] if ag_data else [])};
        const agAllDatesISO = {json.dumps(ag_data['dates_iso'] if ag_data else [])};
        const agAllValues = {json.dumps(ag_data['values'] if ag_data else [])};
        const agAllLabels = {json.dumps(ag_data['dist_labels'] if ag_data else [])};
        const agAllColors = {json.dumps(ag_data['dist_codes'] if ag_data else [])};
        const agAllSizes = {json.dumps(ag_data['dist_sizes'] if ag_data else [])};
        const agAllParkrun = {json.dumps(ag_data['is_parkrun'] if ag_data else [])};
        const agAllRolling = {json.dumps(ag_data['rolling_avg'] if ag_data else [])};
        
        const agCtx = document.getElementById('agChart');
        let agChart = null;
        
        function getAgSlice(days) {{
            if (days >= 99999) return {{ datesISO: agAllDatesISO, dates: agAllDates, values: agAllValues, labels: agAllLabels, colors: agAllColors, sizes: agAllSizes, parkrun: agAllParkrun, rolling: agAllRolling }};
            const cutoff = new Date();
            cutoff.setDate(cutoff.getDate() - days);
            const indices = [];
            for (let i = 0; i < agAllDatesISO.length; i++) {{
                if (new Date(agAllDatesISO[i]) >= cutoff) indices.push(i);
            }}
            return {{
                datesISO: indices.map(i => agAllDatesISO[i]),
                dates: indices.map(i => agAllDates[i]),
                values: indices.map(i => agAllValues[i]),
                labels: indices.map(i => agAllLabels[i]),
                colors: indices.map(i => agAllColors[i]),
                sizes: indices.map(i => agAllSizes[i]),
                parkrun: indices.map(i => agAllParkrun[i]),
                rolling: indices.map(i => agAllRolling[i])
            }};
        }}
        
        function renderAgChart(days) {{
            const d = getAgSlice(days);
            if (d.datesISO.length === 0) return;
            
            if (agChart) agChart.destroy();
            
            // Build rolling average as line data (skip nulls)
            const rollingData = d.datesISO.map((dt, i) => d.rolling[i] !== null ? ({{ x: dt, y: d.rolling[i] }}) : null).filter(p => p !== null);
            
            agChart = new Chart(agCtx.getContext('2d'), {{
                type: 'scatter',
                data: {{
                    datasets: [{{
                        label: 'Trend',
                        data: rollingData,
                        borderColor: 'rgba(220, 38, 38, 0.8)',
                        borderWidth: 2.5,
                        pointRadius: 0,
                        pointHoverRadius: 0,
                        pointHitRadius: 0,
                        showLine: true,
                        tension: 0.3,
                        type: 'line',
                        order: 1,
                    }}, {{
                        label: 'Age Grade %',
                        data: d.datesISO.map((dt, i) => ({{ x: dt, y: d.values[i] }})),
                        pointBackgroundColor: d.colors,
                        pointRadius: d.sizes,
                        order: 2,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            filter: function(item) {{ return item.datasetIndex === 1; }},
                            callbacks: {{
                                title: function(ctx) {{ 
                                    if (!ctx.length || ctx[0].datasetIndex !== 1) return '';
                                    const i = ctx[0].dataIndex;
                                    return (i >= 0 && i < d.dates.length) ? d.dates[i] : '';
                                }},
                                label: function(ctx) {{ 
                                    if (ctx.datasetIndex !== 1) return null;
                                    const i = ctx.dataIndex;
                                    if (i < 0 || i >= d.values.length) return '';
                                    const prefix = d.parkrun[i] ? '🅿️ parkrun ' : '';
                                    return prefix + d.labels[i] + ': ' + d.values[i] + '%'; 
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            type: 'time',
                            time: {{
                                unit: days <= 730 ? 'month' : 'year',
                                displayFormats: {{ month: 'MMM yy', year: 'yyyy' }}
                            }},
                            ticks: {{ maxTicksLimit: 8, font: {{ size: 10 }} }}
                        }},
                        y: {{
                            display: true,
                            ticks: {{
                                callback: function(v) {{ return v + '%'; }},
                                font: {{ size: 10 }}
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        if (agAllDates.length > 0) {{
            renderAgChart(365);
            
            document.getElementById('agToggle').addEventListener('click', function(e) {{
                if (e.target.tagName === 'BUTTON') {{
                    this.querySelectorAll('button').forEach(b => b.classList.remove('active'));
                    e.target.classList.add('active');
                    renderAgChart(parseInt(e.target.dataset.range));
                }}
            }});
        }}
        
        // --- Recent Runs toggle ---
        const recentRunsData = {{
            all: {json.dumps(recent_runs['all'])},
            races: {json.dumps(recent_runs['races'])}
        }};
        
        function updateRecentRunsTable(filter) {{
            const runs = recentRunsData[filter];
            const tbody = document.getElementById('recentRunsBody');
            
            if (runs.length === 0) {{
                tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-dim);">No races found</td></tr>';
                return;
            }}
            
            tbody.innerHTML = runs.map(run => {{
                // v51: Trend mover arrow (mode-aware)
                const modeDelta = currentMode === 'gap' ? run.delta_gap : currentMode === 'sim' ? run.delta_sim : run.delta;
                let deltaHtml = '';
                if (modeDelta !== null && modeDelta !== undefined) {{
                    if (modeDelta >= 1.0) deltaHtml = '<span style="color:#16a34a;font-weight:bold;" title="RFL +' + modeDelta.toFixed(1) + '%"> ▲▲</span>';
                    else if (modeDelta >= 0.3) deltaHtml = '<span style="color:#16a34a;" title="RFL +' + modeDelta.toFixed(1) + '%"> ▲</span>';
                    else if (modeDelta <= -1.0) deltaHtml = '<span style="color:#dc2626;font-weight:bold;" title="RFL ' + modeDelta.toFixed(1) + '%"> ▼▼</span>';
                    else if (modeDelta <= -0.3) deltaHtml = '<span style="color:#dc2626;" title="RFL ' + modeDelta.toFixed(1) + '%"> ▼</span>';
                }}
                const rflVal = currentMode === 'gap' ? (run.rfl_gap || '-') : currentMode === 'sim' ? (run.rfl_sim || '-') : (run.rfl || '-');
                return `
                <tr>
                    <td>${{run.date}}</td>
                    <td>${{run.name}} ${{run.race ? '<span class="race-badge">RACE</span>' : ''}}${{deltaHtml}}</td>
                    <td>${{run.dist ? run.dist + ' km' : '-'}}</td>
                    <td>${{run.pace}}</td>
                    <td class="power-only">${{run.npower || '-'}}</td>
                    <td>${{run.hr || '-'}}</td>
                    <td>${{run.tss || '-'}}</td>
                    <td>${{rflVal}}</td>
                </tr>`;
            }}).join('');
        }}
        
        // Initialize with all runs
        updateRecentRunsTable('all');
        
        // Toggle handler
        document.getElementById('recentRunsToggle').addEventListener('click', function(e) {{
            if (e.target.tagName === 'BUTTON') {{
                this.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
                e.target.classList.add('active');
                const filter = e.target.getAttribute('data-filter');
                updateRecentRunsTable(filter);
            }}
        }});
    
    // Phase 2: Mode switching
    function setMode(mode) {{
        currentMode = mode;
        // Update toggle buttons
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        document.getElementById('mode-' + mode).classList.add('active');
        
        // Update stats cards
        const ms = modeStats[mode];
        const rflEl = document.getElementById('rfl-value');
        if (rflEl) rflEl.textContent = ms.rfl + '%';
        const rflDeltaEl = document.getElementById('rfl-delta');
        if (rflDeltaEl) rflDeltaEl.textContent = ms.rflDelta;
        const agEl = document.getElementById('ag-value');
        if (agEl) agEl.textContent = (ms.ag && ms.ag !== 'None') ? ms.ag + '%' : '-';
        const p5 = document.getElementById('pred-5k');
        if (p5) p5.textContent = ms.pred5k;
        const p10 = document.getElementById('pred-10k');
        if (p10) p10.textContent = ms.pred10k;
        const ph = document.getElementById('pred-hm');
        if (ph) ph.textContent = ms.predHm;
        const pm = document.getElementById('pred-mara');
        if (pm) pm.textContent = ms.predMara;
        
        // Update RFL chart — switch which data series is shown
        if (typeof rfChart !== 'undefined' && typeof currentRfRange !== 'undefined') {{
            updateRfChart(currentRfRange);
        }}
        
        // Update 14-day RFL trend chart
        if (typeof rfl14Chart !== 'undefined' && typeof updateRfl14Chart === 'function') {{
            updateRfl14Chart(mode);
        }}
        
        // Re-render prediction chart with mode-appropriate predictions
        if (typeof renderPredChart === 'function' && typeof currentPredDist !== 'undefined') {{
            const showPR = document.getElementById('predParkrunToggle');
            renderPredChart(currentPredDist, showPR ? showPR.checked : false);
        }}
        
        // Update RFL label
        const rflLabel = document.getElementById('rfl-label');
        if (rflLabel) {{
            const labels = {{ stryd: 'RFL Trend', gap: 'RFL (GAP)', sim: 'RFL (Sim)' }};
            rflLabel.textContent = labels[mode];
        }}
        
        // Update Easy RF Gap stat card
        const easyGapEl = document.getElementById('easy-rf-gap-value');
        if (easyGapEl) {{
            easyGapEl.textContent = easyGapEl.dataset[mode] || '-';
        }}
        
        // Hide/show power-only elements via body class (CSS handles all hiding)
        const isGap = mode === 'gap';
        const isSim = mode === 'sim';
        document.body.classList.toggle('gap-mode', isGap);
        document.body.classList.toggle('sim-mode', isSim);
        
        // Update alert banner for current mode
        if (typeof renderAlertBanner === 'function') renderAlertBanner(mode);
        
        // Update zone badge (hide power info in GAP mode, use mode CP for Sim)
        const zb = document.getElementById('zone-badge');
        if (zb && typeof ZONE_CP !== 'undefined') {{
            if (isGap) {{
                zb.textContent = 'LTHR ' + ZONE_LTHR + ' · Max ~' + ZONE_MAXHR;
            }} else {{
                const modeCP = mode === 'sim' ? Math.round(ZONE_PEAK_CP * parseFloat(modeStats.sim.rfl) / 100) : ZONE_CP;
                zb.textContent = 'CP ' + modeCP + 'W · LTHR ' + ZONE_LTHR + ' · Max ~' + ZONE_MAXHR;
            }}
        }}
        
        // Update top races description
        const trd = document.getElementById('top-races-desc');
        if (trd) trd.textContent = 'Best races by normalised AG% (adjusted for temperature, terrain, surface, distance).';
        
        // Reset zone views when switching to GAP (use Race Pace as default)
        if (isGap) {{
            const wkBtns = document.querySelectorAll('#wk-mode button');
            wkBtns.forEach(b => b.classList.remove('active'));
            // Find the gap-only Race (Pace) button and activate it
            const gapBtn = document.querySelector('#wk-mode .gap-only');
            if (gapBtn) {{ gapBtn.classList.add('active'); wkMode='race'; renderWk(); }}
            const prBtns = document.querySelectorAll('#pr-mode button');
            prBtns.forEach(b => b.classList.remove('active'));
            const gapPrBtn = document.querySelector('#pr-mode .gap-only');
            if (gapPrBtn) {{ gapPrBtn.classList.add('active'); prMode='race'; renderPR(); }}
        }} else {{
            // Switching back from GAP — reset to HR zone view and re-activate button
            wkMode='hr'; prMode='hr';
            document.querySelectorAll('#wk-mode button').forEach(b => b.classList.remove('active'));
            const wkHrBtn = document.querySelector('#wk-mode button');
            if (wkHrBtn) wkHrBtn.classList.add('active');
            document.querySelectorAll('#pr-mode button').forEach(b => b.classList.remove('active'));
            const prHrBtn = document.querySelector('#pr-mode button');
            if (prHrBtn) prHrBtn.classList.add('active');
            if (typeof renderWk === 'function') renderWk();
            if (typeof renderPR === 'function') renderPR();
        }}
        
        // Update race readiness cards (predicted time, target power/pace)
        const _surfF = {{'indoor_track':{{pw:1.0,re:1.04}},'track':{{pw:1.0,re:1.02}},'road':{{pw:1.0,re:1.0}},'trail':{{pw:0.95,re:0.97}},'undulating_trail':{{pw:0.90,re:0.95}}}};
        const _pdA=[[3,1.07],[5,1.05],[10,1.00],[21.097,0.95],[42.195,0.90]];
        function _roadCpF(d){{d=Math.max(d,1);const ld=Math.log(d);if(ld<=Math.log(_pdA[0][0]))return _pdA[0][1];if(ld>=Math.log(_pdA[_pdA.length-1][0]))return _pdA[_pdA.length-1][1];for(let i=0;i<_pdA.length-1;i++){{const l0=Math.log(_pdA[i][0]),l1=Math.log(_pdA[i+1][0]);if(ld>=l0&&ld<=l1){{const f=(ld-l0)/(l1-l0);return _pdA[i][1]+f*(_pdA[i+1][1]-_pdA[i][1]);}}}}return 1.0;}};
        // Map distance_key to modeStats raw seconds keys
        const _stepbKeyMap = {{'5K':'pred5k_s','10K':'pred10k_s','HM':'predHm_s','Mara':'predMara_s'}};
        PLANNED_RACES.forEach((race, idx) => {{
            const dist = race.distance_km || 5.0;
            const sf = _surfF[race.surface||'road'] || _surfF.road;
            const factor = _roadCpF(dist) * sf.pw;
            const mcp = ms.cp;
            const pw = Math.round(mcp * factor);
            // Update power target
            const pwEl = document.getElementById('race-pw-' + idx);
            if (pwEl) pwEl.textContent = pw + 'W';
            // For standard road distances, use StepB predictions (matches stats grid)
            // For non-standard distances or non-road surfaces, use continuous model
            const stepbKey = _stepbKeyMap[race.distance_key];
            const stepbSecs = (stepbKey && (race.surface||'road')==='road') ? ms[stepbKey] : 0;
            let t;
            if (stepbSecs && stepbSecs > 0) {{
                t = stepbSecs;
            }} else {{
                const re = (ZONE_RE || 0.914) * sf.re;
                const speed = (pw / {ATHLETE_MASS_KG_DASH}) * re;
                t = speed > 0 ? Math.round(dist * 1000 / speed) : 0;
            }}
            const predEl = document.getElementById('race-pred-' + idx);
            if (predEl && t > 0) {{
                const hrs = Math.floor(t / 3600);
                const mins = Math.floor((t % 3600) / 60);
                const secs = t % 60;
                predEl.textContent = hrs > 0 ? hrs + ':' + String(mins).padStart(2,'0') + ':' + String(secs).padStart(2,'0') : mins + ':' + String(secs).padStart(2,'0');
            }}
            // Update pace target
            const paceEl = document.getElementById('race-pace-' + idx);
            if (paceEl && t > 0) {{
                const pace = t / dist;
                const pm = Math.floor(pace / 60);
                const ps = Math.round(pace % 60);
                paceEl.textContent = pm + ':' + String(ps).padStart(2,'0') + '/km';
            }}
        }});
        
        // Recalculate race readiness specificity (pace for GAP, power for others)
        if (typeof calcSpecificity === 'function') calcSpecificity();
        
        // Re-render recent runs and top races to apply power-only visibility
        if (typeof updateRecentRunsTable === 'function') {{
            const activeFilter = document.querySelector('#recentRunsToggle button.active');
            updateRecentRunsTable(activeFilter ? activeFilter.getAttribute('data-filter') : 'all');
        }}
        if (typeof updateTopRacesTable === 'function' && document.getElementById('topRacesBody')) {{
            const activePeriod = document.querySelector('#topRacesToggle button.active');
            updateTopRacesTable(activePeriod ? activePeriod.getAttribute('data-period') : '1y');
        }}
    }}
    
    // Apply initial mode (ensures GAP athletes see GAP data on load)
    if (typeof setMode === 'function') setMode(currentMode);
    
    </script>
</body>
</html>
'''
    return html

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("Mobile Dashboard Generator")
    print("=" * 40)
    
    # Load data
    df, ctl, atl, tsb, weight, age_grade, critical_power, race_predictions = load_and_process_data()
    
    # Process data
    print("Processing stats...")
    stats = get_summary_stats(df)
    stats['ctl'] = ctl
    stats['atl'] = atl
    stats['tsb'] = tsb
    stats['weight'] = weight
    stats['age_grade'] = age_grade
    stats['critical_power'] = critical_power
    stats['race_predictions'] = race_predictions
    
    print("Processing RF trend (multiple ranges)...")
    rf_data = {
        '90': get_rfl_trend_data(df, days=90),
        '180': get_rfl_trend_data(df, days=180),
        '365': get_rfl_trend_data(df, days=365),
        '730': get_rfl_trend_data(df, days=730),
        '1095': get_rfl_trend_data(df, days=1095),
        '1825': get_rfl_trend_data(df, days=1825),
        'all': get_rfl_trend_data(df, days=99999),
    }
    
    print("Processing volume data (weeks/months/years)...")
    volume_data = {
        'W': {
            '12': get_weekly_volume(df, weeks=12),
            '26': get_weekly_volume(df, weeks=26),
            '52': get_weekly_volume(df, weeks=52),
        },
        'M': {
            '6': get_monthly_volume(df, months=6),
            '12': get_monthly_volume(df, months=12),
            '24': get_monthly_volume(df, months=24),
            'All': get_monthly_volume(df, months=None),
        },
        'Y': {
            '3': get_yearly_volume(df, years=3),
            '5': get_yearly_volume(df, years=5),
            'All': get_yearly_volume(df, years=None),
        }
    }
    
    print("Processing CTL/ATL trend (multiple ranges)...")
    ctl_atl_data = {
        '90': get_ctl_atl_trend(df, days=90),
        '180': get_ctl_atl_trend(df, days=180),
        '365': get_ctl_atl_trend(df, days=365),
        '730': get_ctl_atl_trend(df, days=730),
        '1095': get_ctl_atl_trend(df, days=1095),
        '1825': get_ctl_atl_trend(df, days=1825),
    }
    
    print("Processing daily CTL/ATL lookup for live updates...")
    ctl_atl_lookup = get_daily_ctl_atl_lookup(MASTER_FILE)
    
    print("Processing daily RFL trend...")
    rfl_trend_dates, rfl_trend_values, rfl_trendline, rfl_projection, rfl_ci_upper, rfl_ci_lower = get_daily_rfl_trend(MASTER_FILE, days=14, rfl_col='RFL_Trend')
    _, rfl_trend_gap_values, rfl_trendline_gap, rfl_proj_gap, rfl_ci_upper_gap, rfl_ci_lower_gap = get_daily_rfl_trend(MASTER_FILE, days=14, rfl_col='RFL_gap_Trend')
    _, rfl_trend_sim_values, rfl_trendline_sim, rfl_proj_sim, rfl_ci_upper_sim, rfl_ci_lower_sim = get_daily_rfl_trend(MASTER_FILE, days=14, rfl_col='RFL_sim_Trend')
    
    print("Processing all-time weekly RFL...")
    alltime_rfl_dates, alltime_rfl_values = get_alltime_weekly_rfl(MASTER_FILE)
    
    print("Processing recent runs...")
    recent_runs = get_recent_runs(df)
    
    print("Processing top races...")
    top_races = get_top_races(df)
    
    # v51: Alert data, weight chart, prediction trend, age grade
    print("Processing alert data...")
    alert_data = get_alert_data(df)
    
    print("Processing weight chart data...")
    weight_data = get_weight_chart_data(MASTER_FILE, months=200)
    
    print("Processing prediction trend data...")
    prediction_data = get_prediction_trend_data(df)
    
    print("Processing age grade data...")
    ag_data = get_age_grade_data(df)
    
    # Zone data for training zones section
    print("Getting zone data...")
    zone_data = get_zone_data(df)
    
    # Generate HTML
    print("Generating HTML...")
    html = generate_html(stats, rf_data, volume_data, ctl_atl_data, ctl_atl_lookup, rfl_trend_dates, rfl_trend_values, rfl_trendline, rfl_projection, rfl_ci_upper, rfl_ci_lower, alltime_rfl_dates, alltime_rfl_values, recent_runs, top_races, alert_data=alert_data, weight_data=weight_data, prediction_data=prediction_data, ag_data=ag_data, zone_data=zone_data, rfl_trend_gap={'values': rfl_trend_gap_values, 'trendline': rfl_trendline_gap, 'projection': rfl_proj_gap, 'ci_upper': rfl_ci_upper_gap, 'ci_lower': rfl_ci_lower_gap}, rfl_trend_sim={'values': rfl_trend_sim_values, 'trendline': rfl_trendline_sim, 'projection': rfl_proj_sim, 'ci_upper': rfl_ci_upper_sim, 'ci_lower': rfl_ci_lower_sim})
    
    # Write file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n[OK] Dashboard generated: {OUTPUT_FILE}")
    print(f"   Open in any browser (desktop or mobile)")
    
    # Try to open in browser
    try:
        import webbrowser
        webbrowser.open(OUTPUT_FILE)
    except:
        pass

if __name__ == "__main__":
    main()
