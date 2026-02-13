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
from datetime import datetime, timedelta
import json
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
MASTER_FILE = r"Master_FULL_GPSQ_ID_post.xlsx"  # v43+ Master with RF/CTL columns
ATHLETE_DATA_FILE = r"athlete_data.csv"  # Weight, non-running TSS
OUTPUT_FILE = r"index.html"

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
        if pd.notna(pred_10k):
            race_predictions['10k'] = format_seconds(pred_10k)
        if pd.notna(pred_hm):
            race_predictions['Half Marathon'] = format_seconds(pred_hm)
        if pd.notna(pred_mara):
            race_predictions['Marathon'] = format_seconds(pred_mara)
        
        print(f"  From Master: CP={critical_power}W, Age Grade={age_grade}%")
        print(f"  Race predictions: {race_predictions}")
        
        # Phase 2: GAP and Sim predictions for dashboard mode toggle
        for mode in ('gap', 'sim'):
            mode_preds = {}
            for dist, col in [('5k', f'pred_5k_s_{mode}'), ('10k', f'pred_10k_s_{mode}'),
                               ('Half Marathon', f'pred_hm_s_{mode}'), ('Marathon', f'pred_marathon_s_{mode}')]:
                val = latest.get(col)
                if pd.notna(val):
                    mode_preds[dist] = format_seconds(val)
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


def _calc_rfl_delta(df, days_back=14):
    """Calculate change in RFL_Trend vs N days ago.
    Returns delta in percentage points (e.g. +2.5 or -1.3), or None."""
    rfl = pd.to_numeric(df.get('RFL_Trend'), errors='coerce')
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
        'rfl_14d_delta': _calc_rfl_delta(df, 14),
        'latest_hr': int(latest[hr_col]) if latest is not None and pd.notna(latest.get(hr_col)) else None,
        'latest_dist': round(latest[dist_col], 2) if latest is not None and pd.notna(latest.get(dist_col)) else None,
        # v51: Easy RF gap
        'easy_rfl_gap': round(latest.get('Easy_RFL_Gap', 0) * 100, 1) if latest is not None and pd.notna(latest.get('Easy_RFL_Gap')) else None,
        # Phase 2: GAP and Sim RFL for mode toggle
        'latest_rfl_gap': round((latest.get('RFL_gap_Trend') or 0) * 100, 1) if latest is not None and pd.notna(latest.get('RFL_gap_Trend')) else None,
        'latest_rfl_sim': round((latest.get('RFL_sim_Trend') or 0) * 100, 1) if latest is not None and pd.notna(latest.get('RFL_sim_Trend')) else None,
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


def get_daily_rfl_trend(master_file, days=14):
    """Get daily RFL trend from Master's Daily sheet with trendline, projection and 95% CI."""
    try:
        df = pd.read_excel(master_file, sheet_name='Daily')
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Get up to end of today
        today = pd.Timestamp.now().normalize() + timedelta(days=1) - timedelta(microseconds=1)
        df_past = df[df['Date'] <= today].tail(days)
        
        dates = df_past['Date'].dt.strftime('%d %b').tolist()
        # Master's Daily sheet has RFL_Trend (0-1 scale) - v44.5
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
                # v51: trend mover
                'delta': round(row['RFL_Trend_Delta'] * 100, 2) if pd.notna(row.get('RFL_Trend_Delta')) else None,
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
    """v51: Get current alert status for dashboard banner."""
    alert_cols = ['Alert_1', 'Alert_1b', 'Alert_2', 'Alert_3b', 'Alert_5']
    alerts = []
    
    if len(df) == 0:
        return alerts
    
    latest = df.iloc[-1]
    
    alert_defs = {
        'Alert_1': {'name': 'Training more, scoring worse', 'level': 'concern', 'icon': '⚠️'},
        'Alert_1b': {'name': 'Taper not working', 'level': 'concern', 'icon': '⚠️'},
        'Alert_2': {'name': 'Deep fatigue', 'level': 'watch', 'icon': '👀'},
        'Alert_3b': {'name': 'Easy run outlier', 'level': 'watch', 'icon': '👀'},
        'Alert_5': {'name': 'Easy RF divergence', 'level': 'concern', 'icon': '⚠️'},
    }
    
    for col in alert_cols:
        if col in df.columns and latest.get(col, False):
            defn = alert_defs.get(col, {})
            detail = ''
            if col == 'Alert_5' and pd.notna(latest.get('Easy_RFL_Gap')):
                detail = f" (gap {latest['Easy_RFL_Gap']*100:.1f}%)"
            elif col == 'Alert_2' and pd.notna(latest.get('TSB')):
                detail = f" (TSB {latest['TSB']:.0f})"
            alerts.append({
                'name': defn.get('name', col),
                'level': defn.get('level', 'info'),
                'icon': defn.get('icon', 'ℹ️'),
                'detail': detail,
            })
    
    return alerts


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


def get_top_races(df, n=10):
    """Get top race performances by RFL for different time periods.
    
    Returns dict with keys: '1y', '2y', '3y', '5y', 'all'
    Each value is a list of up to n race dicts sorted by RFL descending.
    """
    # Filter to races only
    race_col = 'race_flag' if 'race_flag' in df.columns else 'Race'
    races = df[df[race_col] == 1].copy()
    
    if len(races) == 0:
        return {period: [] for period in ['1y', '2y', '3y', '5y', 'all']}
    
    # Sort by RFL (individual run fitness level)
    sort_col = 'RFL' if 'RFL' in races.columns else 'RF_adj'
    if sort_col not in races.columns:
        return {period: [] for period in ['1y', '2y', '3y', '5y', 'all']}
    
    # Filter out rows without valid scores
    races = races[races[sort_col].notna()].copy()
    
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
        
        moving_s = row.get(elapsed_col, row.get(moving_col, None))
        
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
        
        # Age grade percentage
        ag_val = round(row['age_grade_pct'], 1) if pd.notna(row.get('age_grade_pct')) else None
        
        # RFL percentage (all three modes)
        rfl_val = round(row['RFL'] * 100, 1) if pd.notna(row.get('RFL')) else None
        rfl_gap_val = round(row['RFL_gap'] * 100, 1) if pd.notna(row.get('RFL_gap')) else None
        rfl_sim_val = round(row['RFL_sim'] * 100, 1) if pd.notna(row.get('RFL_sim')) else None
        
        return {
            'date': row['date'].strftime('%d %b %y'),
            'name': str(name_val) if name_val else '',
            'dist': round(dist_km, 1) if dist_km > 0 else None,
            'time': time_str,
            'ag': ag_val,
            'rfl': rfl_val,
            'rfl_gap': rfl_gap_val,
            'rfl_sim': rfl_sim_val,
        }
    
    result = {}
    for period_name, delta in periods.items():
        if delta is None:
            period_races = races
        else:
            cutoff = now - delta
            period_races = races[races['date'] >= cutoff]
        
        # Sort by RFL descending, take top n
        top = period_races.nlargest(n, sort_col)
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
    
    return result


# ============================================================================
# v51: AGE GRADE TREND DATA
# ============================================================================
def get_age_grade_data(df):
    """Get age grade trend data for chart.
    Returns dict with dates, ag_pct, distance labels, colours, and rolling average."""
    race_col = 'race_flag' if 'race_flag' in df.columns else 'Race'
    races = df[(df[race_col] == 1) & (df['age_grade_pct'].notna())].copy() if 'age_grade_pct' in df.columns else pd.DataFrame()
    
    if len(races) == 0:
        return {'dates': [], 'values': [], 'dist_labels': [], 'dist_codes': [], 'rolling_avg': []}
    
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
try:
    from config import PEAK_CP_WATTS as _cfg_cp, ATHLETE_MASS_KG as _cfg_mass
    from config import ATHLETE_LTHR as _cfg_lthr, ATHLETE_MAX_HR as _cfg_maxhr
    from config import PLANNED_RACES as _cfg_races
    PEAK_CP_WATTS_DASH = _cfg_cp
    ATHLETE_MASS_KG_DASH = _cfg_mass
    LTHR_DASH = _cfg_lthr
    MAX_HR_DASH = _cfg_maxhr
except ImportError:
    PEAK_CP_WATTS_DASH = 372
    ATHLETE_MASS_KG_DASH = 76.0
    LTHR_DASH = 178
    MAX_HR_DASH = 192
    _cfg_races = None

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
        {'name': r['name'], 'date': r['date'], 'distance_key': _distance_km_to_key(r['distance_km'])}
        for r in _cfg_races
    ]
else:
    PLANNED_RACES_DASH = [
        {'name': '5K London', 'date': '2026-02-27', 'distance_key': '5K'},
        {'name': 'HM Stockholm', 'date': '2026-04-25', 'distance_key': 'HM'},
    ]


def get_zone_data(df):
    """Extract run data with per-second time-in-zone from NPZ cache."""
    import glob, os
    
    recent = df.tail(100).copy()
    npower_col = 'npower_w' if 'npower_w' in recent.columns else 'nPower'
    hr_col = 'avg_hr' if 'avg_hr' in recent.columns else 'Avg_HR'
    moving_col = 'moving_time_s' if 'moving_time_s' in recent.columns else 'Moving_s'
    dist_col = 'distance_km' if 'distance_km' in recent.columns else 'Distance_m'
    
    current_rfl = float(df.iloc[-1].get('RFL_Trend', 0.90)) if len(df) > 0 else 0.90
    if pd.isna(current_rfl):
        current_rfl = 0.90
    current_cp = round(PEAK_CP_WATTS_DASH * current_rfl)
    
    re_col = 'RE_avg'
    if re_col in df.columns:
        recent_re = df.tail(100)[re_col].dropna()
        re_p90 = round(float(recent_re.quantile(0.90)), 4) if len(recent_re) > 0 else 0.92
    else:
        re_p90 = 0.92
    
    # Zone boundaries — 5-zone model anchored to Lactate Threshold
    # LT power derived from CP via Riegel: LT = CP × (40/60)^(1/1.06 - 1) ≈ 0.956 × CP
    LT_POWER = round(current_cp * 0.956)
    RF_THR = current_cp / LTHR_DASH
    # HR zones (5): Z1 Easy, Z2 Aerobic, Z3 Tempo (up to LT), Z4 Threshold, Z5 Max
    hr_bounds = [0, 140, 157, 178, 184, 9999]
    hr_znames = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
    # Power zones (5): LT-anchored
    pw_z12 = round(LT_POWER * 0.75)  # ~240W
    pw_z23 = round(LT_POWER * 0.88)  # ~282W
    pw_z45 = round(current_cp * 1.07)  # ~358W — above 5K effort
    pw_bounds = [0, pw_z12, pw_z23, LT_POWER, pw_z45, 9999]
    pw_znames = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
    # Race power zones: contiguous midpoint bands
    m_pw = current_cp * 0.90
    h_pw = current_cp * 0.95
    t_pw = current_cp * 1.00
    f_pw = current_cp * 1.05
    race_pw_bounds = [0, round(m_pw*0.93), round((m_pw+h_pw)/2), round((h_pw+t_pw)/2), round((t_pw+f_pw)/2), round(f_pw*1.05), 9999]
    race_pw_names = ['Other', 'Mara', 'HM', '10K', '5K', 'Sub-5K']
    # Race HR zones: contiguous
    race_hr_bounds = [0, 163, 170, 175, 180, 184, 9999]
    race_hr_names = ['Other', 'Mara', 'HM', '10K', '5K', 'Sub-5K']
    
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
    
    # Find NPZ cache directory
    npz_dir = None
    for candidate in ['persec_cache_FULL', '../persec_cache_FULL', 'persec_cache']:
        if os.path.isdir(candidate):
            npz_dir = candidate
            break
    
    # Build index of available NPZ files by date prefix
    npz_index = {}
    if npz_dir:
        for fp in glob.glob(os.path.join(npz_dir, '*.npz')):
            base = os.path.basename(fp).replace('.npz', '')
            npz_index[base] = fp
    
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
        npz_path = npz_index.get(run_id)
        if npz_path:
            try:
                npz_data = np.load(npz_path, allow_pickle=True)
                pw_arr = npz_data['power_w'].copy()
                hr_arr = npz_data['hr_bpm']
                spd_arr = npz_data['speed_mps']
                grd_arr = npz_data['grade']
                
                # Clean power: exclude sections with implausible cost-of-transport
                # COT = power/(mass*speed); sensor errors inflate COT >25% above run median
                running = spd_arr > 2.0  # only assess at running pace (>8:20/km)
                if running.sum() > 60:
                    grd_safe = np.where(np.isnan(grd_arr), 0, grd_arr)
                    cot = np.full_like(pw_arr, np.nan)
                    mass_kg = row.get('weight_kg', ATHLETE_MASS_KG_DASH)
                    cot[running] = pw_arr[running] / (mass_kg * spd_arr[running])
                    cot_adj = cot - 4.0 * grd_safe
                    cot_30 = pd.Series(cot_adj).rolling(30, min_periods=15).median().values
                    cot_med = np.nanmedian(cot_adj)
                    bad_pw = running & (~np.isnan(cot_30)) & (cot_30 > cot_med * 1.25)
                    pw_arr[bad_pw] = 0.0  # zero so rolling avg doesn't bleed
                    # Also zero HR for these seconds to keep zones consistent
                    hr_arr = hr_arr.copy()
                    hr_arr[bad_pw] = 0.0
                
                run_entry['hz'] = _time_in_zones(hr_arr, hr_bounds, hr_znames)
                run_entry['pz'] = _time_in_zones(pw_arr, pw_bounds, pw_znames)
                run_entry['rpz'] = _time_in_zones(pw_arr, race_pw_bounds, race_pw_names)
                run_entry['rhz'] = _time_in_zones(hr_arr, race_hr_bounds, race_hr_names)
                npz_hits += 1
            except Exception as e:
                pass  # Fall back to no zone data
        
        runs.append(run_entry)
    
    print(f"  Zone data: {len(runs)} runs, {npz_hits} with NPZ, CP={current_cp}W (RFL={current_rfl:.4f}), RE_p90={re_p90}")
    
    # Compute GAP target paces (sec/km) from latest predictions
    gap_target_paces = {}
    latest = df.iloc[-1] if len(df) > 0 else None
    if latest is not None:
        pace_dists = {'5k': 5.0, '10k': 10.0, 'hm': 21.097, 'marathon': 42.195}
        for dist_key, dist_km in pace_dists.items():
            col = f'pred_{dist_key}_s_gap'
            if col in df.columns and pd.notna(latest.get(col)):
                gap_target_paces[dist_key] = round(latest[col] / dist_km)
    
    return {'runs': runs, 'current_cp': current_cp, 'current_rfl': round(current_rfl, 4), 're_p90': re_p90, 'gap_target_paces': gap_target_paces}


# ============================================================================
# v51.7: ZONE HTML GENERATOR
# ============================================================================
def _generate_zone_html(zone_data):
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
    # Power zones: LT-anchored
    lt_pw = round(cp * 0.956)
    pw_z12 = round(lt_pw * 0.75)
    pw_z23 = round(lt_pw * 0.88)
    pw_z45 = round(cp * 1.07)
    hr_ranges =   ['<140',      '140–157',      '157–178',      '178–184',      '>184']
    pw_strs =     [f'<{pw_z12}', f'{pw_z12}–{pw_z23}', f'{pw_z23}–{lt_pw}', f'{lt_pw}–{pw_z45}', f'>{pw_z45}']
    pct_strs =    [f'<{round(pw_z12/cp*100)}', f'{round(pw_z12/cp*100)}–{round(pw_z23/cp*100)}', f'{round(pw_z23/cp*100)}–{round(lt_pw/cp*100)}', f'{round(lt_pw/cp*100)}–{round(pw_z45/cp*100)}', f'>{round(pw_z45/cp*100)}']
    effort_hints = ['', 'easy–Mara', 'Mara–10K', '10K–5K', '>5K']
    combined_rows = ''
    for i in range(5):
        combined_rows += f'<tr><td><span class="zd" style="background:{zone_colors[i]}"></span><strong>{zone_names[i]}</strong></td><td style="color:var(--text-dim);font-family:DM Sans">{zone_labels[i]}</td><td>{hr_ranges[i]}</td><td class="power-only">{pw_strs[i]}W</td><td class="power-only pct-col" style="color:var(--text-dim)">{pct_strs[i]}%</td></tr>'
    
    # Race readiness cards
    race_cards = ''
    from datetime import date as dt_date
    for race in PLANNED_RACES_DASH:
        key = race['distance_key']
        factor = RACE_POWER_FACTORS_DASH.get(key, 1.0)
        dist_km = RACE_DISTANCES_KM_DASH.get(key, 5.0)
        pw = round(cp * factor)
        dist_m = dist_km * 1000
        speed = (pw / ATHLETE_MASS_KG_DASH) * re_p90
        t = round(dist_m / speed) if speed > 0 else 0
        band = round(pw * 0.03)
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
        days_str = f"{days_to}d" if days_to > 0 else "TODAY"
        
        race_cards += f'''<div class="rc">
            <div class="rh"><span class="rn">{race['name']}</span><span class="rd">{race['date']} · {days_str}</span></div>
            <div class="rs">
                <div class="power-only"><div class="rv" style="color:var(--accent)">{pw}W</div><div class="rl">Target</div><div class="rx">±{band}W</div></div>
                <div class="pace-target" style="display:none"><div class="rv" style="color:#4ade80">{pace_str}</div><div class="rl">Target pace</div></div>
                <div><div class="rv">{t_str}</div><div class="rl">Predicted</div><div class="rx power-only">{pace_str}</div></div>
                <div><div class="rv" id="spec14_{key}">—</div><div class="rl">14-day</div><div class="rx">at effort</div></div>
                <div><div class="rv" id="spec28_{key}">—</div><div class="rl">28-day</div><div class="rx">at effort</div></div>
            </div>
        </div>'''
    
    # Zone run data as JSON for JS
    import json as _json
    zone_runs_json = _json.dumps(zone_data['runs'])
    
    return f'''
    <div class="card" id="zone-table-card">
        <h2>Training Zones <span class="badge" id="zone-badge">LT {lt_pw}W · CP {cp}W · LTHR {LTHR_DASH} · Max ~{MAX_HR_DASH}</span></h2>
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
        <div id="wk-bars"></div>
        <div id="wk-leg" class="legend"></div>
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
        <div class="chart-wrapper"><canvas id="prChart"></canvas></div>
        <div id="pr-leg" class="legend"></div>
        <div class="note">Zone split from 30s rolling average of per-second power/HR data where available.</div>
    </div>

    <script>
    const ZONE_RUNS=''' + zone_runs_json + f''';
    const ZONE_CP={cp};const ZONE_PEAK_CP={PEAK_CP_WATTS_DASH};const ZONE_MASS={ATHLETE_MASS_KG_DASH};const ZONE_RE={re_p90};const ZONE_LTHR={LTHR_DASH};const ZONE_MAXHR={MAX_HR_DASH};
    const HR_Z=[
      {{id:'Z1',name:'Easy',lo:0,hi:140,c:'#3b82f6'}},
      {{id:'Z2',name:'Aerobic',lo:140,hi:157,c:'#22c55e'}},
      {{id:'Z3',name:'Tempo',lo:157,hi:178,c:'#eab308'}},
      {{id:'Z4',name:'Threshold',lo:178,hi:184,c:'#f97316'}},
      {{id:'Z5',name:'Max',lo:184,hi:9999,c:'#ef4444'}}
    ];
    const LT_PW=Math.round(ZONE_CP*0.956);
    const PW_Z12=Math.round(LT_PW*0.75);
    const PW_Z23=Math.round(LT_PW*0.88);
    const PW_Z45=Math.round(ZONE_CP*1.07);
    const PW_Z=[
      {{id:'Z1',name:'Easy',lo:0,hi:PW_Z12,c:'#3b82f6'}},
      {{id:'Z2',name:'Aerobic',lo:PW_Z12,hi:PW_Z23,c:'#22c55e'}},
      {{id:'Z3',name:'Tempo',lo:PW_Z23,hi:LT_PW,c:'#eab308'}},
      {{id:'Z4',name:'Threshold',lo:LT_PW,hi:PW_Z45,c:'#f97316'}},
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
    function makeRaceHrZones(){{return[{{id:'Sub-5K',name:'Sub-5K',lo:184,hi:9999,c:'#4ade80'}},{{id:'5K',name:'5K',lo:180,hi:184,c:'#f472b6'}},{{id:'10K',name:'10K',lo:175,hi:180,c:'#fb923c'}},{{id:'HM',name:'HM',lo:170,hi:175,c:'#a78bfa'}},{{id:'Mara',name:'Mara',lo:163,hi:170,c:'#38bdf8'}},{{id:'Other',name:'Other',lo:0,hi:163,c:'#4b5563'}}];}}
    function makeRacePaceZones(){{const p5={pace_5k},p10={pace_10k},ph={pace_hm},pm={pace_mara};const s5=Math.round(p5*0.97),m5=Math.round((p5+p10)/2),mt=Math.round((p10+ph)/2),mh=Math.round((ph+pm)/2),slow=Math.round(pm*1.07);return[{{id:'Sub-5K',name:'Sub-5K',lo:0,hi:s5,c:'#4ade80'}},{{id:'5K',name:'5K',lo:s5,hi:m5,c:'#f472b6'}},{{id:'10K',name:'10K',lo:m5,hi:mt,c:'#fb923c'}},{{id:'HM',name:'HM',lo:mt,hi:mh,c:'#a78bfa'}},{{id:'Mara',name:'Mara',lo:mh,hi:slow,c:'#38bdf8'}},{{id:'Other',name:'Other',lo:slow,hi:9999,c:'#4b5563'}}];}}
    const RACE_PW_Z=makeRacePwZones(),RACE_HR_Z=makeRaceHrZones(),RACE_PACE_Z=makeRacePaceZones();
    function zonesFor(m){{if(m==='hr')return HR_Z;if(m==='power')return PW_Z;if(m==='race'){{if(typeof currentMode!=='undefined'&&currentMode==='gap')return RACE_PACE_Z;return RACE_PW_Z;}}if(m==='racehr')return RACE_HR_Z;return HR_Z;}}
    function valFor(r,m){{if(m==='hr'||m==='racehr')return r.avg_hr;if(m==='race'&&typeof currentMode!=='undefined'&&currentMode==='gap')return r.avg_pace_skm;return r.npower;}}
    function isRaceMode(m){{return m==='race'||m==='racehr';}}
    function assignZ(val,zones,rm){{if(!val||val<=0)return zones[zones.length-1].id;if(rm){{for(const z of zones){{if(z.id==='Other')continue;if(val>=z.lo&&val<=z.hi)return z.id;}}return'Other';}}for(const z of zones){{if(val<z.hi)return z.id;}}return zones[zones.length-1].id;}}
    function assignToZone(val,zones,result,mins){{for(const z of zones){{if(z.id==='Other')continue;if(val>=z.lo){{result[z.id]+=mins;return;}}}}result['Other']+=mins;}}
    function estimateRaceEffortMins(r,zones){{const result={{}};zones.forEach(z=>result[z.id]=0);const mins=r.duration_min||0,pw=r.npower||0;if(!pw||!mins){{result['Other']=mins;return result;}}assignToZone(pw*1.10,zones,result,mins*0.15);assignToZone(pw,zones,result,mins*0.60);assignToZone(pw*0.88,zones,result,mins*0.25);return result;}}
    function estimateRaceEffortMinsHR(r,zones){{const result={{}};zones.forEach(z=>result[z.id]=0);const mins=r.duration_min||0,hr=r.avg_hr||0;if(!hr||!mins){{result['Other']=mins;return result;}}assignToZone(hr*1.06,zones,result,mins*0.15);assignToZone(hr,zones,result,mins*0.60);assignToZone(hr*0.92,zones,result,mins*0.25);return result;}}
    function getZoneMins(r,mode,zones){{
      // Use pre-computed NPZ zone data if available
      const key={{hr:'hz',power:'pz',race:'rpz',racehr:'rhz'}}[mode];
      if(r[key]){{const result={{}};zones.forEach(z=>result[z.id]=0);Object.keys(r[key]).forEach(k=>{{if(result[k]!==undefined)result[k]=r[key][k];}});return result;}}
      // Fallback to heuristic
      if(mode==='race')return estimateRaceEffortMins(r,zones);
      if(mode==='racehr')return estimateRaceEffortMinsHR(r,zones);
      // HR/Power zone fallback: assign all time to primary zone
      const result={{}};zones.forEach(z=>result[z.id]=0);const mins=r.duration_min||0,v=valFor(r,mode),zid=assignZ(v,zones,false);if(result[zid]!==undefined)result[zid]=mins;return result;
    }}
    function calcSpecificity(){{const today=new Date();const c14=new Date(today);c14.setDate(c14.getDate()-14);const c28=new Date(today);c28.setDate(c28.getDate()-28);const useGAP=(typeof currentMode!=='undefined'&&currentMode==='gap');const zones=useGAP?RACE_PACE_Z:RACE_PW_Z;const modeKey='race';const targets=[{{key:'5K'}},{{key:'HM'}}];targets.forEach(tgt=>{{let m14=0,m28=0;ZONE_RUNS.forEach(r=>{{const d=new Date(r.date);if(d<c28)return;const est=getZoneMins(r,modeKey,zones);const mins=est[tgt.key]||0;if(d>=c14)m14+=mins;m28+=mins;}});const e14=document.getElementById('spec14_'+tgt.key),e28=document.getElementById('spec28_'+tgt.key);if(e14)e14.innerHTML=Math.round(m14)+'<span style="font-size:0.75rem;color:var(--text-dim)">min</span>';if(e28)e28.innerHTML=Math.round(m28)+'<span style="font-size:0.75rem;color:var(--text-dim)">min</span>';}});}}calcSpecificity();
    // Weekly zone bars
    function weekKey(ds){{const d=new Date(ds),day=d.getDay(),m=new Date(d);m.setDate(d.getDate()-((day+6)%7));return m.toISOString().slice(0,10);}}
    function fmtWk(s){{const d=new Date(s),tmp=new Date(d.valueOf());tmp.setDate(tmp.getDate()+3-(tmp.getDay()+6)%7);const w1=new Date(tmp.getFullYear(),0,4);const wk=1+Math.round(((tmp-w1)/864e5-3+(w1.getDay()+6)%7)/7);return'W'+String(wk).padStart(2,'0')+'/'+String(tmp.getFullYear()).slice(-2);}}
    let wkMode='hr',wkN=8;
    function renderWk(){{const el=document.getElementById('wk-bars');el.innerHTML='';const zones=zonesFor(wkMode),rm=isRaceMode(wkMode);const weeks={{}};ZONE_RUNS.forEach(r=>{{const w=weekKey(r.date);if(!weeks[w])weeks[w]=[];weeks[w].push(r);}});const sorted=Object.keys(weeks).sort().slice(-wkN);let maxT=0;const wd=sorted.map(wk=>{{const zm={{}};zones.forEach(z=>zm[z.id]=0);let t=0;weeks[wk].forEach(r=>{{const mins=r.duration_min||0;const est=getZoneMins(r,wkMode,zones);Object.keys(est).forEach(zid=>{{if(zm[zid]!==undefined)zm[zid]+=est[zid];}});t+=mins;}});if(t>maxT)maxT=t;return{{week:wk,zm,t}};}});wd.forEach(w=>{{const row=document.createElement('div');row.className='wr';const lbl=document.createElement('div');lbl.className='wl';lbl.textContent=fmtWk(w.week);row.appendChild(lbl);const bar=document.createElement('div');bar.className='wb';zones.forEach(z=>{{if(w.zm[z.id]>0){{const pct=(w.zm[z.id]/maxT)*100,seg=document.createElement('div');seg.className='ws';seg.style.width=pct+'%';seg.style.background=z.c;seg.innerHTML='<div class="tip">'+z.name+': '+Math.round(w.zm[z.id])+' min</div>';bar.appendChild(seg);}}}});row.appendChild(bar);const tot=document.createElement('div');tot.className='wt';tot.textContent=Math.round(w.t)+' min';row.appendChild(tot);el.appendChild(row);}});document.getElementById('wk-leg').innerHTML=zones.map(z=>'<div class="lg"><div class="lsw" style="background:'+z.c+'"></div>'+z.name+'</div>').join('');}}
    function setWM(m,btn){{wkMode=m;document.querySelectorAll('#wk-mode button').forEach(b=>b.classList.remove('active'));btn.classList.add('active');renderWk();}}
    function setWP(n,btn){{wkN=n;document.querySelectorAll('#wk-period button').forEach(b=>b.classList.remove('active'));btn.classList.add('active');renderWk();}}
    renderWk();
    // Per-run chart
    let prMode='hr',prChart=null;
    function renderPR(){{const ctx=document.getElementById('prChart').getContext('2d');const last30=ZONE_RUNS.slice(-30),zones=zonesFor(prMode),rm=isRaceMode(prMode);const labels=last30.map(r=>{{const d=new Date(r.date);return d.getDate()+'/'+(d.getMonth()+1);}});const datasets=zones.map((z,zi)=>({{label:z.name,data:last30.map(r=>{{const mins=r.duration_min||0;const est=getZoneMins(r,prMode,zones);return Math.round(est[z.id]||0);}}),backgroundColor:z.c+'cc',borderWidth:0,borderRadius:2}}));if(prChart)prChart.destroy();prChart=new Chart(ctx,{{type:'bar',data:{{labels,datasets}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{title:items=>{{const i=items[0].dataIndex;return last30[i].name;}},label:item=>item.dataset.label+': '+item.raw+' min'}}}}}},scales:{{x:{{stacked:true,grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:'#8b90a0',font:{{size:10,family:"'JetBrains Mono'"}}}}}},y:{{stacked:true,grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:'#8b90a0',font:{{size:10,family:"'JetBrains Mono'"}},callback:v=>v+' min'}}}}}}}}}});document.getElementById('pr-leg').innerHTML=zones.map(z=>'<div class="lg"><div class="lsw" style="background:'+z.c+'"></div>'+z.name+'</div>').join('');}}
    function setPR(m,btn){{prMode=m;document.querySelectorAll('#pr-mode button').forEach(b=>b.classList.remove('active'));btn.classList.add('active');renderPR();}}
    renderPR();
    </script>
'''


# ============================================================================
# v51: ALERT BANNER GENERATOR
# ============================================================================
def _generate_alert_banner(alert_data, critical_power=None):
    """Generate dark-themed alert banner."""
    cp_html = f'<span class="power-only" id="banner-cp" style="font-size:0.82em;color:#8b90a0;">⚡ CP {critical_power}W</span>' if critical_power else ''
    if not alert_data:
        return f'<div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.3);border-radius:10px;padding:10px 16px;margin-bottom:14px;display:flex;justify-content:center;align-items:center;gap:8px;"><span style="font-size:1.1em;">🟢</span> <strong style="color:#4ade80;">All clear</strong> <span style="color:#8b90a0;">— no alerts</span>{cp_html}</div>'
    levels = [a['level'] for a in alert_data]
    if 'concern' in levels:
        bg, brd, icon, lc = 'rgba(239,68,68,0.08)', 'rgba(239,68,68,0.3)', '🔴', '#f87171'
    else:
        bg, brd, icon, lc = 'rgba(234,179,8,0.08)', 'rgba(234,179,8,0.3)', '🟡', '#fbbf24'
    items = ''.join(f'<div style="margin:4px 0;font-size:0.85em;color:#e4e7ef;">{a["icon"]} <strong>{a["name"]}</strong><span style="color:#8b90a0"> {a.get("detail","")}</span></div>' for a in alert_data)
    n = len(alert_data)
    s = "s" if n > 1 else ""
    return f'<div style="background:{bg};border:1px solid {brd};border-radius:10px;padding:10px 16px;margin-bottom:14px;"><div style="text-align:center;margin-bottom:4px;"><span style="font-size:1.1em;">{icon}</span> <strong style="color:{lc};">{n} alert{s} active</strong>{cp_html}</div>{items}</div>'


def generate_html(stats, rf_data, volume_data, ctl_atl_data, ctl_atl_lookup, rfl_trend_dates, rfl_trend_values, rfl_trendline, rfl_projection, rfl_ci_upper, rfl_ci_lower, alltime_rfl_dates, alltime_rfl_values, recent_runs, top_races, alert_data=None, weight_data=None, prediction_data=None, ag_data=None, zone_data=None):
    """Generate the HTML dashboard."""
    
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
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Paul Collyer Fitness Dashboard</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏃</text></svg>">
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
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
<body>
<script>Chart.defaults.color="#8b90a0";Chart.defaults.borderColor="rgba(255,255,255,0.04)";Chart.defaults.font.family="'DM Sans',sans-serif";Chart.defaults.plugins.legend.labels.padding=10;</script>
    <h1>🏃 Paul Collyer</h1>
    <div class="dash-sub">{datetime.now().strftime("%A %d %B %Y, %H:%M")}</div>
    
    <!-- Phase 2: Mode Toggle -->
    <div class="mode-toggle" style="display:flex;gap:6px;margin:10px 0 8px;align-items:center;">
        <span style="font-size:0.72rem;color:var(--text-dim);margin-right:4px;">Model:</span>
        <button class="mode-btn active" onclick="setMode('stryd')" id="mode-stryd">⚡ Stryd</button>
        <button class="mode-btn" onclick="setMode('gap')" id="mode-gap">🏃 GAP</button>
        <button class="mode-btn" onclick="setMode('sim')" id="mode-sim">🔬 Sim</button>
    </div>
    <style>
        .mode-btn {{ font-family:'DM Sans',sans-serif; font-size:0.75rem; padding:5px 12px; border-radius:16px;
            border:1px solid var(--border); background:var(--surface); color:var(--text-dim); cursor:pointer; transition:all 0.15s; }}
        .mode-btn:hover {{ border-color:var(--accent); color:var(--text); }}
        .mode-btn.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
        body.gap-mode .power-only {{ display:none !important; }}
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
            <div class="stat-value" id="rfl-value">{stats['latest_rfl']}%</div>
            <div class="stat-label" id="rfl-label">RFL Trend</div>
            <div class="stat-sub">vs peak</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="rfl-delta">{f"{'+' if stats['rfl_14d_delta'] > 0 else ''}{stats['rfl_14d_delta']}%" if stats['rfl_14d_delta'] is not None else '-'}</div>
            <div class="stat-label">RFL 14d</div>
            <div class="stat-sub">change</div>
        </div>
        <div class="stat-card power-only">
            <div class="stat-value">{f"{'+' if stats['easy_rfl_gap'] > 0 else ''}{stats['easy_rfl_gap']}%" if stats['easy_rfl_gap'] is not None else '-'}</div>
            <div class="stat-label">Easy RF Gap</div>
            <div class="stat-sub">vs trend</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="ag-value">{stats['age_grade'] if stats['age_grade'] else '-'}%</div>
            <div class="stat-label">Age Grade</div>
            <div class="stat-sub">5k estimate</div>
        </div>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value" id="pred-5k">{format_race_time(stats['race_predictions'].get('5k', '-'))}</div>
            <div class="stat-label">5k</div>
            <div class="stat-sub">predicted</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="pred-10k">{format_race_time(stats['race_predictions'].get('10k', '-'))}</div>
            <div class="stat-label">10k</div>
            <div class="stat-sub">predicted</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="pred-hm">{format_race_time(stats['race_predictions'].get('Half Marathon', '-'))}</div>
            <div class="stat-label">Half</div>
            <div class="stat-sub">predicted</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="pred-mara">{format_race_time(stats['race_predictions'].get('Marathon', '-'))}</div>
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
            <div class="stat-sub">since 2013</div>
        </div>
    </div>
    
    <!-- RF Trend Chart -->
    
    <!-- Training Zones Section -->
    {_generate_zone_html(zone_data)}
    
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
    
    <!-- v51: Weight Chart -->
    <div class="chart-container">
        <div class="chart-header">
            <div class="chart-title">⚖️ Weight</div>
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
    
    <!-- v51: Race Prediction Trend Chart -->
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
            <div class="chart-desc">Blue = predicted. <span style="color:#4ade80">Green</span> = adjusted for temp &amp; surface. <span style="color:#ef4444">●</span> race <span style="color:#fca5a5">●</span> parkrun.</div>
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
    
    <!-- v51: Age Grade Trend Chart -->
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
                <col class="power-only" style="width: 9%;">
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
                    <th class="power-only">TSS</th>
                    <th>RFL%</th>
                </tr>
            </thead>
            <tbody id="recentRunsBody">
            </tbody>
        </table>
        </div>
    </div>
    
    <!-- Top Races Section -->
    <div class="chart-container">
        <div class="chart-title-row">
            <span class="chart-title">🏆 Top Race Performances</span>
            <div class="chart-toggle" id="topRacesToggle">
                <button class="active" data-period="1y">1Y</button>
                <button data-period="2y">2Y</button>
                <button data-period="3y">3Y</button>
                <button data-period="5y">5Y</button>
                <button data-period="all">All</button>
            </div>
        </div>
        <div class="chart-desc" id="top-races-desc">Best races by RFL% (fitness level at race date).</div>
        <div class="table-wrapper">
        <table id="topRacesTable">
            <colgroup>
                <col style="width: 15%;">
                <col style="width: 35%;">
                <col style="width: 12%;">
                <col style="width: 16%;">
                <col style="width: 11%;">
                <col style="width: 11%;">
            </colgroup>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Race</th>
                    <th>Dist</th>
                    <th>Time</th>
                    <th>AG%</th>
                    <th>RFL%</th>
                </tr>
            </thead>
            <tbody id="topRacesBody">
                <!-- Populated by JavaScript -->
            </tbody>
        </table>
        </div>
    </div>
    
    <div class="footer">
        Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}<br>
        Data through: {stats['data_date']}
    </div>
    
    <script>
        // RFL 14-day Trend Chart with trendline, 7-day projection and 95% CI
        const rflTrendCtx = document.getElementById('rflTrendChart').getContext('2d');
        new Chart(rflTrendCtx, {{
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
                    }}
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
        let currentMode = 'stryd';
        
        // Phase 2: Mode data for stats switching
        const modeStats = {{
            stryd: {{ rfl: '{stats["latest_rfl"]}', ag: '{stats["age_grade"] or "-"}',
                pred5k: '{format_race_time(stats["race_predictions"].get("5k", "-"))}',
                pred10k: '{format_race_time(stats["race_predictions"].get("10k", "-"))}',
                predHm: '{format_race_time(stats["race_predictions"].get("Half Marathon", "-"))}',
                predMara: '{format_race_time(stats["race_predictions"].get("Marathon", "-"))}' }},
            gap: {{ rfl: '{stats.get("latest_rfl_gap", "-")}', ag: '{stats["race_predictions"].get("_ag_gap") or "-"}',
                pred5k: '{format_race_time(stats["race_predictions"].get("_mode_gap", dict()).get("5k", "-"))}',
                pred10k: '{format_race_time(stats["race_predictions"].get("_mode_gap", dict()).get("10k", "-"))}',
                predHm: '{format_race_time(stats["race_predictions"].get("_mode_gap", dict()).get("Half Marathon", "-"))}',
                predMara: '{format_race_time(stats["race_predictions"].get("_mode_gap", dict()).get("Marathon", "-"))}' }},
            sim: {{ rfl: '{stats.get("latest_rfl_sim", "-")}', ag: '{stats["race_predictions"].get("_ag_sim") or "-"}',
                pred5k: '{format_race_time(stats["race_predictions"].get("_mode_sim", dict()).get("5k", "-"))}',
                pred10k: '{format_race_time(stats["race_predictions"].get("_mode_sim", dict()).get("10k", "-"))}',
                predHm: '{format_race_time(stats["race_predictions"].get("_mode_sim", dict()).get("Half Marathon", "-"))}',
                predMara: '{format_race_time(stats["race_predictions"].get("_mode_sim", dict()).get("Marathon", "-"))}' }}
        }};
        
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
                                const ds = chart.data.datasets;
                                return [
                                    {{ text: 'RFL', fillStyle: 'rgba(129, 140, 248, 0.4)', strokeStyle: 'rgba(129, 140, 248, 0.3)', pointStyle: 'circle', hidden: !chart.isDatasetVisible(0), datasetIndex: 0, fontColor: '#8b90a0' }},
                                    {{ text: 'RFL Trend', fillStyle: 'rgba(129, 140, 248, 1)', strokeStyle: 'rgba(129, 140, 248, 1)', pointStyle: 'circle', hidden: !chart.isDatasetVisible(1), datasetIndex: 1, fontColor: '#8b90a0' }},
                                    {{ text: 'Easy RF', fillStyle: 'rgba(34, 197, 94, 0.8)', strokeStyle: 'rgba(34, 197, 94, 0.8)', pointStyle: 'circle', hidden: !chart.isDatasetVisible(2), datasetIndex: 2, fontColor: '#8b90a0' }}
                                ];
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
                    }}
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
        
        function updateTopRacesTable(period) {{
            const tbody = document.getElementById('topRacesBody');
            const races = (topRacesData[period] || []).slice();
            
            if (races.length === 0) {{
                tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-dim);">No races in this period</td></tr>';
                return;
            }}
            
            // Sort by mode-appropriate RFL descending
            const rflKey = currentMode === 'gap' ? 'rfl_gap' : currentMode === 'sim' ? 'rfl_sim' : 'rfl';
            races.sort((a, b) => (b[rflKey] || 0) - (a[rflKey] || 0));
            
            tbody.innerHTML = races.map((race, idx) => {{
                const rflVal = race[rflKey] || '-';
                return `
                <tr>
                    <td>${{race.date}}</td>
                    <td>${{race.name}}</td>
                    <td>${{race.dist ? race.dist + ' km' : '-'}}</td>
                    <td>${{race.time}}</td>
                    <td>${{race.ag ? race.ag + '%' : '-'}}</td>
                    <td>${{rflVal}}</td>
                </tr>`;
            }}).join('');
        }}
        
        // Initialize with 1 year view
        updateTopRacesTable('1y');
        
        // Toggle handler
        document.getElementById('topRacesToggle').addEventListener('click', function(e) {{
            if (e.target.tagName === 'BUTTON') {{
                this.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
                e.target.classList.add('active');
                const period = e.target.getAttribute('data-period');
                updateTopRacesTable(period);
            }}
        }});
        
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
            const predicted = indices.map(i => d.predicted[i]);
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
            
            // Compute stable y-axis range covering both adjusted and unadjusted data
            const allYVals = [];
            for (let i = 0; i < actual.length; i++) {{ if (actual[i]) allYVals.push(actual[i]); }}
            for (let i = 0; i < predicted.length; i++) {{ if (predicted[i]) allYVals.push(predicted[i]); }}
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
                    datasets: [{{
                        label: adjustConditions ? 'Predicted (adj)' : 'Predicted',
                        data: predPoints,
                        borderColor: adjustConditions ? 'rgba(74, 222, 128, 0.7)' : 'rgba(129, 140, 248, 0.7)',
                        backgroundColor: adjustConditions ? 'rgba(74, 222, 128, 0.05)' : 'rgba(129, 140, 248, 0.05)',
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: true,
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
                                    // Gap (use displayPredicted for adjusted comparison)
                                    if (displayPredicted[i] && actual[i]) {{
                                        const gap = actual[i] - displayPredicted[i];
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
                // v51: Trend mover arrow
                let deltaHtml = '';
                if (run.delta !== null && run.delta !== undefined) {{
                    if (run.delta >= 1.0) deltaHtml = '<span style="color:#16a34a;font-weight:bold;" title="RFL +' + run.delta.toFixed(1) + '%"> ▲▲</span>';
                    else if (run.delta >= 0.3) deltaHtml = '<span style="color:#16a34a;" title="RFL +' + run.delta.toFixed(1) + '%"> ▲</span>';
                    else if (run.delta <= -1.0) deltaHtml = '<span style="color:#dc2626;font-weight:bold;" title="RFL ' + run.delta.toFixed(1) + '%"> ▼▼</span>';
                    else if (run.delta <= -0.3) deltaHtml = '<span style="color:#dc2626;" title="RFL ' + run.delta.toFixed(1) + '%"> ▼</span>';
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
                    <td class="power-only">${{run.tss || '-'}}</td>
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
        
        // Update RFL label
        const rflLabel = document.getElementById('rfl-label');
        if (rflLabel) {{
            const labels = {{ stryd: 'RFL Trend', gap: 'RFL (GAP)', sim: 'RFL (Sim)' }};
            rflLabel.textContent = labels[mode];
        }}
        
        // Hide/show power-only elements via body class (CSS handles all hiding)
        const isGap = mode === 'gap';
        document.body.classList.toggle('gap-mode', isGap);
        
        // Update CP in alert banner for Stryd/Sim modes
        const cpBanner = document.getElementById('banner-cp');
        if (cpBanner) {{
            const cpVal = mode === 'sim' ? modeStats.sim : modeStats.stryd;
            cpBanner.textContent = '⚡ CP ' + Math.round(372 * parseFloat(cpVal.rfl) / 100) + 'W';
        }}
        
        // Update zone badge (hide power info in GAP mode, use mode CP for Sim)
        const zb = document.getElementById('zone-badge');
        if (zb && typeof ZONE_CP !== 'undefined') {{
            if (isGap) {{
                zb.textContent = 'LTHR ' + ZONE_LTHR + ' · Max ~' + ZONE_MAXHR;
            }} else {{
                const modeCP = mode === 'sim' ? Math.round(ZONE_PEAK_CP * parseFloat(modeStats.sim.rfl) / 100) : ZONE_CP;
                const modeLT = Math.round(modeCP * 0.956);
                zb.textContent = 'LT ' + modeLT + 'W · CP ' + modeCP + 'W · LTHR ' + ZONE_LTHR + ' · Max ~' + ZONE_MAXHR;
            }}
        }}
        
        // Update top races description
        const trd = document.getElementById('top-races-desc');
        if (trd) trd.textContent = 'Best races by RFL% (fitness level at race date).';
        
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
            // Re-render current zone views (zones now mode-aware)
            if (typeof renderWk === 'function') renderWk();
            if (typeof renderPR === 'function') renderPR();
        }}
        
        // Recalculate race readiness specificity (HR-based for GAP, power for others)
        if (typeof calcSpecificity === 'function') calcSpecificity();
        
        // Re-render recent runs and top races to apply power-only visibility
        if (typeof updateRecentRunsTable === 'function') {{
            const activeFilter = document.querySelector('#recentRunsToggle button.active');
            updateRecentRunsTable(activeFilter ? activeFilter.getAttribute('data-filter') : 'all');
        }}
        if (typeof updateTopRacesTable === 'function') {{
            const activePeriod = document.querySelector('#topRacesToggle button.active');
            updateTopRacesTable(activePeriod ? activePeriod.getAttribute('data-period') : '1y');
        }}
    }}
    
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
    rfl_trend_dates, rfl_trend_values, rfl_trendline, rfl_projection, rfl_ci_upper, rfl_ci_lower = get_daily_rfl_trend(MASTER_FILE, days=14)
    
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
    html = generate_html(stats, rf_data, volume_data, ctl_atl_data, ctl_atl_lookup, rfl_trend_dates, rfl_trend_values, rfl_trendline, rfl_projection, rfl_ci_upper, rfl_ci_lower, alltime_rfl_dates, alltime_rfl_values, recent_runs, top_races, alert_data=alert_data, weight_data=weight_data, prediction_data=prediction_data, ag_data=ag_data, zone_data=zone_data)
    
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
