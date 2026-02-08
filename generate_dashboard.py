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
                    if remainder and ',' in remainder and not remainder.startswith('#'):
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
        return [], [], [], [], []
    
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
    
    return dates, rfl_values, rfl_trend, easy_rfl, race_flags

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
                'date': row['date'].strftime('%d %b'),
                'name': str(name_val) if name_val else '',
                'dist': round(dist_km, 1) if dist_km > 0 else None,
                'pace': pace_str,
                'npower': npower_val,
                'hr': hr_val,
                'tss': tss_val,
                'rfl': rfl_val,
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
    """v51.6: Get weekly average weight data for chart.
    
    Master weight_kg is already smoothed (±3 day centred average from athlete_data).
    Dashboard simply computes weekly averages.
    """
    wt_df = None
    
    # Try Master first
    try:
        df = pd.read_excel(master_file, sheet_name='Master', usecols=['date', 'weight_kg'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['weight_kg'])
        if len(df) > 0:
            wt_df = df[['date', 'weight_kg']]
    except Exception:
        pass
    
    # Fall back to athlete_data.csv
    if wt_df is None or len(wt_df) == 0:
        athlete_file = os.path.join(os.path.dirname(master_file), 'athlete_data.csv')
        if not os.path.exists(athlete_file):
            athlete_file = 'athlete_data.csv'
        if os.path.exists(athlete_file):
            try:
                import io
                with open(athlete_file, 'r') as f:
                    raw_lines = f.readlines()
                clean_lines = []
                for line in raw_lines:
                    stripped = line.lstrip()
                    if stripped.startswith('#'):
                        last_hash = stripped.rfind('#')
                        remainder = stripped[last_hash + 1:].strip()
                        if remainder and ',' in remainder:
                            clean_lines.append(remainder + '\n')
                    else:
                        clean_lines.append(line)
                ad = pd.read_csv(io.StringIO(''.join(clean_lines)))
                ad['date'] = pd.to_datetime(ad['date'], format='%Y-%m-%d', errors='coerce')
                if 'weight_kg' in ad.columns:
                    wt_df = ad[['date', 'weight_kg']].dropna(subset=['weight_kg'])
            except Exception as e:
                print(f"  Warning: Could not load weight from athlete_data.csv: {e}")
    
    if wt_df is None or len(wt_df) == 0:
        return [], []
    
    try:
        today = datetime.now()
        cutoff = today - timedelta(days=months * 30)
        wt_df = wt_df[wt_df['date'] > cutoff].copy()
        
        # Simple weekly averages (data is already smoothed upstream)
        wt_df['week'] = wt_df['date'].dt.to_period('W').apply(lambda p: p.start_time)
        weekly = wt_df.groupby('week')['weight_kg'].mean().reset_index()
        weekly = weekly.sort_values('week')
        
        dates = weekly['week'].dt.strftime('%d %b %y').tolist()
        values = [round(v, 1) for v in weekly['weight_kg'].tolist()]
        
        return dates, values
    except Exception as e:
        print(f"  Warning: Could not process weight data: {e}")
        return [], []


def get_top_races(df, n=10):
    """Get top race performances by Power Score for different time periods.
    
    Returns dict with keys: '1y', '2y', '3y', '5y', 'all'
    Each value is a list of up to n race dicts sorted by Power_Score descending.
    """
    # Filter to races only
    race_col = 'race_flag' if 'race_flag' in df.columns else 'Race'
    races = df[df[race_col] == 1].copy()
    
    if len(races) == 0:
        return {period: [] for period in ['1y', '2y', '3y', '5y', 'all']}
    
    # Need Power_Score column
    if 'Power_Score' not in races.columns:
        print("  Warning: Power_Score column not found, using RF_adj for top races")
        score_col = 'RF_adj' if 'RF_adj' in races.columns else None
        if score_col is None:
            return {period: [] for period in ['1y', '2y', '3y', '5y', 'all']}
    else:
        score_col = 'Power_Score'
    
    # Filter out rows without valid scores
    races = races[races[score_col].notna()].copy()
    
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
        
        # Power Score
        ps_val = round(row[score_col], 1) if pd.notna(row.get(score_col)) else None
        
        # Age grade percentage
        ag_val = round(row['age_grade_pct'], 1) if pd.notna(row.get('age_grade_pct')) else None
        
        return {
            'date': row['date'].strftime('%d %b %y'),
            'name': str(name_val) if name_val else '',
            'dist': round(dist_km, 1) if dist_km > 0 else None,
            'time': time_str,
            'ps': ps_val,
            'ag': ag_val,
        }
    
    result = {}
    for period_name, delta in periods.items():
        if delta is None:
            period_races = races
        else:
            cutoff = now - delta
            period_races = races[races['date'] >= cutoff]
        
        # Sort by Power Score descending, take top n
        top = period_races.nlargest(n, score_col)
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
        empty = {'dates': [], 'dates_iso': [], 'predicted': [], 'actual': [], 'is_parkrun': [], 'names': [], 'temps': [], 'surfaces': []}
        
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
        }
    
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
        '5k': '#2563eb',       # blue
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
# v51: ALERT BANNER GENERATOR
# ============================================================================
def _generate_alert_banner(alert_data, critical_power=None):
    """Generate the HTML for the health check banner with CP."""
    cp_html = f'<span style="float: right; font-size: 0.9em; color: #374151;"><strong>⚡ CP {critical_power}W</strong></span>' if critical_power else ''
    
    if not alert_data:
        return f'''<div style="background: #f0fdf4; border: 1px solid #86efac; border-radius: 12px; padding: 10px 16px; margin-bottom: 12px; text-align: center;">
            <span style="font-size: 1.1em;">🟢</span> <strong>All clear</strong> — no alerts active{cp_html}
        </div>'''
    
    # Determine overall level
    levels = [a['level'] for a in alert_data]
    if 'concern' in levels:
        bg = '#fef2f2'; border = '#fca5a5'; icon = '🔴'
    else:
        bg = '#fffbeb'; border = '#fcd34d'; icon = '🟡'
    
    items = ''.join(
        f'<div style="margin: 4px 0; font-size: 0.85em;">{a["icon"]} <strong>{a["name"]}</strong>{a.get("detail","")}</div>'
        for a in alert_data
    )
    return f'''<div style="background: {bg}; border: 1px solid {border}; border-radius: 12px; padding: 10px 16px; margin-bottom: 12px;">
        <div style="text-align: center; margin-bottom: 4px;"><span style="font-size: 1.1em;">{icon}</span> <strong>{len(alert_data)} alert{"s" if len(alert_data) > 1 else ""} active</strong>{cp_html}</div>
        {items}
    </div>'''


# ============================================================================
# HTML GENERATION
# ============================================================================
def generate_html(stats, rf_data, volume_data, ctl_atl_data, ctl_atl_lookup, rfl_trend_dates, rfl_trend_values, rfl_trendline, rfl_projection, rfl_ci_upper, rfl_ci_lower, alltime_rfl_dates, alltime_rfl_values, recent_runs, top_races, alert_data=None, weight_data=None, prediction_data=None, ag_data=None):
    """Generate the HTML dashboard."""
    
    # Extract data for each time range - RF (v51: now includes easy_rfl and race_flags)
    rf_dates_90, rf_values_90, rf_trend_90, rf_easy_90, rf_races_90 = rf_data['90']
    rf_dates_180, rf_values_180, rf_trend_180, rf_easy_180, rf_races_180 = rf_data['180']
    rf_dates_365, rf_values_365, rf_trend_365, rf_easy_365, rf_races_365 = rf_data['365']
    rf_dates_730, rf_values_730, rf_trend_730, rf_easy_730, rf_races_730 = rf_data['730']
    rf_dates_1095, rf_values_1095, rf_trend_1095, rf_easy_1095, rf_races_1095 = rf_data['1095']
    rf_dates_1825, rf_values_1825, rf_trend_1825, rf_easy_1825, rf_races_1825 = rf_data['1825']
    
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
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
    <style>
        * {{
            box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 12px;
            background: #f5f5f5;
            color: #333;
            font-size: 14px;
        }}
        
        h1 {{
            font-size: 1.5em;
            margin: 0 0 12px 0;
            text-align: center;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 16px;
        }}
        
        .stat-card {{
            background: white;
            border-radius: 12px;
            padding: 12px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .stat-value {{
            font-size: 1.6em;
            font-weight: bold;
            color: #2563eb;
        }}
        
        .stat-label {{
            font-size: 0.8em;
            color: #666;
            margin-top: 4px;
        }}
        
        .stat-sub {{
            font-size: 0.75em;
            color: #999;
        }}
        
        .chart-container {{
            background: white;
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .chart-title {{
            font-size: 1em;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        
        .chart-title-row {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
        }}
        
        .chart-title-row .chart-title {{
            margin-bottom: 0;
        }}
        
        .chart-title-row .chart-toggle {{
            flex-shrink: 0;
        }}
        
        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }}
        
        .chart-toggle {{
            display: flex;
            gap: 4px;
        }}
        
        .chart-toggle button {{
            padding: 3px 8px;
            font-size: 0.7em;
            border: 1px solid #ddd;
            background: #f5f5f5;
            border-radius: 4px;
            cursor: pointer;
            color: #666;
        }}
        
        .chart-toggle button.active {{
            background: #2563eb;
            border-color: #2563eb;
            color: white;
        }}
        
        .chart-desc {{
            font-size: 0.75em;
            color: #666;
            margin-bottom: 10px;
            line-height: 1.3;
        }}
        
        .chart-wrapper {{
            position: relative;
            height: 200px;
            width: 100%;
        }}
        
        /* Table wrapper for horizontal scroll on mobile */
        .table-wrapper {{
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            margin: 0 -12px;
            padding: 0 12px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8em;
            table-layout: fixed;
            min-width: 500px;  /* Force scroll on narrow screens */
        }}
        
        th, td {{
            padding: 6px 3px;
            text-align: left;
            border-bottom: 1px solid #eee;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        /* Activity/Race name column - allow wrapping */
        td:nth-child(2), td:nth-child(3) {{
            white-space: normal;
            word-wrap: break-word;
            line-height: 1.2;
        }}
        
        th {{
            font-weight: 600;
            color: #666;
            font-size: 0.7em;
            text-transform: uppercase;
            white-space: nowrap;
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
            font-size: 0.75em;
            color: #999;
            margin-top: 16px;
            padding-top: 12px;
            border-top: 1px solid #ddd;
        }}
        
        /* Larger screens */
        @media (min-width: 600px) {{
            body {{
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .stats-grid {{
                grid-template-columns: repeat(4, 1fr);
            }}
            
            .chart-wrapper {{
                height: 280px;
            }}
        }}
    </style>
</head>
<body>
    <h1>🏃 Paul Collyer Fitness Dashboard</h1>
    <div style="text-align: center; color: #666; font-size: 0.85em; margin-bottom: 12px;">
        {datetime.now().strftime('%A %d %B %Y, %H:%M')}
    </div>
    
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
            <div class="stat-value">{stats['latest_rfl']}%</div>
            <div class="stat-label">RFL Trend</div>
            <div class="stat-sub">vs peak</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{f"{'+' if stats['rfl_14d_delta'] > 0 else ''}{stats['rfl_14d_delta']}%" if stats['rfl_14d_delta'] is not None else '-'}</div>
            <div class="stat-label">RFL 14d</div>
            <div class="stat-sub">change</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{f"{'+' if stats['easy_rfl_gap'] > 0 else ''}{stats['easy_rfl_gap']}%" if stats['easy_rfl_gap'] is not None else '-'}</div>
            <div class="stat-label">Easy RF Gap</div>
            <div class="stat-sub">vs trend</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['age_grade'] if stats['age_grade'] else '-'}%</div>
            <div class="stat-label">Age Grade</div>
            <div class="stat-sub">5k estimate</div>
        </div>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{format_race_time(stats['race_predictions'].get('5k', '-'))}</div>
            <div class="stat-label">5k</div>
            <div class="stat-sub">predicted</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{format_race_time(stats['race_predictions'].get('10k', '-'))}</div>
            <div class="stat-label">10k</div>
            <div class="stat-sub">predicted</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{format_race_time(stats['race_predictions'].get('Half Marathon', '-'))}</div>
            <div class="stat-label">Half</div>
            <div class="stat-sub">predicted</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{format_race_time(stats['race_predictions'].get('Marathon', '-'))}</div>
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
    <div class="chart-container">
        <div class="chart-header">
            <div class="chart-title">📈 Relative Fitness Level</div>
            <div class="chart-toggle" id="rfToggle">
                <button class="active" data-range="90">90d</button>
                <button data-range="180">6m</button>
                <button data-range="365">1yr</button>
                <button data-range="730">2yr</button>
                <button data-range="1095">3yr</button>
                <button data-range="1825">5yr</button>
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
        <div class="chart-wrapper" style="height: 150px;">
            <canvas id="rflTrendChart"></canvas>
        </div>
    </div>
    
    <!-- All-time Weekly RFL Chart -->
    <div class="chart-container">
        <div class="chart-title">📊 Relative Fitness Level (all time)</div>
        <div class="chart-desc">Weekly peak RFL since 2013. Shows long-term fitness progression and seasonal patterns.</div>
        <div class="chart-wrapper">
            <canvas id="alltimeRflChart"></canvas>
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
            <div class="chart-desc">Blue = predicted. <span style="color:#ef4444">●</span> race <span style="color:#fca5a5">●</span> parkrun.</div>
            <label style="font-size: 0.75em; color: #666; white-space: nowrap; cursor: pointer;">
                <input type="checkbox" id="predParkrunToggle" checked style="margin-right: 3px;">parkruns
            </label>
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
                <col style="width: 10%;">
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
                    <th>nPwr</th>
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
        <div class="chart-desc">Best races by Power Score (duration-normalised power, heat-adjusted).</div>
        <div class="table-wrapper">
        <table id="topRacesTable">
            <colgroup>
                <col style="width: 5%;">
                <col style="width: 15%;">
                <col style="width: 32%;">
                <col style="width: 12%;">
                <col style="width: 14%;">
                <col style="width: 11%;">
                <col style="width: 11%;">
            </colgroup>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Date</th>
                    <th>Race</th>
                    <th>Dist</th>
                    <th>Time</th>
                    <th>PS</th>
                    <th>AG%</th>
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
                    borderColor: 'rgba(37, 99, 235, 1)',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 2,
                    pointRadius: 3,
                    pointBackgroundColor: 'rgba(37, 99, 235, 1)',
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
                    borderColor: 'rgba(239, 68, 68, 0.3)',
                    borderWidth: 1,
                    borderDash: [2, 2],
                    pointRadius: 0,
                    fill: false,
                    tension: 0,
                    order: 4,
                }}, {{
                    label: '95% CI lower',
                    data: {json.dumps(rfl_ci_lower)},
                    borderColor: 'rgba(239, 68, 68, 0.3)',
                    borderWidth: 1,
                    borderDash: [2, 2],
                    pointRadius: 0,
                    fill: '-1',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
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
                        position: 'top',
                        labels: {{
                            boxWidth: 10,
                            padding: 4,
                            font: {{ size: 8 }},
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
                            font: {{ size: 9 }}
                        }}
                    }},
                    y: {{
                        display: true,
                        grace: '10%',
                        ticks: {{
                            font: {{ size: 9 }},
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
            '90': {{ dates: {json.dumps(rf_dates_90)}, values: {json.dumps(rf_values_90)}, trend: {json.dumps(rf_trend_90)}, easy: {json.dumps(rf_easy_90)}, races: {json.dumps(rf_races_90)} }},
            '180': {{ dates: {json.dumps(rf_dates_180)}, values: {json.dumps(rf_values_180)}, trend: {json.dumps(rf_trend_180)}, easy: {json.dumps(rf_easy_180)}, races: {json.dumps(rf_races_180)} }},
            '365': {{ dates: {json.dumps(rf_dates_365)}, values: {json.dumps(rf_values_365)}, trend: {json.dumps(rf_trend_365)}, easy: {json.dumps(rf_easy_365)}, races: {json.dumps(rf_races_365)} }},
            '730': {{ dates: {json.dumps(rf_dates_730)}, values: {json.dumps(rf_values_730)}, trend: {json.dumps(rf_trend_730)}, easy: {json.dumps(rf_easy_730)}, races: {json.dumps(rf_races_730)} }},
            '1095': {{ dates: {json.dumps(rf_dates_1095)}, values: {json.dumps(rf_values_1095)}, trend: {json.dumps(rf_trend_1095)}, easy: {json.dumps(rf_easy_1095)}, races: {json.dumps(rf_races_1095)} }},
            '1825': {{ dates: {json.dumps(rf_dates_1825)}, values: {json.dumps(rf_values_1825)}, trend: {json.dumps(rf_trend_1825)}, easy: {json.dumps(rf_easy_1825)}, races: {json.dumps(rf_races_1825)} }}
        }};
        
        // v51: Generate per-point colours (red for races, blue for training)
        function racePointColors(races, baseColor, raceColor) {{
            return races.map(r => r === 1 ? raceColor : baseColor);
        }}
        
        const rfCtx = document.getElementById('rfChart').getContext('2d');
        let rfChart = new Chart(rfCtx, {{
            type: 'line',
            data: {{
                labels: rfData['90'].dates,
                datasets: [{{
                    label: 'RFL',
                    data: rfData['90'].values,
                    borderColor: 'rgba(37, 99, 235, 0.3)',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 1,
                    pointRadius: 3,
                    pointBackgroundColor: racePointColors(rfData['90'].races, 'rgba(37, 99, 235, 0.5)', 'rgba(239, 68, 68, 0.9)'),
                    pointBorderColor: racePointColors(rfData['90'].races, 'rgba(37, 99, 235, 0.3)', 'rgba(239, 68, 68, 0.7)'),
                    pointBorderWidth: 1,
                    fill: false,
                    tension: 0,
                }}, {{
                    label: 'RFL Trend',
                    data: rfData['90'].trend,
                    borderColor: 'rgba(37, 99, 235, 1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3,
                }}, {{
                    label: 'Easy RF',
                    data: rfData['90'].easy,
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
                        position: 'top',
                        labels: {{ boxWidth: 12, padding: 8, font: {{ size: 11 }} }}
                    }}
                }},
                scales: {{
                    x: {{ display: true, ticks: {{ maxTicksLimit: 5, maxRotation: 0, font: {{ size: 9 }} }} }},
                    y: {{ 
                        display: true, 
                        ticks: {{ 
                            font: {{ size: 10 }},
                            callback: function(value) {{ return value + '%'; }}
                        }},
                        suggestedMin: 70,
                        suggestedMax: 100
                    }}
                }}
            }}
        }});
        
        document.getElementById('rfToggle').addEventListener('click', function(e) {{
            if (e.target.tagName === 'BUTTON') {{
                const range = e.target.dataset.range;
                this.querySelectorAll('button').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                rfChart.data.labels = rfData[range].dates;
                rfChart.data.datasets[0].data = rfData[range].values;
                rfChart.data.datasets[1].data = rfData[range].trend;
                rfChart.data.datasets[2].data = rfData[range].easy;
                
                // Adjust point size and opacity based on data density
                const pointSettings = {{
                    '90': {{ radius: 3, borderColor: 'rgba(37, 99, 235, 0.3)', pointColor: 'rgba(37, 99, 235, 0.5)' }},
                    '180': {{ radius: 2, borderColor: 'rgba(37, 99, 235, 0.25)', pointColor: 'rgba(37, 99, 235, 0.4)' }},
                    '365': {{ radius: 1.5, borderColor: 'rgba(37, 99, 235, 0.2)', pointColor: 'rgba(37, 99, 235, 0.3)' }},
                    '730': {{ radius: 1, borderColor: 'rgba(37, 99, 235, 0.15)', pointColor: 'rgba(37, 99, 235, 0.25)' }},
                    '1095': {{ radius: 0.5, borderColor: 'rgba(37, 99, 235, 0.1)', pointColor: 'rgba(37, 99, 235, 0.2)' }},
                    '1825': {{ radius: 0, borderColor: 'rgba(37, 99, 235, 0.1)', pointColor: 'rgba(37, 99, 235, 0.15)' }}
                }};
                const settings = pointSettings[range];
                const races = rfData[range].races;
                rfChart.data.datasets[0].pointRadius = settings.radius;
                rfChart.data.datasets[0].pointBackgroundColor = racePointColors(races, settings.pointColor, 'rgba(239, 68, 68, 0.9)');
                rfChart.data.datasets[0].pointBorderColor = racePointColors(races, settings.borderColor, 'rgba(239, 68, 68, 0.7)');
                rfChart.data.datasets[0].borderColor = settings.borderColor;
                
                rfChart.update();
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
                    borderColor: 'rgba(37, 99, 235, 1)',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3,
                }}, {{
                    label: 'ATL (Fatigue)',
                    data: ctlAtlData['90'].atl,
                    borderColor: 'rgba(239, 68, 68, 1)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3,
                }}, {{
                    label: 'CTL Projection',
                    data: ctlAtlData['90'].ctlProj,
                    borderColor: 'rgba(37, 99, 235, 0.5)',
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
                        position: 'top',
                        labels: {{ 
                            boxWidth: 12, 
                            padding: 8, 
                            font: {{ size: 11 }},
                            filter: function(item) {{
                                // Hide projection labels from legend
                                return !item.text.includes('Projection');
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{ display: true, ticks: {{ maxTicksLimit: 5, maxRotation: 0, font: {{ size: 9 }} }} }},
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
                    backgroundColor: 'rgba(37, 99, 235, 0.7)',
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
                        color: '#333',
                        font: {{ size: 9, weight: 'bold' }},
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
        const alltimeRflCtx = document.getElementById('alltimeRflChart').getContext('2d');
        new Chart(alltimeRflCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(alltime_rfl_dates)},
                datasets: [{{
                    label: 'RFL vs peak',
                    data: {json.dumps(alltime_rfl_values)},
                    borderColor: 'rgba(37, 99, 235, 1)',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.3,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'top',
                        labels: {{
                            boxWidth: 10,
                            padding: 4,
                            font: {{ size: 9 }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        display: true,
                        ticks: {{
                            maxTicksLimit: 12,
                            font: {{ size: 8 }}
                        }}
                    }},
                    y: {{
                        display: true,
                        min: 50,
                        max: 100,
                        ticks: {{
                            font: {{ size: 9 }},
                            callback: function(value) {{ return value + '%'; }}
                        }}
                    }}
                }}
            }}
        }});
        
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
            const races = topRacesData[period] || [];
            
            if (races.length === 0) {{
                tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: #666;">No races in this period</td></tr>';
                return;
            }}
            
            tbody.innerHTML = races.map((race, idx) => `
                <tr>
                    <td>${{idx + 1}}</td>
                    <td>${{race.date}}</td>
                    <td>${{race.name}}</td>
                    <td>${{race.dist ? race.dist + ' km' : '-'}}</td>
                    <td>${{race.time}}</td>
                    <td>${{race.ps || '-'}}</td>
                    <td>${{race.ag ? race.ag + '%' : '-'}}</td>
                </tr>
            `).join('');
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
                        x: {{ display: true, ticks: {{ maxTicksLimit: 5, maxRotation: 0, font: {{ size: 9 }} }} }},
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
        let showParkruns = true;
        
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
            
            // Colour actual dots: parkruns lighter, non-parkrun races darker
            const dotColors = isParkrun.map(p => p === 1 ? 'rgba(252, 165, 165, 0.8)' : 'rgba(239, 68, 68, 0.9)');
            
            // Build data points for time axis
            const predPoints = datesISO.map((dt, i) => predicted[i] !== null ? ({{ x: dt, y: predicted[i] }}) : null).filter(p => p !== null);
            const actualPoints = datesISO.map((dt, i) => ({{ x: dt, y: actual[i] }}));
            
            if (predChart) predChart.destroy();
            
            predChart = new Chart(predCtx.getContext('2d'), {{
                type: 'line',
                data: {{
                    datasets: [{{
                        label: 'Predicted',
                        data: predPoints,
                        borderColor: 'rgba(37, 99, 235, 0.7)',
                        backgroundColor: 'rgba(37, 99, 235, 0.05)',
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
                                    // Gap
                                    if (predicted[i] && actual[i]) {{
                                        const gap = actual[i] - predicted[i];
                                        const sign = gap > 0 ? '+' : '';
                                        lines.push('Gap: ' + sign + formatPredTime(Math.abs(gap)) + (gap > 0 ? ' slower' : ' faster'));
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
                            ticks: {{
                                callback: function(v) {{ return formatPredTime(v); }},
                                font: {{ size: 10 }},
                                maxTicksLimit: 6
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        if (predCtx && predData && predData['5k'] && predData['5k'].dates.length > 0) {{
            renderPredChart('5k', true);
            
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
                tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: #666;">No races found</td></tr>';
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
                return `
                <tr>
                    <td>${{run.date}}</td>
                    <td>${{run.name}} ${{run.race ? '<span class="race-badge">RACE</span>' : ''}}${{deltaHtml}}</td>
                    <td>${{run.dist ? run.dist + ' km' : '-'}}</td>
                    <td>${{run.pace}}</td>
                    <td>${{run.npower || '-'}}</td>
                    <td>${{run.hr || '-'}}</td>
                    <td>${{run.tss || '-'}}</td>
                    <td>${{run.rfl || '-'}}</td>
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
    
    # Generate HTML
    print("Generating HTML...")
    html = generate_html(stats, rf_data, volume_data, ctl_atl_data, ctl_atl_lookup, rfl_trend_dates, rfl_trend_values, rfl_trendline, rfl_projection, rfl_ci_upper, rfl_ci_lower, alltime_rfl_dates, alltime_rfl_values, recent_runs, top_races, alert_data=alert_data, weight_data=weight_data, prediction_data=prediction_data, ag_data=ag_data)
    
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
