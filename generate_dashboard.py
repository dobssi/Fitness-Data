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
def _last_valid(df, col):
    """Return last non-NaN value from column, or None."""
    if col not in df.columns:
        return None
    valid = df[col].dropna()
    return valid.iloc[-1] if len(valid) > 0 else None


def load_and_process_data():
    """Load Master and athlete_data files, prepare data for dashboard."""
    
    # --- Load from Master (v43 columns) ---
    print(f"Loading {MASTER_FILE}...")
    df = pd.read_excel(MASTER_FILE, sheet_name=0)  # Master sheet (may have athlete ID suffix)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # v53: Read singleton values (AG, CP, PEAK_CP) from the unfiltered last row
    # before auto_exclude filter, since StepB writes these to dfm.index[-1]
    # which may be auto-excluded (e.g. short jog at end of day).
    _unfiltered_last = df.iloc[-1] if len(df) > 0 else None
    
    # v52: Filter out auto-excluded activities (junk, duplicates, non-running)
    # These are flagged by StepB's apply_auto_excludes() with auto_exclude=1
    # Races are preserved even if auto-excluded (e.g. no HR) — they still
    # belong in race history, milestones, and age grading.
    if 'auto_exclude' in df.columns:
        _excluded = (df['auto_exclude'] == 1) & (df.get('race_flag', 0) != 1)
        if _excluded.any():
            print(f"  Filtered {_excluded.sum()} auto-excluded non-race activities")
        _excl_races = (df['auto_exclude'] == 1) & (df.get('race_flag', 0) == 1)
        if _excl_races.any():
            print(f"  Kept {_excl_races.sum()} auto-excluded race(s) for race history")
        df = df[~_excluded].reset_index(drop=True)
    else:
        # Fallback for older masters without auto_exclude column
        _pace = pd.to_numeric(df.get('avg_pace_min_per_km'), errors='coerce')
        _dist = pd.to_numeric(df.get('distance_km'), errors='coerce')
        _implausible = (_pace < 3.0) & (_dist > 5.0)
        if _implausible.any():
            print(f"  Filtered {_implausible.sum()} implausible runs (pace < 3:00/km, dist > 5km)")
            df = df[~_implausible].reset_index(drop=True)
    
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
        print(f"  Note: {ATHLETE_DATA_FILE} not found")
    
    # Fallback: if no weight from athlete_data, try Master's weight_kg column
    if weight is None and len(df) > 0 and 'weight_kg' in df.columns:
        wt_vals = df['weight_kg'].dropna()
        if len(wt_vals) > 0:
            weight = round(float(wt_vals.iloc[-1]), 1)
            print(f"  Weight fallback from Master: {weight}kg")
    
    # Final fallback: use config mass (from athlete.yml / --mass-kg)
    if weight is None:
        try:
            from config import ATHLETE_MASS_KG
            if ATHLETE_MASS_KG and ATHLETE_MASS_KG > 0:
                weight = round(float(ATHLETE_MASS_KG), 1)
                print(f"  Weight fallback from config: {weight}kg (static)")
        except (ImportError, Exception):
            pass
    
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
        
        # v53: Read effective PEAK_CP directly from Master (written by bootstrap).
        # This eliminates the shift(1)/current RFL mismatch bug that occurred when
        # reverse-engineering PEAK_CP from CP/RFL_Trend.
        global PEAK_CP_WATTS_DASH, _PEAK_CP_OVERRIDDEN
        _epcp = latest.get('effective_peak_cp')
        if pd.notna(_epcp) and float(_epcp) > 200:
            _new_peak = round(float(_epcp))
            if abs(_new_peak - PEAK_CP_WATTS_DASH) > 5:
                print(f"  PEAK_CP from Master: {_new_peak}W (config: {_cfg_cp}W)")
                PEAK_CP_WATTS_DASH = _new_peak
                _PEAK_CP_OVERRIDDEN = True
        elif pd.notna(cp_val):
            # Fallback for older Master files without effective_peak_cp column
            _rfl_col = 'RFL_Trend'
            _rfl_val = float(latest.get(_rfl_col, 0))
            if _rfl_val > 0.3:
                _derived_peak = round(float(cp_val) / _rfl_val)
                if _derived_peak > 200 and abs(_derived_peak - PEAK_CP_WATTS_DASH) > 5:
                    print(f"  PEAK_CP derived from Master (legacy): {_derived_peak}W (config: {_cfg_cp}W)")
                    PEAK_CP_WATTS_DASH = _derived_peak
                    _PEAK_CP_OVERRIDDEN = True
        
        # Predicted 5K age grade
        # v53: Read from unfiltered last row (StepB writes to absolute last row, may be auto-excluded)
        _ag_source = _unfiltered_last if _unfiltered_last is not None else latest
        ag_val = _ag_source.get('pred_5k_age_grade')
        if not (pd.notna(ag_val) and 30 < float(ag_val) < 120):
            # Stryd AG missing or implausible — try mode-specific
            _ag_mode_col = f'pred_5k_age_grade_{_cfg_power_mode}'
            ag_val = _ag_source.get(_ag_mode_col)
        if not (pd.notna(ag_val) and 30 < float(ag_val) < 120):
            # Also try filtered latest as fallback
            ag_val = latest.get('pred_5k_age_grade')
            if not (pd.notna(ag_val) and 30 < float(ag_val) < 120):
                ag_val = latest.get(f'pred_5k_age_grade_{_cfg_power_mode}')
        if pd.notna(ag_val) and 30 < float(ag_val) < 120:
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
        
        # For GAP/sim athletes, Stryd prediction columns are NaN.
        # Fall back to mode-specific columns so headline stats aren't blank.
        if _cfg_power_mode in ('gap', 'sim'):
            _sfx = f'_{_cfg_power_mode}'
            if pd.isna(pred_5k): pred_5k = latest.get(f'pred_5k_s{_sfx}')
            if pd.isna(pred_10k): pred_10k = latest.get(f'pred_10k_s{_sfx}')
            if pd.isna(pred_hm): pred_hm = latest.get(f'pred_hm_s{_sfx}')
            if pd.isna(pred_mara): pred_mara = latest.get(f'pred_marathon_s{_sfx}')
        
        # If still NaN (latest row has no predictions because previous row had NaN RFL,
        # e.g. a very short jog), fall back to last valid prediction in the Master.
        if pd.isna(pred_5k):
            _col = f'pred_5k_s_{_cfg_power_mode}' if _cfg_power_mode in ('gap', 'sim') else 'pred_5k_s'
            pred_5k = _last_valid(df, _col)
        if pd.isna(pred_10k):
            _col = f'pred_10k_s_{_cfg_power_mode}' if _cfg_power_mode in ('gap', 'sim') else 'pred_10k_s'
            pred_10k = _last_valid(df, _col)
        if pd.isna(pred_hm):
            _col = f'pred_hm_s_{_cfg_power_mode}' if _cfg_power_mode in ('gap', 'sim') else 'pred_hm_s'
            pred_hm = _last_valid(df, _col)
        if pd.isna(pred_mara):
            _col = f'pred_marathon_s_{_cfg_power_mode}' if _cfg_power_mode in ('gap', 'sim') else 'pred_marathon_s'
            pred_mara = _last_valid(df, _col)
        
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
                # Fall back to last valid if latest row is NaN
                if pd.isna(val):
                    val = _last_valid(df, col)
                if val is not None and pd.notna(val):
                    mode_preds[dist] = format_seconds(val)
                    mode_preds[raw_key] = int(val)
            race_predictions[f'_mode_{mode}'] = mode_preds
            
            cp_mode = latest.get(f'CP_{mode}')
            if pd.isna(cp_mode):
                cp_mode = _last_valid(df, f'CP_{mode}')
            race_predictions[f'_cp_{mode}'] = int(round(float(cp_mode))) if cp_mode is not None and pd.notna(cp_mode) else None
            
            peak_cp_mode = latest.get(f'effective_peak_cp_{mode}')
            race_predictions[f'_peak_cp_{mode}'] = int(round(float(peak_cp_mode))) if pd.notna(peak_cp_mode) else None
            
            ag_mode = _ag_source.get(f'pred_5k_age_grade_{mode}') if _unfiltered_last is not None else None
            if not (ag_mode is not None and pd.notna(ag_mode)):
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

    Only includes days up to the last actual training date. For days after that
    (projected planned sessions, zero-TSS tail), replaces with zero-TSS decay
    so the headline stats show real fitness/fatigue, not planned session projections.
    """
    try:
        df = pd.read_excel(master_file, sheet_name=1)  # Daily sheet
        df['Date'] = pd.to_datetime(df['Date'])

        # Find the last day with actual training (non-zero TSS that isn't
        # from a planned session injection). This is the boundary between
        # real data and projection.
        has_planned_col = 'Planned_Source' in df.columns
        if has_planned_col:
            actual_training = df[
                (df['TSS_Running'] > 0) &
                (df['Planned_Source'].isna() | (df['Planned_Source'] == ''))
            ]
        else:
            actual_training = df[df['TSS_Running'] > 0]

        if len(actual_training) == 0:
            print("  Warning: no actual training days found in Daily sheet")
            return {}

        last_real_date = actual_training['Date'].max()

        # Include all days up to and including the last actual training date
        actual_days = df[df['Date'] <= last_real_date]

        # Build lookup from actual days
        lookup = {}
        last_ctl = None
        last_atl = None
        for _, row in actual_days.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d')
            ctl = round(row['CTL'], 1) if pd.notna(row.get('CTL')) else None
            atl = round(row['ATL'], 1) if pd.notna(row.get('ATL')) else None
            tsb_val = round(row['TSB'], 1) if pd.notna(row.get('TSB')) else None
            lookup[date_str] = {'ctl': ctl, 'atl': atl, 'tsb': tsb_val}
            if ctl is not None:
                last_ctl = ctl
            if atl is not None:
                last_atl = atl

        # Project zero-TSS decay for up to 30 days beyond the last actual day
        # so the headline always shows current fitness when viewed days later
        if last_ctl is not None and last_atl is not None:
            decay_c = 1 - 1/42  # CTL decay constant
            decay_a = 1 - 1/7   # ATL decay constant
            ctl_proj = last_ctl
            atl_proj = last_atl
            for d in range(1, 31):
                proj_date = last_real_date + pd.Timedelta(days=d)
                proj_str = proj_date.strftime('%Y-%m-%d')
                ctl_proj *= decay_c
                atl_proj *= decay_a
                lookup[proj_str] = {
                    'ctl': round(ctl_proj, 1),
                    'atl': round(atl_proj, 1),
                    'tsb': round(ctl_proj - atl_proj, 1),
                }

        actual_count = len(actual_days)
        projected_count = len(lookup) - actual_count
        print(f"  CTL/ATL lookup: {actual_count} actual days (to {last_real_date.strftime('%Y-%m-%d')}) + {projected_count} decay days")
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
    
    latest_valid = rfl.dropna()
    if latest_valid.empty:
        return None
    latest_rfl = latest_valid.iloc[-1]
    
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


def get_upcoming_sessions(master_file):
    """Extract upcoming planned sessions from the Daily sheet.

    Returns list of dicts: [{date_str, day_name, description, tss, source, is_race}, ...]
    Only includes future dates with planned sessions. If today already has an
    actual activity, starts from tomorrow.
    """
    try:
        df = pd.read_excel(master_file, sheet_name=1)  # Daily sheet
        df['Date'] = pd.to_datetime(df['Date'])
        if 'Planned_Description' not in df.columns or 'Planned_Source' not in df.columns:
            return []

        today = pd.Timestamp(datetime.now().date())

        # Check if today already has an actual run (non-planned TSS)
        # But still show today if it's a planned race that hasn't been run yet
        today_row = df[df['Date'] == today]
        has_actual_today = False
        today_is_planned_race = False
        if len(today_row) > 0:
            row = today_row.iloc[0]
            src = str(row.get('Planned_Source', ''))
            has_actual_today = row.get('TSS_Running', 0) > 0 and src in ('', 'nan')
            today_is_planned_race = src == 'race'

        # Skip today only if we have actual data AND it's not a planned race day
        start_date = today + timedelta(days=1) if (has_actual_today and not today_is_planned_race) else today

        # Filter to future planned sessions
        future = df[
            (df['Date'] >= start_date) &
            (df['Planned_Description'].notna()) &
            (df['Planned_Description'] != '') &
            (df['Planned_Description'] != 'nan')
        ].sort_values('Date')

        if len(future) == 0:
            return []

        # Calculate actual TSS already completed earlier this week
        # (so the first week total reflects the full week, not just remaining days)
        first_future_date = future.iloc[0]['Date']
        iso_cal = first_future_date.isocalendar()
        week_start = first_future_date - timedelta(days=first_future_date.weekday())  # Monday
        earlier_this_week = df[
            (df['Date'] >= week_start) &
            (df['Date'] < start_date)
        ]
        completed_tss_this_week = round(earlier_this_week['TSS_Running'].fillna(0).sum())

        sessions = []
        for _, row in future.iterrows():
            dt = row['Date']
            sessions.append({
                'date_str': dt.strftime('%a %d %b'),
                'date_iso': dt.strftime('%Y-%m-%d'),
                'description': str(row['Planned_Description']),
                'tss': round(row.get('TSS_Running', 0)),
                'source': str(row.get('Planned_Source', '')),
                'is_race': str(row.get('Planned_Source', '')) == 'race',
                'ctl': round(row.get('CTL', 0), 1) if pd.notna(row.get('CTL', None)) else None,
                'atl': round(row.get('ATL', 0), 1) if pd.notna(row.get('ATL', None)) else None,
                'tsb': round(row.get('TSB', 0), 1) if pd.notna(row.get('TSB', None)) else None,
            })

        return sessions, completed_tss_this_week
    except Exception as e:
        print(f"  Warning: could not load upcoming sessions: {e}")
        return []


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
        # Phase 2: GAP and Sim RFL for mode toggle — use last valid row (short runs have NaN RF)
        'latest_rfl_gap': round(_last_valid(df, 'RFL_gap_Trend') * 100, 1) if _last_valid(df, 'RFL_gap_Trend') is not None else None,
        'latest_rfl_sim': round(_last_valid(df, 'RFL_sim_Trend') * 100, 1) if _last_valid(df, 'RFL_sim_Trend') is not None else None,
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
        daily_df = pd.read_excel(MASTER_FILE, sheet_name=1)  # Daily sheet
        daily_df['Date'] = pd.to_datetime(daily_df['Date'])
        
        # Today boundary
        today = pd.Timestamp.now().normalize()
        cutoff = today - timedelta(days=days)
        
        # Historical data (up to and including today)
        df_hist = daily_df[(daily_df['Date'] > cutoff) & (daily_df['Date'] <= today)].copy()
        df_hist = df_hist[df_hist['CTL'].notna()]
        df_hist = df_hist.sort_values('Date')
        
        # Future projection data — extend to last planned session day + 1
        # Read Planned_Source column to find extent of planned sessions
        has_planned = 'Planned_Source' in daily_df.columns
        if has_planned:
            planned_future = daily_df[(daily_df['Date'] > today) & (daily_df['Planned_Source'] != '')]
            if len(planned_future) > 0:
                last_planned = planned_future['Date'].max()
                future_end = last_planned + timedelta(days=1)
            else:
                future_end = today + timedelta(days=14)
        else:
            future_end = today + timedelta(days=14)
        
        df_future = daily_df[(daily_df['Date'] > today) & (daily_df['Date'] <= future_end)].copy()
        df_future = df_future[df_future['CTL'].notna()]
        df_future = df_future.sort_values('Date')
        
        if len(df_hist) == 0:
            print("  Warning: No CTL/ATL data found in date range")
            return [], [], [], [], [], [], []
        
        # Historical series
        dates = df_hist['Date'].dt.strftime('%d %b %y').tolist()
        ctl_values = [round(v, 1) if pd.notna(v) else None for v in df_hist['CTL'].tolist()]
        atl_values = [round(v, 1) if pd.notna(v) else None for v in df_hist['ATL'].tolist()]
        
        # Projection series (starts from last historical point for continuity)
        if len(df_future) > 0:
            proj_dates = df_future['Date'].dt.strftime('%d %b %y').tolist()
            proj_ctl = [round(v, 1) if pd.notna(v) else None for v in df_future['CTL'].tolist()]
            proj_atl = [round(v, 1) if pd.notna(v) else None for v in df_future['ATL'].tolist()]
            proj_tsb = [round(v, 1) if pd.notna(v) else None for v in df_future['TSB'].tolist()]
            
            # Planned session markers (description + source for tooltips)
            proj_planned = []
            if has_planned:
                for _, row in df_future.iterrows():
                    desc = row.get('Planned_Description', '')
                    source = row.get('Planned_Source', '')
                    if desc:
                        proj_planned.append({'desc': str(desc), 'source': str(source)})
                    else:
                        proj_planned.append(None)
            else:
                proj_planned = [None] * len(proj_dates)
            
            # Combine dates for x-axis
            all_dates = dates + proj_dates
            
            # Historical values with nulls for projection period
            ctl_with_nulls = ctl_values + [None] * len(proj_dates)
            atl_with_nulls = atl_values + [None] * len(proj_dates)
            
            # Projection values: null for historical except last point, then projection
            ctl_proj_line = [None] * (len(dates) - 1) + [ctl_values[-1]] + proj_ctl
            atl_proj_line = [None] * (len(dates) - 1) + [atl_values[-1]] + proj_atl
            tsb_proj_line = [None] * (len(dates) - 1) + [round(ctl_values[-1] - atl_values[-1], 1) if ctl_values[-1] and atl_values[-1] else None] + proj_tsb
            planned_markers = [None] * len(dates) + proj_planned
            
            print(f"  Loaded {len(dates)} days of CTL/ATL data + {len(proj_dates)} days projection")
            n_planned = sum(1 for p in proj_planned if p is not None)
            if n_planned > 0:
                print(f"  Projection includes {n_planned} planned sessions")
            return all_dates, ctl_with_nulls, atl_with_nulls, ctl_proj_line, atl_proj_line, tsb_proj_line, planned_markers
        else:
            print(f"  Loaded {len(dates)} days of CTL/ATL data (no projection available)")
            return dates, ctl_values, atl_values, [], [], [], []
        
    except Exception as e:
        print(f"  Warning: Could not load CTL/ATL trend: {e}")
        return [], [], [], [], [], [], []


def get_daily_rfl_trend(master_file, days=14, rfl_col='RFL_Trend'):
    """Get daily RFL trend from Master's Daily sheet with trendline, projection and 95% CI."""
    try:
        df = pd.read_excel(master_file, sheet_name=1)  # Daily sheet
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
        df = pd.read_excel(master_file, sheet_name=2)  # Weekly sheet
        
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
                'date_iso': row['date'].strftime('%Y-%m-%d'),
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
        df = pd.read_excel(master_file, sheet_name=0,  # Master sheet
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
    # v52: 2.0 × ln(dist/5.0) — penalises sub-5K (WMA tables inflate short distances)
    # and boosts longer distances. Anchored at 5K=0%.
    # Mile=-2.3%, 3K=-1.0%, 5K=0%, 10K=+1.4%, HM=+2.9%, Marathon=+4.3%
    _dist_slope = 0.02        # 2.0% per ln(km) unit
    _dist_ref = 5.0           # Reference distance (no correction)
    
    dist_adj = 1.0
    if dist_km is not None and pd.notna(dist_km) and dist_km > 0:
        dist_adj = 1.0 + _dist_slope * math.log(dist_km / _dist_ref)
    
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
        
        # Use elapsed time for races (actual finish time, not auto-pause moving time)
        elapsed_s = row.get(elapsed_col)
        moving_s = elapsed_s if pd.notna(elapsed_s) and elapsed_s > 0 else row.get(moving_col, None)
        
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
            'date_iso': row['date'].strftime('%Y-%m-%d'),
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
# RACE HISTORY DATA — for side-by-side comparison cards
# ============================================================================
def get_race_history_data(df, ctl_atl_lookup, zone_data=None):
    """Extract all past races with training context for comparison cards.
    
    Uses NPZ per-second data for accurate time-in-zone and long run tail
    metrics in the 14d/42d windows before each race.
    
    Returns list of race dicts sorted by date descending.
    """
    import glob, os, re as _re_mod
    
    race_col = 'race_flag' if 'race_flag' in df.columns else 'Race'
    races = df[df[race_col] == 1].copy()
    
    if len(races) == 0:
        return []
    
    # Column name resolution
    dist_col = 'distance_km' if 'distance_km' in df.columns else 'Distance_m'
    official_dist_col = 'official_distance_km' if 'official_distance_km' in races.columns else 'Official_Dist_km'
    elapsed_col = 'elapsed_time_s' if 'elapsed_time_s' in races.columns else 'Elapsed_s'
    moving_col = 'moving_time_s' if 'moving_time_s' in races.columns else 'Moving_s'
    
    # ── Zone boundaries ──
    _z3_hr = _z23_hr  # Z3 floor = module-level value (LTHR × 0.90)
    
    # Determine effort mode: power (Stryd) or GAP pace or HR fallback
    _effort_mode = _cfg_power_mode  # 'stryd', 'gap', 'sim'
    
    # Race HR zone bounds (module-level, constant across races)
    _race_hr_bounds = [0, _rhr_mara, _rhr_hm, _rhr_10k, _rhr_5k, _rhr_sub5k, 9999]
    _race_hr_names  = ['Other', 'Mara', 'HM', '10K', '5K', 'Sub-5K']
    
    # Power/pace zone bounds are computed PER RACE from race-day fitness
    # (see _build_race_pw_bounds and _build_race_pace_bounds below)
    
    # Effort zones: match SPEC_ZONES from Race Readiness (specific, not cumulative)
    _effort_zones = {
        '3K':       ['Sub-5K'],
        '5K':       ['Sub-5K', '5K'],
        '10K':      ['10K'],
        '10M':      ['HM'],
        'HM':       ['HM'],
        '30K':      ['Mara'],
        'Marathon':  ['Mara'],
    }
    
    def _build_race_pw_bounds(race_cp):
        """Build race power zone bounds from a specific CP value."""
        m = race_cp * 0.90
        h = race_cp * 0.95
        t = race_cp * 1.00
        f = race_cp * 1.05
        return [0, round(m*0.93), round((m+h)/2), round((h+t)/2), round((t+f)/2), round(f*1.05), 9999]
    
    _race_pw_names = ['Other', 'Mara', 'HM', '10K', '5K', 'Sub-5K']
    
    def _build_race_pace_bounds(row):
        """Build race GAP pace zone bounds from predictions on a specific row."""
        _suffix = '_gap' if _effort_mode == 'gap' else ('_sim' if _effort_mode == 'sim' else '')
        p5  = round(row.get(f'pred_5k_s{_suffix}', 0) / 5.0) if pd.notna(row.get(f'pred_5k_s{_suffix}')) and row.get(f'pred_5k_s{_suffix}', 0) > 0 else 240
        p10 = round(row.get(f'pred_10k_s{_suffix}', 0) / 10.0) if pd.notna(row.get(f'pred_10k_s{_suffix}')) and row.get(f'pred_10k_s{_suffix}', 0) > 0 else 252
        ph  = round(row.get(f'pred_hm_s{_suffix}', 0) / 21.097) if pd.notna(row.get(f'pred_hm_s{_suffix}')) and row.get(f'pred_hm_s{_suffix}', 0) > 0 else 265
        pm  = round(row.get(f'pred_marathon_s{_suffix}', 0) / 42.195) if pd.notna(row.get(f'pred_marathon_s{_suffix}')) and row.get(f'pred_marathon_s{_suffix}', 0) > 0 else 282
        return [0, round(p5*0.97), round((p5+p10)/2), round((p10+ph)/2), round((ph+pm)/2), round(pm*1.07), 9999]
    
    _race_pace_names = ['Sub-5K', '5K', '10K', 'HM', 'Mara', 'Other']
    
    print(f"  Race history effort mode: {_effort_mode}")
    
    # Long run threshold: fixed 60 minutes
    LR_THRESHOLD_S = 3600  # 60 min in seconds
    LR_THRESHOLD_MIN = 60
    
    # Overnight decay for race-morning CTL/ATL (Banister exponential, zero TSS)
    import math
    _morning_decay_ctl = math.exp(-1/42)
    _morning_decay_atl = math.exp(-1/7)
    
    # ── NPZ cache index ──
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
    
    npz_index = {}
    npz_by_date = {}
    if npz_dir:
        for fp in glob.glob(os.path.join(npz_dir, '*.npz')):
            base = os.path.basename(fp).replace('.npz', '')
            npz_index[base] = fp
            date_prefix = base[:10]
            if date_prefix not in npz_by_date:
                npz_by_date[date_prefix] = []
            npz_by_date[date_prefix].append(fp)
        print(f"  Race history NPZ cache: {npz_dir} ({len(npz_index)} files)")
    
    # ── Build run-to-NPZ lookup for all runs ──
    # Pre-index: for each run in df, resolve its NPZ path once
    _run_npz = {}  # index in df -> npz_path
    for idx, row in df.iterrows():
        run_date = row['date']
        run_id = run_date.strftime('%Y-%m-%d_%H-%M-%S')
        
        master_run_id = str(row.get('run_id', '')).strip() if pd.notna(row.get('run_id')) else ''
        npz_key = master_run_id.replace('&', '_').replace('?', '_').replace('=', '_') if master_run_id else ''
        
        file_id = ''
        if not npz_key and pd.notna(row.get('file')):
            file_id = str(row['file']).replace('.fit', '').replace('.FIT', '').strip()
        
        npz_path = npz_index.get(npz_key) if npz_key else None
        if not npz_path and file_id:
            npz_path = npz_index.get(file_id)
        if not npz_path:
            npz_path = npz_index.get(run_id)
        if not npz_path:
            date_prefix = run_date.strftime('%Y-%m-%d')
            candidates = npz_by_date.get(date_prefix, [])
            if len(candidates) == 1:
                npz_path = candidates[0]
        if npz_path:
            _run_npz[idx] = npz_path
    
    print(f"  Race history: {len(_run_npz)}/{len(df)} runs have NPZ files")
    
    # ── NPZ helpers with caching ──
    _tail_cache = {}   # npz_path -> (total_tail_min, z3_tail_min)
    _effort_cache = {} # (npz_path, tuple(zones)) -> effort_min
    
    def _npz_tail(npz_path):
        """Load NPZ, return (hr_arr_or_None, total_tail_mins, z3_tail_mins)."""
        if npz_path in _tail_cache:
            return _tail_cache[npz_path]
        try:
            data = np.load(npz_path, allow_pickle=True)
            hr_arr = data['hr_bpm']
            n = len(hr_arr)
            if n < 60:
                _tail_cache[npz_path] = (0, 0)
                return (0, 0)
            
            # Long run tail: seconds beyond 60 min mark
            total_tail_s = 0
            z3_tail_s = 0
            if n > LR_THRESHOLD_S:
                hr_s = pd.Series(hr_arr).rolling(30, min_periods=1).mean().values
                tail_hr = hr_s[LR_THRESHOLD_S:]
                valid_tail = tail_hr[~np.isnan(tail_hr) & (tail_hr > 0)]
                total_tail_s = len(valid_tail)
                z3_tail_s = int(np.sum(valid_tail >= _z3_hr))
            
            result = (total_tail_s / 60.0, z3_tail_s / 60.0)
            _tail_cache[npz_path] = result
            return result
        except Exception:
            _tail_cache[npz_path] = (0, 0)
            return (0, 0)
    
    def _npz_effort(npz_path, effort_zones_tuple, pw_bounds=None, pace_bounds=None, era_adj=1.0):
        """Count minutes at race-relevant zones from NPZ.
        Stryd mode: race power zones (pw_bounds), with era_adj applied to raw power.
        GAP mode: race GAP pace zones (pace_bounds). Fallback: HR zones.
        Zone bounds are per-race (derived from race-day CP / predictions).
        """
        # Cache key includes bounds and era_adj since they vary per race/run
        cache_key = (npz_path, effort_zones_tuple, 
                     tuple(pw_bounds) if pw_bounds else None,
                     tuple(pace_bounds) if pace_bounds else None,
                     round(era_adj, 4))
        if cache_key in _effort_cache:
            return _effort_cache[cache_key]
        try:
            data = np.load(npz_path, allow_pickle=True)
            n = len(data['hr_bpm'])
            if n < 60:
                _effort_cache[cache_key] = 0
                return 0
            
            effort_set = set(effort_zones_tuple)
            effort_s = 0
            
            if _effort_mode not in ('gap', 'sim') and pw_bounds and 'power_w' in data:
                # Stryd mode: use race power zones
                # Apply era_adj to normalise raw pod power to S4 baseline
                pw_arr = data['power_w'].copy()
                pw_arr = np.where(np.isnan(pw_arr), 0.0, pw_arr) * era_adj
                pw_s = pd.Series(pw_arr).rolling(30, min_periods=1).mean().values
                for v in pw_s:
                    if np.isnan(v) or v <= 0:
                        continue
                    for i in range(len(pw_bounds) - 2, -1, -1):
                        if v >= pw_bounds[i]:
                            if _race_pw_names[i] in effort_set:
                                effort_s += 1
                            break
            elif _effort_mode == 'gap' and pace_bounds and 'speed_mps' in data and 'grade' in data:
                # GAP mode: use race GAP pace zones
                try:
                    from gap_power import compute_gap_for_run
                    spd_arr = data['speed_mps']
                    grd_arr = data['grade']
                    grd_safe = np.where(np.isnan(grd_arr), 0, grd_arr)
                    gap_speed = compute_gap_for_run(spd_arr, grd_safe)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        gap_pace = np.where(gap_speed > 0.5, 1000.0 / gap_speed, np.nan)
                    gap_s = pd.Series(gap_pace).rolling(30, min_periods=1).mean().values
                    for v in gap_s:
                        if np.isnan(v) or v <= 0:
                            continue
                        assigned = False
                        for i in range(len(pace_bounds) - 1):
                            if pace_bounds[i] <= v < pace_bounds[i + 1]:
                                if _race_pace_names[i] in effort_set:
                                    effort_s += 1
                                assigned = True
                                break
                        if not assigned and _race_pace_names[-1] in effort_set:
                            effort_s += 1
                except Exception:
                    pass  # Fall through to HR fallback
            
            if effort_s == 0:
                # HR fallback (sim mode, or if power/GAP failed)
                hr_arr = data['hr_bpm']
                hr_s = pd.Series(hr_arr).rolling(30, min_periods=1).mean().values
                for v in hr_s:
                    if np.isnan(v) or v <= 0:
                        continue
                    for i in range(len(_race_hr_bounds) - 2, -1, -1):
                        if v >= _race_hr_bounds[i]:
                            if _race_hr_names[i] in effort_set:
                                effort_s += 1
                            break
            
            result = effort_s / 60.0
            _effort_cache[cache_key] = result
            return result
        except Exception:
            _effort_cache[cache_key] = 0
            return 0
    
    # ── Process each race ──
    result = []
    
    for _, row in races.iterrows():
        # Distance
        official_dist = row.get(official_dist_col)
        if pd.notna(official_dist) and official_dist > 0:
            dist_km = float(official_dist)
        elif dist_col == 'distance_km':
            dist_km = row[dist_col] if pd.notna(row.get(dist_col)) else 0
        else:
            dist_km = row[dist_col] / 1000 if pd.notna(row.get(dist_col)) else 0
        
        if dist_km <= 0:
            continue
        
        # Distance category — standard distances get their label,
        # bespoke distances (not close to any standard) get 'Other'
        # Same tolerance formula as classify_races:
        #   Base: max(2%, 300m), Parkrun: max(3%, 400m), Bad GPS: max(4%, 500m)
        _std_dists = [(1.5, '1500m'), (1.609, 'Mile'),
                      (3.0, '3K'), (5.0, '5K'), (10.0, '10K'),
                      (16.0934, '10M'), (21.097, 'HM'),
                      (30.0, '30K'), (42.195, 'Marathon')]
        _is_parkrun = bool(_re_mod.search(r'parkrun|park run', str(row.get('activity_name', '')), _re_mod.I))
        _max_seg = row.get('gps_max_seg_m', 0) if 'gps_max_seg_m' in row.index else 0
        _outlier = row.get('gps_outlier_frac', 0) if 'gps_outlier_frac' in row.index else 0
        _max_seg = _max_seg if (pd.notna(_max_seg)) else 0
        _outlier = _outlier if (pd.notna(_outlier)) else 0
        _bad_gps = (_max_seg > 200) or (_outlier > 0.005)
        _matched_cat = None
        _best_gap = float('inf')
        for _sd, _lbl in _std_dists:
            if _bad_gps:
                _tol = max(_sd * 0.04, 0.5)
            elif _is_parkrun and _lbl == '5K':
                _tol = max(_sd * 0.03, 0.4)
            else:
                _tol = max(_sd * 0.02, 0.3)
            _gap = abs(dist_km - _sd)
            if _gap <= _tol and _gap < _best_gap:
                _matched_cat = _lbl
                _best_gap = _gap
        if _matched_cat:
            dist_cat = _matched_cat
        else:
            dist_cat = 'Other'
        
        # Time — prefer elapsed for races
        elapsed_s = row.get(elapsed_col)
        time_s = elapsed_s if pd.notna(elapsed_s) and elapsed_s > 0 else row.get(moving_col, None)
        
        if pd.notna(time_s) and time_s > 0:
            hours = int(time_s // 3600)
            mins = int((time_s % 3600) // 60)
            secs = int(time_s % 60)
            time_str = f"{hours}:{mins:02d}:{secs:02d}" if hours > 0 else f"{mins}:{secs:02d}"
            pace_skm = time_s / dist_km if dist_km > 0 else 0
            pace_min = int(pace_skm // 60)
            pace_sec = int(pace_skm % 60)
            pace_str = f"{pace_min}:{pace_sec:02d}"
        else:
            time_str = "-"
            pace_str = "-"
            time_s = None
        
        # Name
        name_val = row.get('activity_name', row.get('Activity_Name', ''))
        name = str(name_val) if pd.notna(name_val) and name_val else ''
        
        # HR
        hr = int(round(row['avg_hr'])) if pd.notna(row.get('avg_hr')) else None
        
        # AG (normalised)
        raw_ag = round(row['age_grade_pct'], 1) if pd.notna(row.get('age_grade_pct')) else None
        _off_dist_col_r = 'official_distance_km' if 'official_distance_km' in races.columns else 'Official_Dist_km'
        nag_raw = _calc_normalised_ag(
            row.get('age_grade_pct'),
            row.get('moving_time_s', row.get('elapsed_time_s')),
            row.get('Temp_Adj'),
            row.get('Terrain_Adj'),
            row.get('surface_adj'),
            row.get('Elevation_Adj'),
            row.get('avg_temp_c'),
            row.get(_off_dist_col_r, row.get('distance_km')),
        ) if pd.notna(row.get('age_grade_pct')) else None
        nag = round(nag_raw, 1) if nag_raw is not None and pd.notna(nag_raw) else raw_ag
        
        # TSS
        tss = int(round(row['TSS'])) if pd.notna(row.get('TSS')) else None
        
        # Training state MORNING of race: previous day's end-of-day CTL/ATL
        # with one day of zero-TSS decay applied (overnight recovery)
        date_str = row['date'].strftime('%Y-%m-%d')
        prev_date = (row['date'] - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        day_data = ctl_atl_lookup.get(prev_date, ctl_atl_lookup.get(date_str, {}))
        _prev_ctl = day_data.get('ctl')
        _prev_atl = day_data.get('atl')
        if _prev_ctl is not None and _prev_atl is not None:
            # Apply one day of exponential decay (Banister model, zero TSS)
            ctl = round(_prev_ctl * _morning_decay_ctl, 1)
            atl = round(_prev_atl * _morning_decay_atl, 1)
            tsb = round(ctl - atl, 1)
        else:
            ctl = _prev_ctl
            atl = _prev_atl
            tsb = day_data.get('tsb')
        
        # RFL on race day
        rfl = round(row['RFL_Trend'] * 100, 1) if pd.notna(row.get('RFL_Trend')) else None
        rfl_gap = round(row['RFL_gap_Trend'] * 100, 1) if pd.notna(row.get('RFL_gap_Trend')) else None
        
        # Prediction at the time (mode-aware) — only for distances with dedicated prediction columns
        _pred_dist_map = {'5K': '5k', '10K': '10k', 'HM': 'hm', 'Marathon': 'marathon'}
        _pred_dist_key = _pred_dist_map.get(dist_cat)
        pred_time_s = None
        if _pred_dist_key:
            _pred_base_col = f'pred_{_pred_dist_key}_s'
            _pred_mode_suffix = {'gap': '_gap', 'sim': '_sim'}.get(_cfg_power_mode, '')
            _pred_col = _pred_base_col + _pred_mode_suffix
            _pred_s = row.get(_pred_col)
            # Fallback to base column if mode-specific is missing
            if not (pd.notna(_pred_s) and float(_pred_s) > 0):
                _pred_s = row.get(_pred_base_col)
            pred_time_s = round(float(_pred_s)) if pd.notna(_pred_s) and float(_pred_s) > 0 else None
        
        # Conditions
        temp = round(row['avg_temp_c'], 1) if pd.notna(row.get('avg_temp_c')) else None
        terrain = round(row['Terrain_Adj'], 3) if pd.notna(row.get('Terrain_Adj')) else None
        solar_wm2 = round(row['avg_solar_rad_wm2']) if pd.notna(row.get('avg_solar_rad_wm2')) else None
        elev_gain = round(row['elev_gain_m']) if pd.notna(row.get('elev_gain_m')) else None
        undulation = round(row['rf_window_undulation_score'], 1) if pd.notna(row.get('rf_window_undulation_score')) else None
        surface = row.get('surface', 'road')
        surface = str(surface) if pd.notna(surface) else 'road'
        
        # ── NPZ-based effort + long run tail for 14d/42d windows ──
        race_date = row['date']
        e14, e42 = 0.0, 0.0
        lr_total_14, lr_total_42 = 0.0, 0.0
        lr_z3_14, lr_z3_42 = 0.0, 0.0
        
        effort_zone_list = _effort_zones.get(dist_cat, ['HM', '10K', '5K', 'Sub-5K'])
        
        # Build race-day-specific zone bounds
        # Power: CP on race day = PEAK_CP × RFL_Trend (era-adjusted, already in master)
        _rfl_col = {'gap': 'RFL_gap_Trend', 'sim': 'RFL_sim_Trend'}.get(_effort_mode, 'RFL_Trend')
        _race_rfl = row.get(_rfl_col)
        if not pd.notna(_race_rfl):
            _race_rfl = row.get('RFL_Trend', 0.90)
        if not pd.notna(_race_rfl):
            _race_rfl = 0.90
        _race_cp = round(PEAK_CP_WATTS_DASH * float(_race_rfl))
        _pw_bounds_for_race = _build_race_pw_bounds(_race_cp) if _effort_mode not in ('gap', 'sim') else None
        _pace_bounds_for_race = _build_race_pace_bounds(row) if _effort_mode == 'gap' else None
        
        for train_idx, train_row in df.iterrows():
            days_before = (race_date - train_row['date']).days
            if days_before <= 0 or days_before > 42:
                continue
            
            npz_path = _run_npz.get(train_idx)
            if not npz_path:
                continue
            
            effort_zones_tuple = tuple(effort_zone_list)
            _train_era_adj = float(train_row.get('Era_Adj', 1.0)) if pd.notna(train_row.get('Era_Adj')) else 1.0
            eff = _npz_effort(npz_path, effort_zones_tuple, _pw_bounds_for_race, _pace_bounds_for_race, _train_era_adj)
            tail, z3_tail = _npz_tail(npz_path)
            
            if days_before <= 14:
                e14 += eff
                lr_total_14 += tail
                lr_z3_14 += z3_tail
            e42 += eff
            lr_total_42 += tail
            lr_z3_42 += z3_tail
        
        result.append({
            'date': date_str,
            'date_display': row['date'].strftime('%d %b %y'),
            'name': name,
            'dist_km': round(dist_km, 2),
            'dist_cat': dist_cat,
            'time': time_str,
            'time_s': round(time_s) if time_s else None,
            'pace': pace_str,
            'hr': hr,
            'nag': nag,
            'raw_ag': raw_ag,
            'tss': tss,
            'ctl': ctl,
            'atl': atl,
            'tsb': tsb,
            'rfl': rfl,
            'rfl_gap': rfl_gap,
            'pred_s': pred_time_s,
            'e14': round(e14),
            'e42': round(e42),
            'lr_total_14': round(lr_total_14),
            'lr_total_42': round(lr_total_42),
            'lr_z3_14': round(lr_z3_14),
            'lr_z3_42': round(lr_z3_42),
            'lr_threshold': LR_THRESHOLD_MIN,
            'temp': temp,
            'solar': solar_wm2,
            'elev_gain': elev_gain,
            'undulation': undulation,
            'terrain': terrain,
            'surface': surface,
        })
    
    # Sort by date descending (most recent first)
    result.sort(key=lambda x: x['date'], reverse=True)
    print(f"  Race history: {len(result)} races extracted for comparison")
    return result


# ============================================================================
# v51: RACE PREDICTION TREND DATA
# ============================================================================
def get_prediction_trend_data(df):
    """Get predicted vs actual race times — prediction shown only at race dates.
    Returns dict keyed by distance."""
    result = {}
    
    # Mode-aware prediction columns: GAP athletes have pred_5k_s_gap, not pred_5k_s
    def _pred_col(base):
        """Return best available prediction column for current power mode."""
        mode_col = f'{base}_{_cfg_power_mode}' if _cfg_power_mode else base
        if mode_col in df.columns and df[mode_col].notna().any():
            return mode_col
        if base in df.columns and df[base].notna().any():
            return base
        for m in ('gap', 'sim'):
            c = f'{base}_{m}'
            if c in df.columns and df[c].notna().any():
                return c
        return base

    distances = {
        '5k': {'pred_col': _pred_col('pred_5k_s'), 'base_col': 'pred_5k_s', 'official_km': 5.0},
        '10k': {'pred_col': _pred_col('pred_10k_s'), 'base_col': 'pred_10k_s', 'official_km': 10.0},
        'hm': {'pred_col': _pred_col('pred_hm_s'), 'base_col': 'pred_hm_s', 'official_km': 21.097},
        'marathon': {'pred_col': _pred_col('pred_marathon_s'), 'base_col': 'pred_marathon_s', 'official_km': 42.195},
    }
    
    for dist_key, info in distances.items():
        pred_col = info['pred_col']
        dist_km = info['official_km']
        # v53: Tight tolerance — only include races at genuinely standard distances.
        # Bespoke races (Kvartsmarathon 10.356km, Assembly 4.78km) should NOT appear
        # on the 10K or 5K prediction tabs. They show in Race History "Other" instead.
        tolerance = max(dist_km * 0.02, 0.3)  # max(2%, 300m) — matches classify_races
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
        race_re_adjs = []
        race_solar = []
        race_elev = []
        race_undulation = []
        race_dists = []
        for _, race_row in dist_match.iterrows():
            ta = race_row.get('Temp_Adj')
            race_temp_adjs.append(round(float(ta), 4) if pd.notna(ta) else 1.0)
            sa = race_row.get('surface_adj')
            race_surface_adjs.append(round(float(sa), 4) if pd.notna(sa) else 1.0)
            # v53: RE condition adjustment (Stryd mode only)
            ra = race_row.get('RE_Adj')
            race_re_adjs.append(round(float(ra), 4) if pd.notna(ra) and 0.5 < float(ra) < 2.0 else 1.0)
            sol = race_row.get('avg_solar_rad_wm2')
            race_solar.append(round(float(sol)) if pd.notna(sol) else None)
            eg = race_row.get('elev_gain_m')
            race_elev.append(round(float(eg)) if pd.notna(eg) else None)
            und = race_row.get('rf_window_undulation_score')
            race_undulation.append(round(float(und), 1) if pd.notna(und) else None)
            dk = race_row.get('distance_km')
            race_dists.append(round(float(dk), 2) if pd.notna(dk) else None)
        
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
            're_adjs': race_re_adjs,
            'solar': race_solar,
            'elev_gain': race_elev,
            'undulation': race_undulation,
            'dists': race_dists,
        }
        
        # Phase 2: GAP and Sim predicted times at each race date
        base_col = info['base_col']
        for mode in ('gap', 'sim'):
            mode_col = f'{base_col}_{mode}'
            mode_preds = []
            for _, race_row in dist_match.iterrows():
                pv = race_row.get(mode_col)
                if pd.notna(pv) and float(pv) > 0:
                    mode_preds.append(round(float(pv), 0))
                else:
                    # Nearby fallback: find closest prediction within 7 days
                    race_dt = race_row['date']
                    if mode_col in df.columns:
                        nearby = df[(df['date'] - race_dt).abs() <= pd.Timedelta(days=7)]
                        nearby_pred = nearby[mode_col].dropna()
                        nearby_pred = nearby_pred[nearby_pred > 0]
                        if len(nearby_pred) > 0:
                            mode_preds.append(round(float(nearby_pred.iloc[-1]), 0))
                        else:
                            mode_preds.append(None)
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
                mode_col = f'{base_col}_{mode}'
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
    
    # Conditions arrays for chart tooltips
    ag_temps = [round(float(v), 1) if pd.notna(v) else None for v in races['avg_temp_c'].tolist()] if 'avg_temp_c' in races.columns else [None]*len(races)
    ag_solar = [round(float(v)) if pd.notna(v) else None for v in races['avg_solar_rad_wm2'].tolist()] if 'avg_solar_rad_wm2' in races.columns else [None]*len(races)
    ag_elev = [round(float(v)) if pd.notna(v) else None for v in races['elev_gain_m'].tolist()] if 'elev_gain_m' in races.columns else [None]*len(races)
    ag_undulation = [round(float(v), 1) if pd.notna(v) else None for v in races['rf_window_undulation_score'].tolist()] if 'rf_window_undulation_score' in races.columns else [None]*len(races)
    ag_surfaces = [str(v) if pd.notna(v) and str(v) != 'nan' else None for v in races['surface'].tolist()] if 'surface' in races.columns else [None]*len(races)
    ag_dists = [round(float(v), 2) if pd.notna(v) else None for v in races['distance_km'].tolist()] if 'distance_km' in races.columns else [None]*len(races)
    ag_names = [str(v)[:60] if pd.notna(v) else '' for v in races['activity_name'].tolist()] if 'activity_name' in races.columns else ['']*len(races)
    
    return {'dates': dates, 'dates_iso': dates_iso, 'values': values, 'dist_labels': dist_labels, 
            'dist_codes': dist_codes, 'dist_sizes': dist_sizes, 'is_parkrun': is_parkrun, 'rolling_avg': rolling_avg,
            'temps': ag_temps, 'solar': ag_solar, 'elev_gain': ag_elev, 'undulation': ag_undulation,
            'surfaces': ag_surfaces, 'dists': ag_dists, 'names': ag_names}


# ============================================================================
# v51.7: ZONE & RACE CONFIGURATION
# ============================================================================
from config import PEAK_CP_WATTS as _cfg_cp, ATHLETE_MASS_KG as _cfg_mass
from config import ATHLETE_LTHR as _cfg_lthr, ATHLETE_MAX_HR as _cfg_maxhr
from config import PLANNED_RACES as _cfg_races
from config import POWER_MODE as _cfg_power_mode
from config import ATHLETE_NAME as _cfg_name
PEAK_CP_WATTS_DASH = _cfg_cp

# v53: If PEAK_CP was bootstrapped by StepB (config value is placeholder 300W),
# back-calculate effective PEAK_CP from the Master's CP and RFL columns.
# This ensures zones, race readiness, and all dashboard sections use the
# bootstrapped value rather than the placeholder.
_PEAK_CP_OVERRIDDEN = False

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
# v53: Override with data-driven factors from Master if available
# These are populated by bootstrap_peak_speed in StepB
def _load_race_factors_from_master(df_in):
    """Read data-driven race factors from Master (written by StepB bootstrap)."""
    global RACE_POWER_FACTORS_DASH
    if df_in is None or len(df_in) == 0:
        return
    latest = df_in.iloc[-1]
    _mode_suffix = {'gap': '_gap', 'sim': '_sim'}.get(_cfg_power_mode, '')
    _factor_map = {
        '5K': f'race_factor_5k{_mode_suffix}',
        '10K': f'race_factor_10k{_mode_suffix}',
        'HM': f'race_factor_hm{_mode_suffix}',
        'Mara': f'race_factor_marathon{_mode_suffix}',
    }
    updated = False
    for key, col in _factor_map.items():
        val = latest.get(col)
        if val is None and _mode_suffix:
            # Fallback to Stryd factors if mode-specific not available
            val = latest.get(col.replace(_mode_suffix, ''))
        if pd.notna(val) and float(val) > 0.5 and float(val) < 1.5:
            RACE_POWER_FACTORS_DASH[key] = round(float(val), 4)
            updated = True
    if updated:
        # Extrapolate Sub-5K: slightly above 5K factor
        _f5k = RACE_POWER_FACTORS_DASH.get('5K', 1.05)
        RACE_POWER_FACTORS_DASH['Sub-5K'] = round(_f5k * 1.02, 4)  # ~2% above 5K
        print(f"  Data-driven race factors: {RACE_POWER_FACTORS_DASH}")
def _distance_km_to_key(km):
    """Map distance_km to race category key."""
    if km <= 3.5: return 'Sub-5K'
    if km <= 7.5: return '5K'
    if km <= 15.0: return '10K'
    if km <= 30.0: return 'HM'
    return 'Mara'

if _cfg_races is not None:
    PLANNED_RACES_DASH = sorted([
        {'name': r['name'], 'date': r['date'], 'distance_key': _distance_km_to_key(r['distance_km']),
         'distance_km': r['distance_km'], 'priority': r.get('priority', 'B'), 'surface': r.get('surface', 'road')}
        for r in _cfg_races
    ], key=lambda r: r['date'])
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
                
                run_entry['_npz_path'] = npz_path  # stored for tail-slice in race readiness
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
    
    # ── Race week planner data ──
    # Recent runs with TSS for forward projection (last 14 days)
    # v52: Aggregate multiple runs on same day (warmup + race + cooldown)
    import math as _rw_math
    _TC_CTL, _TC_ATL = 42, 7
    _ALPHA_CTL = 1 - _rw_math.exp(-1.0 / _TC_CTL)
    _ALPHA_ATL = 1 - _rw_math.exp(-1.0 / _TC_ATL)
    _rw_recent_tss = []
    _today_dt = df['date'].iloc[-1] if len(df) > 0 else pd.Timestamp.now()
    _14d_ago = _today_dt - pd.Timedelta(days=14)
    _tss_col = 'TSS' if 'TSS' in df.columns else None
    _name_col = 'activity_name' if 'activity_name' in df.columns else 'Activity_Name'
    _race_col = 'race_flag' if 'race_flag' in df.columns else 'Race'
    if _tss_col:
        _day_agg = {}  # date_str -> {name, tss, race}
        for _, _row in df[df['date'] > _14d_ago].iterrows():
            _dstr = _row['date'].strftime('%Y-%m-%d')
            _rname = str(_row.get(_name_col, ''))[:40] if pd.notna(_row.get(_name_col)) else ''
            _rtss = round(float(_row[_tss_col])) if pd.notna(_row.get(_tss_col)) else 0
            _rrace = bool(_row.get(_race_col, 0) == 1)
            if _dstr in _day_agg:
                _day_agg[_dstr]['tss'] += _rtss
                _day_agg[_dstr]['race'] = _day_agg[_dstr]['race'] or _rrace
                # Use race activity name if available, otherwise keep first
                if _rrace and _rname:
                    _day_agg[_dstr]['name'] = _rname
            else:
                _day_agg[_dstr] = {'date': _dstr, 'name': _rname, 'tss': _rtss, 'race': _rrace}
        _rw_recent_tss = list(_day_agg.values())
    
    # Inject planned sessions as future "decided" entries so the taper solver
    # uses them instead of its template when they fall in the taper window
    try:
        _daily_full = pd.read_excel(MASTER_FILE, sheet_name=1)  # Daily sheet
        _daily_full['Date'] = pd.to_datetime(_daily_full['Date'])
        _has_planned_col = 'Planned_Description' in _daily_full.columns
        if _has_planned_col:
            _future_planned = _daily_full[
                (_daily_full['Date'] > pd.Timestamp.now()) &
                (_daily_full['Planned_Description'].notna()) &
                (_daily_full['Planned_Description'] != '')
            ]
            for _, _fp in _future_planned.iterrows():
                _fp_dstr = _fp['Date'].strftime('%Y-%m-%d')
                _fp_source = str(_fp.get('Planned_Source', ''))
                # Skip race entries — the taper solver handles race day specially (TSS=0, morning TSB)
                if _fp_source == 'race':
                    continue
                if _fp_dstr not in _day_agg:
                    _rw_recent_tss.append({
                        'date': _fp_dstr,
                        'name': str(_fp['Planned_Description']),
                        'tss': round(float(_fp.get('TSS_Running', 0))),
                        'race': str(_fp.get('Planned_Source', '')) == 'race',
                        'planned': True  # Flag for taper solver display
                    })
            if len(_future_planned) > 0:
                print(f"  Injected {len(_future_planned)} planned sessions into taper data")
    except Exception as _e:
        pass  # Daily sheet may not have planned columns yet
    
    # Current CTL/ATL for taper solver starting point
    # Use YESTERDAY's end-of-day CTL/ATL from Daily sheet, not today's.
    # The solver's forward loop starts at day 0 (today) and applies today's
    # TSS from recent_runs — using today's post-run CTL/ATL would double-count.
    _latest = df.iloc[-1] if len(df) > 0 else None
    _rw_ctl = 0
    _rw_atl = 0
    
    # Pre-race morning CTL/ATL/TSB
    # = previous day's end-of-day values decayed one step with TSS=0
    # StepB formula: ctl_new = ctl_old + (0 - ctl_old) / tc = ctl_old * (1 - 1/tc)
    _pre_race_tsb = {}
    _daily_lookup = {}
    if _tss_col and 'CTL' in df.columns:
        # Read Daily sheet for day-by-day CTL/ATL
        try:
            _daily = pd.read_excel(MASTER_FILE, sheet_name=1)  # Daily sheet
            _daily['Date'] = pd.to_datetime(_daily['Date'])
            for _, _dr in _daily.iterrows():
                _daily_lookup[_dr['Date'].strftime('%Y-%m-%d')] = {
                    'ctl': _dr.get('CTL'), 'atl': _dr.get('ATL')
                }
        except Exception:
            _daily_lookup = {}
    
    # Set taper starting point: yesterday's end-of-day CTL/ATL
    from datetime import date as _rw_date_cls
    _yesterday_str = (_rw_date_cls.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    _yesterday_daily = _daily_lookup.get(_yesterday_str)
    if _yesterday_daily and all(pd.notna(v) for v in [_yesterday_daily['ctl'], _yesterday_daily['atl']]):
        _rw_ctl = round(float(_yesterday_daily['ctl']), 1)
        _rw_atl = round(float(_yesterday_daily['atl']), 1)
    elif _latest is not None:
        # Fallback: use latest run's CTL/ATL (may double-count today, but better than 0)
        _rw_ctl = round(float(_latest['CTL']), 1) if pd.notna(_latest.get('CTL')) else 0
        _rw_atl = round(float(_latest['ATL']), 1) if pd.notna(_latest.get('ATL')) else 0
    
    if _daily_lookup:
        for _idx, _row in df[df[_race_col] == 1].iterrows():
            _r_date = _row['date']
            _prev_date_str = (_r_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            _prev = _daily_lookup.get(_prev_date_str)
            if not _prev or not all(pd.notna(v) for v in [_prev['ctl'], _prev['atl']]):
                continue
            # Decay one day with TSS=0: Banister exponential
            _ctl_pre = _prev['ctl'] * (1 - _ALPHA_CTL)
            _atl_pre = _prev['atl'] * (1 - _ALPHA_ATL)
            _tsb_pre = _ctl_pre - _atl_pre
            _ratio_pre = (_tsb_pre / _ctl_pre * 100) if _ctl_pre > 0 else 0
            _date_str = _r_date.strftime('%Y-%m-%d')
            _pre_race_tsb[_date_str] = {
                'ctl_pre': round(_ctl_pre, 1),
                'atl_pre': round(_atl_pre, 1),
                'tsb_pre': round(_tsb_pre, 1),
                'tsb_pct': round(_ratio_pre, 1),
            }
    
    return {
        'runs': runs, 'current_cp': current_cp, 'current_rfl': round(current_rfl, 4),
        're_p90': re_p90, 'gap_target_paces': gap_target_paces_local,
        'rfl_proj_per_day': _rfl_proj_per_day,
        # Race week planner data
        'recent_tss': _rw_recent_tss,
        'current_ctl': _rw_ctl, 'current_atl': _rw_atl,
        'pre_race_tsb': _pre_race_tsb,
        'daily_lookup': _daily_lookup,
    }


# ============================================================================
# ADAPTIVE TAPER SOLVER (module-level, shared by zone HTML + chart JSON)
# ============================================================================
import math as _taper_math
from datetime import timedelta as _taper_td

_RWP_TSB_TARGETS = {
    '5K':  {'A': (5, 25),  'B': (2, 18)},
    '10K': {'A': (5, 25),  'B': (2, 18)},
    'HM':  {'A': (5, 32),  'B': (2, 22)},
    'Mara':{'A': (10, 35), 'B': (5, 28)},
}
# Distance-interpolated TSB targets (anchored at 5K/10K/HM/Mara)
_RWP_TARGET_ANCHORS = [
    (5.0,   5, 25, 2, 18),
    (10.0,  5, 25, 2, 18),
    (21.1,  5, 32, 2, 22),
    (42.2, 10, 35, 5, 28),
]
def _rwp_interp_targets(dist_km):
    """Interpolate TSB target percentages by race distance."""
    anchors = _RWP_TARGET_ANCHORS
    if dist_km <= anchors[0][0]:
        return {'A': (anchors[0][1], anchors[0][2]), 'B': (anchors[0][3], anchors[0][4])}
    if dist_km >= anchors[-1][0]:
        return {'A': (anchors[-1][1], anchors[-1][2]), 'B': (anchors[-1][3], anchors[-1][4])}
    for i in range(len(anchors) - 1):
        if anchors[i][0] <= dist_km <= anchors[i+1][0]:
            f = (dist_km - anchors[i][0]) / (anchors[i+1][0] - anchors[i][0])
            lo, hi = anchors[i], anchors[i+1]
            return {
                'A': (round(lo[1] + f*(hi[1]-lo[1])), round(lo[2] + f*(hi[2]-lo[2]))),
                'B': (round(lo[3] + f*(hi[3]-lo[3])), round(lo[4] + f*(hi[4]-lo[4]))),
            }
    return _RWP_TSB_TARGETS.get('5K')
# CTL-scaled taper templates: (days_before_race, session_name, tss_ratio_of_ctl, reducible?)
_RWP_TEMPLATES = {
    '5K': [
        (7, 'Easy + race-pace surges', 1.50, True),
        (6, 'Easy', 1.30, True),
        (5, 'Easy + race-pace surges', 1.15, True),
        (4, 'Easy', 1.00, True),
        (3, 'Shakeout + race-pace strides', 0.55, True),
        (2, 'Rest', 0.00, False),
        (1, 'Shakeout + race-pace strides', 0.50, False),
    ],
    '10K': [
        (7, 'Easy + race-pace surges', 1.50, True),
        (6, 'Easy', 1.30, True),
        (5, 'Easy + race-pace surges', 1.15, True),
        (4, 'Easy', 1.00, True),
        (3, 'Shakeout + race-pace strides', 0.55, True),
        (2, 'Rest', 0.00, False),
        (1, 'Shakeout + race-pace strides', 0.50, False),
    ],
    'HM': [
        (7, 'Easy + race-pace surges', 1.50, True),
        (6, 'Moderate + HM-pace km', 1.35, True),
        (5, 'Easy', 1.15, True),
        (4, 'Easy + race-pace surges', 1.00, True),
        (3, 'Shakeout + race-pace strides', 0.55, True),
        (2, 'Rest', 0.00, False),
        (1, 'Shakeout + race-pace strides', 0.50, False),
    ],
    'Mara': [
        (10, 'Easy + MP surges', 2.00, True),
        (9, 'Easy', 1.75, True),
        (8, 'Moderate + MP km', 1.50, True),
        (7, 'Easy', 1.30, True),
        (6, 'Easy + MP surges', 1.15, True),
        (5, 'Easy', 1.00, True),
        (4, 'Easy', 0.65, True),
        (3, 'Rest', 0.00, False),
        (2, 'Shakeout + race-pace strides', 0.50, False),
        (1, 'Rest', 0.00, False),
    ],
}
_RWP_DUR_RATIOS = {'Easy': 0.70, 'Moderate': 0.65, 'Shakeout': 0.35, 'Light': 0.35, 'Rest': 0, 'Race': 0}
import math as _rwp_math
_RWP_ALPHA_C = 1 - _rwp_math.exp(-1.0/42)  # Banister: 1 - exp(-1/tau)
_RWP_ALPHA_A = 1 - _rwp_math.exp(-1.0/7)
_RWP_DECAY_C = 1 - _RWP_ALPHA_C  # For zero-TSS day: ctl_next = ctl * decay
_RWP_DECAY_A = 1 - _RWP_ALPHA_A

def _rwp_project(ctl, atl, tss):
    return (ctl + (tss - ctl) * _RWP_ALPHA_C,
            atl + (tss - atl) * _RWP_ALPHA_A)

def _rwp_duration(name, ctl_ref):
    for key, ratio in _RWP_DUR_RATIOS.items():
        if key.lower() in name.lower():
            return max(10, round(ctl_ref * ratio / 5) * 5)
    return max(10, round(ctl_ref * 0.50 / 5) * 5)

def _solve_taper(ctl0, atl0, days_to_race, ctl_ref, dist_cat, priority,
                 today_date, recent_runs, race_name, dist_km=None,
                 planned_sessions=None):
    """Adaptive taper: CTL-scaled template -> forward project -> reduce if out of zone.

    If planned_sessions is provided (dict of date_iso -> {description, tss}),
    those sessions override the generic template for matching days.

    Returns (plan_info_dict, results_list) where results_list has one entry per day
    from tomorrow through race day, each with date/session/tss/ctl/atl/tsb/tag.
    """
    # Use distance-interpolated targets if dist_km provided, else fall back to category
    if dist_km is not None:
        tgt = _rwp_interp_targets(dist_km).get(priority, (2, 18))
    else:
        tgt = _RWP_TSB_TARGETS.get(dist_cat, _RWP_TSB_TARGETS['5K']).get(priority, (2, 18))
    
    # Build initial plan from template (only days we have left, excluding D-0)
    tpl = [(d, n, r, red) for d, n, r, red in
           _RWP_TEMPLATES.get(dist_cat, _RWP_TEMPLATES['5K'])
           if 1 <= d <= days_to_race]
    
    plan = []
    for d, name, ratio, reducible in tpl:
        tss = round(ctl_ref * ratio / 5) * 5
        dur = 0 if tss == 0 else _rwp_duration(name, ctl_ref)
        plan.append({'d': d, 'name': name, 'tss': tss, 'dur': dur,
                     'ratio': ratio, 'protected': not reducible})
    plan.sort(key=lambda x: -x['d'])  # furthest-out first
    
    # Track which days have actual completed runs (not modifiable)
    actual_tss = {}
    for p in plan:
        day_date = today_date + _taper_td(days=(days_to_race - p['d']))
        dstr = day_date.strftime('%Y-%m-%d')
        if dstr in recent_runs:
            actual_tss[p['d']] = recent_runs[dstr]
    
    def evaluate(plan):
        c, a = ctl0, atl0
        results = []
        
        # Lookback: show completed days in the 6 days before race day
        # v52: Window is [race-6, race-1]. Only look back before today for days in that window.
        # e.g. race Sun 8 Mar, today Mon 2 Mar (dtr=6): lookback=0, all forward
        # e.g. race Sun 8 Mar, today Thu 5 Mar (dtr=3): lookback=3, show Mon-Wed done
        lookback_days = max(0, 6 - days_to_race)
        lookback_rows = []
        for lb in range(lookback_days, 0, -1):
            lb_date = today_date + _taper_td(days=-lb)
            lb_dstr = lb_date.strftime('%Y-%m-%d')
            lb_days_rem = days_to_race + lb
            if lb_dstr in recent_runs:
                act = recent_runs[lb_dstr]
                lookback_rows.append({
                    'date': lb_date, 'days_rem': lb_days_rem, 'session': act['name'],
                    'tss': act['tss'], 'ctl': 0, 'atl': 0, 'tsb': 0,
                    'tag': 'done', 'is_race': act.get('race', False),
                    'lookback': True,
                })
        
        # Reverse-calculate CTL/ATL for lookback rows
        # Start from ctl0/atl0 and walk backwards: ctl_prev = (ctl_now - tss*(1-dc)) / dc
        if lookback_rows:
            c_back, a_back = ctl0, atl0
            # Walk backwards through lookback rows (they're in chronological order)
            # Reverse them, peel off TSS, then re-forward
            tss_seq = [r['tss'] for r in lookback_rows]
            # Reverse-project to get state before first lookback day
            for tss_val in reversed(tss_seq):
                c_back = (c_back - tss_val * _RWP_ALPHA_C) / _RWP_DECAY_C
                a_back = (a_back - tss_val * _RWP_ALPHA_A) / _RWP_DECAY_A
            # Now forward-project to fill in actual CTL/ATL/TSB
            c_fwd, a_fwd = c_back, a_back
            for r in lookback_rows:
                c_fwd, a_fwd = _rwp_project(c_fwd, a_fwd, r['tss'])
                r['ctl'] = c_fwd
                r['atl'] = a_fwd
                r['tsb'] = c_fwd - a_fwd
        
        # Forward projection: today through race day
        for i_d in range(0, days_to_race + 1):
            day_date = today_date + _taper_td(days=i_d)
            days_rem = days_to_race - i_d
            dstr = day_date.strftime('%Y-%m-%d')
            is_race = (days_rem == 0)
            
            if dstr in recent_runs:
                act = recent_runs[dstr]
                sess, tss_val, tag = act['name'], act['tss'], 'done'
            elif is_race:
                sess, tss_val, tag = f'\U0001f3c1 {race_name}', 0, 'race'
            elif planned_sessions and dstr in planned_sessions:
                # Use uploaded training plan session instead of generic template
                _ps = planned_sessions[dstr]
                sess, tss_val = _ps['description'], _ps['tss']
                tag = 'today' if i_d == 0 else 'plan'
            else:
                match = [p for p in plan if p['d'] == days_rem]
                if match:
                    p = match[0]
                    dur_str = f"{p['dur']}'" if p['dur'] > 0 else ''
                    sess = f"{dur_str} {p['name']}".strip() if dur_str else p['name']
                    tss_val = p['tss']
                    tag = 'today' if i_d == 0 else 'plan'
                else:
                    sess, tss_val = 'Easy', round(ctl_ref * 0.40 / 5) * 5
                    tag = 'today' if i_d == 0 else 'plan'
            
            c, a = _rwp_project(c, a, tss_val)
            results.append({
                'date': day_date, 'days_rem': days_rem, 'session': sess,
                'tss': tss_val, 'ctl': c, 'atl': a, 'tsb': c - a,
                'tag': tag, 'is_race': is_race,
            })
        return lookback_rows + results, c, c - a
    
    results, race_ctl, race_tsb = evaluate(plan)
    tsb_lo = race_ctl * tgt[0] / 100
    mode = 'template'
    
    # If too fatigued, progressively reduce planned (non-actual) sessions
    if race_tsb < tsb_lo:
        reducible = [i for i, p in enumerate(plan)
                     if not p['protected'] and p['tss'] > 0
                     and p['d'] not in actual_tss]
        reducible.sort(key=lambda i: -plan[i]['d'])
        
        for idx in reducible:
            # Try halving first
            plan[idx]['tss'] = round(plan[idx]['tss'] * 0.5 / 5) * 5
            plan[idx]['dur'] = max(10, round(plan[idx]['dur'] * 0.5 / 5) * 5)
            plan[idx]['name'] = 'Light easy'
            results, race_ctl, race_tsb = evaluate(plan)
            tsb_lo = race_ctl * tgt[0] / 100
            if race_tsb >= tsb_lo:
                mode = 'adapted'
                break
            
            # Try full rest
            plan[idx]['tss'] = 0
            plan[idx]['dur'] = 0
            plan[idx]['name'] = 'Rest'
            results, race_ctl, race_tsb = evaluate(plan)
            tsb_lo = race_ctl * tgt[0] / 100
            if race_tsb >= tsb_lo:
                mode = 'adapted'
                break
        else:
            mode = 'adapted_max'
            results, race_ctl, race_tsb = evaluate(plan)
    
    return {'plan': plan, 'mode': mode}, results


# ============================================================================
# v51.7: ZONE HTML GENERATOR
# ============================================================================
def _generate_zone_html(zone_data, stats=None, upcoming_sessions=None):
    """Generate the Training Zones, Race Readiness, and Weekly Zone Volume HTML."""
    if zone_data is None:
        return '<!-- zones: no data -->'
    
    cp = zone_data['current_cp']
    re_p90 = zone_data['re_p90']
    _daily_lookup = zone_data.get('daily_lookup', {})
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
    # v53: Uses data-driven factors from bootstrap (via RACE_POWER_FACTORS_DASH)
    import math as _math
    _pd_anchors = [
        (3.0, RACE_POWER_FACTORS_DASH.get('Sub-5K', 1.07)),
        (5.0, RACE_POWER_FACTORS_DASH.get('5K', 1.05)),
        (10.0, RACE_POWER_FACTORS_DASH.get('10K', 1.00)),
        (21.097, RACE_POWER_FACTORS_DASH.get('HM', 0.95)),
        (42.195, RACE_POWER_FACTORS_DASH.get('Mara', 0.90)),
    ]
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
    _c28 = _today - _td(days=42)
    
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

    # ── Long run readiness + Tempo volume (8w / 2w windows) ──
    # Long run readiness: runs with duration >= 60 minutes.
    # Count qualifying runs and total minutes beyond threshold in those runs.
    # Tempo volume: total minutes at Z3+ (HR >= LTHR*0.90) beyond threshold.
    # Long run readiness + tempo volume for HM+ race cards.
    # Reports total mins beyond 60 min + Z3+ mins beyond 60 min.
    # Windows: 14d and 42d.
    _c14_lr      = _today - _td(days=14)
    _c42         = _today - _td(days=42)
    _z3_hr_floor = round(_cfg_lthr * 0.90)  # Z3 lower bound (LTHR * 0.90)

    # _lr_data[race_idx] = dict with keys:
    #   threshold_label (int, nearest 10min), total_14, total_56, z3_14, z3_56
    # None if race is past or < 15km.
    _lr_data = {}

    for _lr_idx, _lr_race in enumerate(PLANNED_RACES_DASH):
        try:
            _lr_race_dt = _dt.strptime(_lr_race['date'], '%Y-%m-%d').date()
            if _lr_race_dt < _today:
                _lr_data[_lr_idx] = None
                continue
        except (ValueError, KeyError):
            pass

        if _lr_race.get('distance_km', 0) < 15.0:
            _lr_data[_lr_idx] = None
            continue

        # Predicted finish time — same logic as race card
        _lr_dist_km  = _lr_race.get('distance_km', 21.097)
        _lr_surface  = _lr_race.get('surface', 'road')
        _lr_road_fac = _road_cp_factor(_lr_dist_km)
        _lr_sf       = _surface_factors.get(_lr_surface, _surface_factors['road'])
        _lr_factor   = _lr_road_fac * _lr_sf['power_mult']
        _lr_re       = re_p90 * _lr_sf['re_mult']
        _lr_pw       = round(cp * _lr_factor)
        _lr_dist_m   = _lr_dist_km * 1000
        _lr_dk       = _lr_race.get('distance_key', 'HM')
        _lr_stepb_key_map = {'5K': '5k_raw', '10K': '10k_raw', 'HM': 'hm_raw', 'Mara': 'marathon_raw'}
        _lr_stepb_std_km = {'5K': 5.0, '10K': 10.0, 'HM': 21.097, 'Mara': 42.195}
        _lr_is_std = abs(_lr_dist_km - _lr_stepb_std_km.get(_lr_dk, 0)) < 0.1
        _lr_rp       = stats.get('race_predictions', {}) if stats else {}
        _lr_pred_s = 0
        if _lr_is_std and _lr_surface == 'road':
            if _cfg_power_mode in ('gap', 'sim'):
                _lr_mode_preds = _lr_rp.get(f'_mode_{_cfg_power_mode}', {})
                _lr_pred_s = _lr_mode_preds.get(_lr_stepb_key_map.get(_lr_dk, ''), 0)
            else:
                _lr_pred_s = _lr_rp.get(_lr_stepb_key_map.get(_lr_dk, ''), 0)
        if not (_lr_pred_s and _lr_pred_s > 0) and _lr_surface == 'road':
            # Riegel interpolation from mode-specific predictions
            import math as _m
            _lr_pred_order = [('5K', 5.0, '5k_raw'), ('10K', 10.0, '10k_raw'), ('HM', 21.097, 'hm_raw'), ('Mara', 42.195, 'marathon_raw')]
            if _cfg_power_mode in ('gap', 'sim'):
                _lr_isrc = _lr_rp.get(f'_mode_{_cfg_power_mode}', {})
            else:
                _lr_isrc = _lr_rp
            _lr_pts = [(d, _lr_isrc.get(k, 0)) for _, d, k in _lr_pred_order if _lr_isrc.get(k, 0) > 0]
            if len(_lr_pts) >= 2:
                _lr_lo, _lr_hi = _lr_pts[0], _lr_pts[-1]
                for _li in range(len(_lr_pts) - 1):
                    if _lr_dist_km >= _lr_pts[_li][0] and _lr_dist_km <= _lr_pts[_li+1][0]:
                        _lr_lo, _lr_hi = _lr_pts[_li], _lr_pts[_li+1]
                        break
                if _lr_dist_km < _lr_pts[0][0]:
                    _lr_lo, _lr_hi = _lr_pts[0], _lr_pts[1]
                if _lr_dist_km > _lr_pts[-1][0]:
                    _lr_lo, _lr_hi = _lr_pts[-2], _lr_pts[-1]
                if _lr_lo[0] != _lr_hi[0]:
                    _lr_e = _m.log(_lr_hi[1] / _lr_lo[1]) / _m.log(_lr_hi[0] / _lr_lo[0])
                    _lr_pred_s = round(_lr_lo[1] * (_lr_dist_km / _lr_lo[0]) ** _lr_e)
                else:
                    _lr_pred_s = _lr_lo[1]
        if not (_lr_pred_s and _lr_pred_s > 0):
            _lr_speed  = (_lr_pw / ATHLETE_MASS_KG_DASH) * _lr_re
            _lr_pred_s = round(_lr_dist_m / _lr_speed) if _lr_speed > 0 else 0

        _lr_threshold_min   = 60.0  # fixed 60-minute threshold
        _lr_threshold_label = 60

        total_14 = 0.0; total_42 = 0.0
        z3_14    = 0.0; z3_42    = 0.0
        _thr_s   = _lr_threshold_min * 60.0  # threshold in seconds for NPZ slicing

        for _r in _runs_for_spec:
            _rd = _dt.strptime(_r['date'], '%Y-%m-%d').date()
            if _rd < _c42:
                continue
            _dur = _r.get('duration_min', 0) or 0
            if _dur < _lr_threshold_min:
                continue  # too short — not a qualifying long run

            # Minutes BEYOND the threshold only (not total run duration)
            _tail_min = _dur - _lr_threshold_min

            # Z3+ mins within the tail only
            _hz = _r.get('hz')
            _npz_path = _r.get('_npz_path')  # populated if NPZ was loaded
            if _hz and _npz_path:
                # Slice HR array from threshold_s onwards for accurate tail Z3
                try:
                    import numpy as _np2
                    _npz_data2 = _np2.load(_npz_path, allow_pickle=True)
                    _hr_full   = _npz_data2['hr_bpm']
                    _thr_idx   = min(int(_thr_s), len(_hr_full) - 1)
                    _hr_tail   = _hr_full[_thr_idx:]
                    _z3_secs   = float(_np2.sum(_hr_tail >= _z3_hr_floor))
                    _z3_mins   = _z3_secs / 60.0
                except Exception:
                    # Fallback: proportional estimate from full-run Z3 mins
                    _z3_full = _hz.get('Z3', 0) + _hz.get('Z4', 0) + _hz.get('Z5', 0)
                    _z3_mins = _z3_full * (_tail_min / _dur) if _dur > 0 else 0
            elif _hz:
                # Have summary zone data but no NPZ path — proportional estimate
                _z3_full = _hz.get('Z3', 0) + _hz.get('Z4', 0) + _hz.get('Z5', 0)
                _z3_mins = _z3_full * (_tail_min / _dur) if _dur > 0 else 0
            else:
                # No zone data — avg HR proxy, proportional
                _avg_hr  = _r.get('avg_hr', 0) or 0
                _z3_full = _dur if _avg_hr >= _z3_hr_floor else 0
                _z3_mins = _z3_full * (_tail_min / _dur) if _dur > 0 else 0

            total_42 += _tail_min
            z3_42    += _z3_mins
            if _rd >= _c14_lr:
                total_14 += _tail_min
                z3_14    += _z3_mins
        _lr_data[_lr_idx] = {
            'threshold_label': int(_lr_threshold_label),
            'total_14': round(total_14),
            'total_42': round(total_42),
            'z3_14':    round(z3_14),
            'z3_42':    round(z3_42),
        }

    # ── Uses module-level _solve_taper, _RWP_TSB_TARGETS ──
    
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
        # For bespoke distances (e.g. 16.2km), interpolate from StepB predictions via Riegel
        # Use mode-appropriate predictions: GAP mode → _mode_gap, Sim → _mode_sim
        _stepb_key_map = {'5K': '5k_raw', '10K': '10k_raw', 'HM': 'hm_raw', 'Mara': 'marathon_raw'}
        _stepb_std_km = {'5K': 5.0, '10K': 10.0, 'HM': 21.097, 'Mara': 42.195}
        _is_standard_dist = abs(dist_km - _stepb_std_km.get(key, 0)) < 0.1
        _rp = stats.get('race_predictions', {}) if stats else {}
        _stepb_raw = 0
        if _is_standard_dist and surface == 'road':
            # Pick mode-appropriate prediction source
            if _cfg_power_mode in ('gap', 'sim'):
                _mode_preds = _rp.get(f'_mode_{_cfg_power_mode}', {})
                _stepb_raw = _mode_preds.get(_stepb_key_map.get(key, ''), 0)
            else:
                _stepb_raw = _rp.get(_stepb_key_map.get(key, ''), 0)
        if _stepb_raw and _stepb_raw > 0:
            t = _stepb_raw
        elif not _is_standard_dist and surface == 'road':
            # Riegel interpolation from mode-specific StepB predictions
            import math as _m
            _pred_order = [('5K', 5.0, '5k_raw'), ('10K', 10.0, '10k_raw'), ('HM', 21.097, 'hm_raw'), ('Mara', 42.195, 'marathon_raw')]
            if _cfg_power_mode in ('gap', 'sim'):
                _interp_src = _rp.get(f'_mode_{_cfg_power_mode}', {})
            else:
                _interp_src = _rp
            _pts = [(d, _interp_src.get(k, 0)) for _, d, k in _pred_order if _interp_src.get(k, 0) > 0]
            if len(_pts) >= 2:
                # Find bracket
                _lo, _hi = _pts[0], _pts[-1]
                for _i in range(len(_pts) - 1):
                    if dist_km >= _pts[_i][0] and dist_km <= _pts[_i+1][0]:
                        _lo, _hi = _pts[_i], _pts[_i+1]
                        break
                if dist_km < _pts[0][0]:
                    _lo, _hi = _pts[0], _pts[1]
                if dist_km > _pts[-1][0]:
                    _lo, _hi = _pts[-2], _pts[-1]
                if _lo[0] != _hi[0]:
                    _e = _m.log(_hi[1] / _lo[1]) / _m.log(_hi[0] / _lo[0])
                    t = round(_lo[1] * (dist_km / _lo[0]) ** _e)
                else:
                    t = _lo[1]
            else:
                speed = (pw / ATHLETE_MASS_KG_DASH) * surface_re
                t = round(dist_m / speed) if speed > 0 else 0
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
        
        # Projected arrival CTL/TSB — computed after race week plan so we can use its values
        # For races with a race week plan (≤7d, A/B priority), use the plan's projection
        # For other races, fall back to zero-TSS decay from daily lookup
        _arrival_html = ''
        
        # Build training specificity HTML (HM+ races only)
        # Build training specificity HTML (HM+ races only)
        _lrd = _lr_data.get(race_idx)
        if dist_km >= 15.0 and _lrd:
            _thr  = _lrd['threshold_label']
            _t14  = _lrd['total_14'];  _t42  = _lrd['total_42']
            _z14  = _lrd['z3_14'];     _z42  = _lrd['z3_42']
            _tip_lr  = f"Total running time beyond {_thr} min"
            _tip_z3  = f"Minutes at Z3+ effort (HR ≥ {_z3_hr_floor} bpm) within those long runs only"
            _specificity_html = (
                f'<div style="grid-column:1/-1;font-size:0.68rem;color:var(--text-dim);'
                f'margin-top:8px;padding-top:8px;border-top:1px solid var(--border)">'
                f'Run time ≥ {_thr} min</div>'
                f'<div class="ws-tip"><div class="rv" style="color:#a78bfa">{_t14}'
                f'<span style="font-size:0.75rem;color:var(--text-dim)">min</span></div>'
                f'<div class="rl">Long run 14d</div>'
                f'<div class="tip">Time spent running beyond {_thr} mins in last 14 days</div></div>'
                f'<div class="ws-tip"><div class="rv" style="color:#a78bfa">{_t42}'
                f'<span style="font-size:0.75rem;color:var(--text-dim)">min</span></div>'
                f'<div class="rl">Long run 42d</div>'
                f'<div class="tip">Time spent running beyond {_thr} mins in last 42 days</div></div>'
                f'<div class="ws-tip"><div class="rv" style="color:#fb923c">{_z14}'
                f'<span style="font-size:0.75rem;color:var(--text-dim)">min</span></div>'
                f'<div class="rl">Z3+ in long 14d</div>'
                f'<div class="tip">Time spent running in Z3 or harder beyond {_thr} mins in last 14 days</div></div>'
                f'<div class="ws-tip"><div class="rv" style="color:#fb923c">{_z42}'
                f'<span style="font-size:0.75rem;color:var(--text-dim)">min</span></div>'
                f'<div class="rl">Z3+ in long 42d</div>'
                f'<div class="tip">Time spent running in Z3 or harder beyond {_thr} mins in last 42 days</div></div>'
            )
        else:
            _specificity_html = ''

        # Distance label — show for all races
        _standard_dists = {'5K': 5.0, '10K': 10.0, 'HM': 21.097, 'Mara': 42.195}
        _std_km = _standard_dists.get(key)
        if _std_km and abs(dist_km - _std_km) < 0.1:
            _dist_label = key
        else:
            _dist_label = f"{dist_km:.1f}km"

        race_cards += f'''<div class="rc">
            <div class="rh">
                <span class="rn">{race['name']} <span style="font-size:0.65rem;padding:2px 6px;border-radius:4px;background:{p_color}22;color:{p_color};font-weight:600;margin-left:6px;vertical-align:middle">{p_label}</span>{f' <span style="font-size:0.65rem;color:var(--text-dim)">{s_label}</span>' if surface != 'road' else ''}</span>
                <span class="rd">{race['date']} · {_dist_label} · {days_str}</span>
            </div>
            <div class="rs">
                <div class="power-only ws-tip"><div class="rv" style="color:var(--accent)" id="race-pw-{race_idx}">{pw}W</div><div class="rl">Target</div><div class="rx">±{band}W</div><div class="tip">Target power for this race distance and surface · ±3% band shown</div></div>
                <div class="pace-target" style="display:none"><div class="rv" style="color:#4ade80" id="race-pace-{race_idx}">{pace_str}</div><div class="rl">Target pace</div></div>
                <div class="ws-tip"><div class="rv" id="race-pred-{race_idx}">{t_str}</div><div class="rl">Predicted</div><div class="rx power-only">{pace_str}</div><div class="tip">Predicted finish time based on current fitness (RFL trend × CP)</div></div>
                <div class="ws-tip"><div class="rv" id="spec14_{race_idx}">{_spec_values.get(race_idx, (0, 0))[0]}<span style="font-size:0.75rem;color:var(--text-dim)">min</span></div><div class="rl">14d at effort</div><div class="tip">Minutes at race-specific effort (pace or power zone) in the last 14 days</div></div>
                <div class="ws-tip"><div class="rv" id="spec28_{race_idx}">{_spec_values.get(race_idx, (0, 0))[1]}<span style="font-size:0.75rem;color:var(--text-dim)">min</span></div><div class="rl">42d at effort</div><div class="tip">Minutes at race-specific effort (pace or power zone) in the last 42 days</div></div>
                {_specificity_html}
            </div>
            <div style="margin-top:6px">{taper_html}{_arrival_html}</div>\n'''
        
        # ── Race Week Plan (≤7 days, A or B priority) ──
        _rwp_html = ''
        if priority in ('A', 'B') and 0 < days_to <= 7:
            _rwp_dist_cat = 'Mara' if dist_km >= 35 else ('HM' if dist_km > 12 else ('10K' if dist_km > 5.5 else '5K'))
            _rwp_taper_label = _rwp_dist_cat
            # For bespoke distances, show actual distance in label
            _bespoke_check = {5.0: 0.3, 10.0: 0.3, 21.097: 0.5, 42.195: 1.5}
            _is_std_taper = any(abs(dist_km - sd) <= tol for sd, tol in _bespoke_check.items())
            if not _is_std_taper:
                _rwp_taper_label = f'{dist_km:.1f}km'
            _rwp_ctl0 = zone_data.get('current_ctl', 0)
            _rwp_atl0 = zone_data.get('current_atl', 0)
            _rwp_recent = {r['date']: r for r in zone_data.get('recent_tss', [])}

            # Build planned sessions lookup from uploaded training plan
            _rwp_planned = None
            if upcoming_sessions:
                _rwp_planned = {s['date_iso']: {'description': s['description'], 'tss': s['tss']}
                                for s in upcoming_sessions if not s.get('is_race')}

            _rwp_plan, _rwp_results = _solve_taper(
                _rwp_ctl0, _rwp_atl0, days_to, _rwp_ctl0, _rwp_dist_cat, priority,
                _today, _rwp_recent, race['name'], dist_km=dist_km,
                planned_sessions=_rwp_planned
            )
            
            _pcts = _rwp_interp_targets(dist_km).get(priority, (2, 18)) if dist_km else _RWP_TSB_TARGETS.get(_rwp_dist_cat, _RWP_TSB_TARGETS['5K']).get(priority, (2, 18))
            _rwp_race_day = _rwp_results[-1]
            _rwp_race_ctl = _rwp_race_day['ctl']
            _rwp_race_tsb = _rwp_race_day['tsb']
            _tsb_lo = _rwp_race_ctl * _pcts[0] / 100
            _tsb_hi = _rwp_race_ctl * _pcts[1] / 100
            _race_pct = (_rwp_race_tsb / _rwp_race_ctl * 100) if _rwp_race_ctl > 0 else 0
            
            _fmt_tsb = f"{'+'if _rwp_race_tsb>=0 else ''}{_rwp_race_tsb:.1f}"
            _fmt_pct = f"{'+'if _race_pct>=0 else ''}{_race_pct:.0f}%"
            _tgt_str = f"{'+'if _pcts[0]>=0 else ''}{_pcts[0]}% to {'+'if _pcts[1]>=0 else ''}{_pcts[1]}% of CTL"
            
            _adapted = _rwp_plan.get('mode', 'template')
            _adapt_note = ''
            if _adapted == 'adapted':
                _adapt_note = ' · plan adjusted for fatigue'
            elif _adapted == 'adapted_max':
                _adapt_note = ' · ⚠ max rest applied, still short'
            
            if _tsb_lo <= _rwp_race_tsb <= _tsb_hi:
                _v_cls, _v_txt = 'tsb-v-ok', f'✓ Race morning TSB {_fmt_tsb} ({_fmt_pct} of CTL) — in the zone · target {_tgt_str}{_adapt_note}'
            elif _rwp_race_tsb < _tsb_lo:
                _v_cls, _v_txt = 'tsb-v-lo', f'⚠ Race morning TSB {_fmt_tsb} ({_fmt_pct} of CTL) — {_tsb_lo - _rwp_race_tsb:.1f} below target · target {_tgt_str}{_adapt_note}'
            else:
                _v_cls, _v_txt = 'tsb-v-hi', f'Race morning TSB {_fmt_tsb} ({_fmt_pct} of CTL) — very fresh · target {_tgt_str}'
            
            _DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            _MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            
            _rwp_source_label = 'from coach plan' if _rwp_planned else f'{_rwp_taper_label} taper'
            _rwp_html = f'<div class="rwp"><div class="rwp-label">📋 Race Week Plan · {_rwp_source_label}</div>'
            _rwp_html += '<div class="dh"><span>Day</span><span>Session</span><span style="text-align:right">TSS</span><span style="text-align:right">CTL</span><span style="text-align:right">ATL</span><span style="text-align:right">TSB</span></div>'
            
            for _dd in _rwp_results:
                _dow = _DOW[_dd['date'].weekday()]
                _dm = _dd['date'].day
                _mn = _MON[_dd['date'].month - 1]
                _rcls = ' dr-race' if _dd['is_race'] else ''
                _tcls = 'dt-d' if _dd['tag'] == 'done' else ('dt-r' if _dd['tag'] == 'race' else ('dt-t' if _dd['tag'] == 'today' else 'dt-p'))
                _tlbl = 'done' if _dd['tag'] == 'done' else ('race' if _dd['tag'] == 'race' else ('today' if _dd['tag'] == 'today' else 'plan'))
                _tsb_col = 'color:#4ade80' if _tsb_lo <= _dd['tsb'] <= _tsb_hi else ('color:#fbbf24' if _dd['tsb'] < _tsb_lo else 'color:#3b82f6')
                _tsb_sign = '+' if _dd['tsb'] >= 0 else ''
                _tss_display = str(_dd["tss"]) if _dd["tss"] > 0 else ''
                _rwp_html += f'<div class="dr{_rcls}"><div class="dd"><b>{_dow}</b> {_dm} {_mn}</div><div class="ds">{_dd["session"]}<span class="dt {_tcls}">{_tlbl}</span></div><div class="dv dv-t">{_tss_display}</div><div class="dv dv-c">{_dd["ctl"]:.1f}</div><div class="dv dv-a">{_dd["atl"]:.1f}</div><div class="dv dv-s" style="{_tsb_col}">{_tsb_sign}{_dd["tsb"]:.1f}</div></div>'
            
            _rwp_html += f'<div class="tsb-v {_v_cls}">{_v_txt}</div>'
            _rwp_html += f'<div class="tsb-cw"><canvas id="tsbChart_{race_idx}"></canvas></div>'
            _rwp_html += '</div>'
            # Use race week plan's projection for arrival
            _arrival_html = f'<div style="margin-top:4px;font-size:0.72rem;color:var(--text-dim)">📊 Projected arrival: CTL <b style="color:var(--accent)">{_rwp_race_ctl:.0f}</b> · TSB <b style="color:{"#4ade80" if _rwp_race_tsb >= 0 else "#fbbf24"}">{"+" if _rwp_race_tsb >= 0 else ""}{_rwp_race_tsb:.0f}</b> ({"+" if _race_pct >= 0 else ""}{_race_pct:.0f}% of CTL)</div>'
        
        # Fallback arrival for races without a race week plan
        if not _arrival_html and days_to > 0 and _daily_lookup:
            import math as _arr_math
            _arr_alpha_c = 1 - _arr_math.exp(-1.0 / 42)
            _arr_alpha_a = 1 - _arr_math.exp(-1.0 / 7)
            _prev_date_str = (_race_dt - timedelta(days=1)).strftime('%Y-%m-%d')
            _prev_daily = _daily_lookup.get(_prev_date_str)
            if _prev_daily and all(pd.notna(v) for v in [_prev_daily['ctl'], _prev_daily['atl']]):
                _arr_ctl = _prev_daily['ctl'] * (1 - _arr_alpha_c)
                _arr_atl = _prev_daily['atl'] * (1 - _arr_alpha_a)
                _arr_tsb = _arr_ctl - _arr_atl
                _arr_pct = (_arr_tsb / _arr_ctl * 100) if _arr_ctl > 0 else 0
                _tsb_sign = '+' if _arr_tsb >= 0 else ''
                _tsb_color = '#4ade80' if _arr_tsb >= 0 else '#fbbf24'
                _arrival_html = f'<div style="margin-top:4px;font-size:0.72rem;color:var(--text-dim)">📊 Projected arrival: CTL <b style="color:var(--accent)">{_arr_ctl:.0f}</b> · TSB <b style="color:{_tsb_color}">{_tsb_sign}{_arr_tsb:.0f}</b> ({_tsb_sign}{_arr_pct:.0f}% of CTL)</div>'
        
        race_cards += _rwp_html + '</div>'''
    
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
    function calcSpecificity(){{const today=new Date();const c14=new Date(today);c14.setDate(c14.getDate()-14);const c28=new Date(today);c28.setDate(c28.getDate()-42);const useGAP=(typeof currentMode!=='undefined'&&currentMode==='gap');const zones=useGAP?RACE_PACE_Z:RACE_PW_Z;const modeKey='race';PLANNED_RACES.forEach((tgt,idx)=>{{let m14=0,m28=0;const atOrAbove=SPEC_ZONES[tgt.distance_key]||[tgt.distance_key];ZONE_RUNS.forEach(r=>{{const d=new Date(r.date);if(d<c28)return;const est=getZoneMins(r,modeKey,zones);let mins=0;atOrAbove.forEach(k=>{{mins+=(est[k]||0);}});if(d>=c14)m14+=mins;m28+=mins;}});const e14=document.getElementById('spec14_'+idx),e28=document.getElementById('spec28_'+idx);if(e14)e14.innerHTML=Math.round(m14)+'<span style="font-size:0.75rem;color:var(--text-dim)">min</span>';if(e28)e28.innerHTML=Math.round(m28)+'<span style="font-size:0.75rem;color:var(--text-dim)">min</span>';}});}}calcSpecificity();
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
# ============================================================
# MILESTONES FEATURE
# ============================================================

def get_recent_achievements(df, lookback_days=30):
    """Scan recent races for 'best X in Y years' achievements."""
    df = df.sort_values('date').reset_index(drop=True)
    if len(df) == 0:
        return []
    achievements = []
    cutoff = df['date'].max() - pd.Timedelta(days=lookback_days)
    races = df[df['race_flag'] == 1].copy()
    lookback_windows = [(1, '1 year'), (2, '2 years'), (3, '3 years'), (5, '5 years'), (10, '10 years')]

    # --- AGE GRADE ---
    recent_races = races[races['date'] >= cutoff]
    for _, r in recent_races.iterrows():
        ag = r.get('age_grade_pct', None)
        if pd.isna(ag):
            continue
        best_window = None
        for years_back, label in lookback_windows:
            lookback_start = r['date'] - pd.Timedelta(days=365 * years_back)
            prev = races[(races['date'] >= lookback_start) & (races['date'] < r['date']) & races['age_grade_pct'].notna()]
            if len(prev) == 0:
                best_window = (years_back, label)
            elif ag >= prev['age_grade_pct'].max():
                best_window = (years_back, label)
            else:
                break
        all_prev = races[(races['date'] < r['date']) & races['age_grade_pct'].notna()]
        is_all_time = len(all_prev) == 0 or ag >= all_prev['age_grade_pct'].max()
        if is_all_time:
            achievements.append({'date': r['date'].strftime('%Y-%m-%d'), 'type': 'age_grade',
                'title': f'Best Ever Age Grade: {ag:.1f}%', 'description': r.get('activity_name', ''),
                'icon': '\U0001f451', 'significance': 5, 'window': 'all-time'})
        elif best_window and best_window[0] >= 2:
            achievements.append({'date': r['date'].strftime('%Y-%m-%d'), 'type': 'age_grade',
                'title': f'Best Age Grade in {best_window[1]}: {ag:.1f}%', 'description': r.get('activity_name', ''),
                'icon': '\U0001f396\ufe0f', 'significance': min(best_window[0], 4), 'window': best_window[1]})

    # --- RACE TIMES (overall + surface-specific PBs) ---
    dist_names = {5.0: '5K', 10.0: '10K', 21.097: 'Half Marathon', 42.195: 'Marathon', 3.0: '3K'}
    
    # Surface grouping for surface-specific PBs
    def _surface_group(s):
        if pd.isna(s): return 'road'
        s = str(s).upper()
        if 'INDOOR' in s: return 'indoor'
        if s == 'TRACK': return 'track'
        if s in ('TRAIL', 'SNOW', 'HEAVY_SNOW'): return 'trail'
        return 'road'
    
    surface_labels = {'road': '', 'indoor': 'Indoor ', 'track': 'Track ', 'trail': 'Trail '}
    
    if 'surface' in races.columns:
        races = races.copy()
        races['_sgroup'] = races['surface'].apply(_surface_group)
    else:
        races = races.copy()
        races['_sgroup'] = 'road'
    
    for dist in [5.0, 10.0, 21.097, 42.195, 3.0]:
        dist_races = races[races['official_distance_km'] == dist].sort_values('date')
        recent_dist = dist_races[dist_races['date'] >= cutoff]
        for _, r in recent_dist.iterrows():
            t = r['elapsed_time_s']
            dist_name = dist_names.get(dist, f'{dist}km')
            hrs, mins, secs = int(t // 3600), int((t % 3600) // 60), int(t % 60)
            time_str = f'{hrs}:{mins:02d}:{secs:02d}' if hrs > 0 else f'{mins}:{secs:02d}'
            
            # --- Overall PB/best-in-period ---
            best_window = None
            for years_back, label in lookback_windows:
                lookback_start = r['date'] - pd.Timedelta(days=365 * years_back)
                prev = dist_races[(dist_races['date'] >= lookback_start) & (dist_races['date'] < r['date'])]
                if len(prev) == 0:
                    best_window = (years_back, label)
                elif t <= prev['elapsed_time_s'].min():
                    best_window = (years_back, label)
                else:
                    break
            all_prev = dist_races[dist_races['date'] < r['date']]
            is_pb = len(all_prev) == 0 or t <= all_prev['elapsed_time_s'].min()
            
            if is_pb and len(all_prev) > 0:
                achievements.append({'date': r['date'].strftime('%Y-%m-%d'), 'type': 'race_pb',
                    'title': f'{dist_name} PB: {time_str}', 'description': r.get('activity_name', ''),
                    'icon': '\U0001f3c5', 'significance': 5, 'window': 'all-time'})
            elif best_window and best_window[0] >= 2:
                achievements.append({'date': r['date'].strftime('%Y-%m-%d'), 'type': 'race_time',
                    'title': f'Fastest {dist_name} in {best_window[1]}: {time_str}', 'description': r.get('activity_name', ''),
                    'icon': '\u26a1', 'significance': min(best_window[0], 4), 'window': best_window[1]})
            
            # --- Surface-specific PB (only if surface != road, to avoid duplicating the overall PB) ---
            sgroup = r.get('_sgroup', 'road')
            if sgroup != 'road':
                surface_dist = dist_races[dist_races['_sgroup'] == sgroup].sort_values('date')
                surf_prev = surface_dist[surface_dist['date'] < r['date']]
                is_surface_pb = len(surf_prev) > 0 and t <= surf_prev['elapsed_time_s'].min()
                # Only show if it's a surface PB but NOT an overall PB (avoid double-reporting)
                if is_surface_pb and not is_pb:
                    slabel = surface_labels.get(sgroup, '')
                    achievements.append({'date': r['date'].strftime('%Y-%m-%d'), 'type': 'surface_pb',
                        'title': f'{slabel}{dist_name} PB: {time_str}', 'description': r.get('activity_name', ''),
                        'icon': '\U0001f3c5', 'significance': 4, 'window': f'{slabel.strip()} PB'})

    # --- RFL TREND (mode-aware) ---
    _rfl_trend_col = {'gap': 'RFL_gap_Trend', 'sim': 'RFL_sim_Trend'}.get(_cfg_power_mode, 'RFL_Trend')
    if _rfl_trend_col not in df.columns or df[_rfl_trend_col].dropna().empty:
        _rfl_trend_col = 'RFL_Trend'  # fallback
    recent_all = df[df['date'] >= cutoff]
    if len(recent_all) > 0 and _rfl_trend_col in df.columns:
        rfl_recent = recent_all[['date', _rfl_trend_col]].dropna()
        if len(rfl_recent) > 0:
            peak_row = rfl_recent.loc[rfl_recent[_rfl_trend_col].idxmax()]
            peak_val, peak_date = peak_row[_rfl_trend_col], peak_row['date']
            best_window = None
            for months_back, label in [(3, '3 months'), (6, '6 months'), (12, '1 year'), (24, '2 years')]:
                lookback_start = peak_date - pd.Timedelta(days=30 * months_back)
                prev = df[(df['date'] >= lookback_start) & (df['date'] < peak_date)][_rfl_trend_col].dropna()
                if len(prev) > 0 and peak_val >= prev.max():
                    best_window = (months_back, label)
                else:
                    break
            if best_window and best_window[0] >= 6:
                achievements.append({'date': peak_date.strftime('%Y-%m-%d'), 'type': 'fitness',
                    'title': f'Highest Fitness in {best_window[1]}: {peak_val:.1%}',
                    'description': 'RFL Trend peak', 'icon': '\U0001f4c8',
                    'significance': 2 if best_window[0] >= 12 else 1, 'window': best_window[1]})

    achievements.sort(key=lambda a: (-int(a['date'].replace('-','')), -a['significance']))
    
    # Deduplicate: per distance keep best overall achievement + best surface PB
    # For AG, keep only the most significant
    seen_overall = set()  # track dist for race_time/race_pb
    seen_surface = set()  # track dist+surface for surface_pb
    seen_ag = False
    deduped = []
    for a in achievements:
        if a['type'] in ('race_time', 'race_pb'):
            dist_word = a['title'].split(':')[0].split()[-1] if ':' in a['title'] else a['title'].split()[0]
            # Extract just the distance name (5K, 3K, etc)
            for dw in ['3K', '5K', '10K', 'Half', 'Marathon']:
                if dw in a['title']:
                    dist_word = dw
                    break
            if dist_word not in seen_overall:
                seen_overall.add(dist_word)
                deduped.append(a)
        elif a['type'] == 'surface_pb':
            key = a['title'].split(':')[0]  # e.g. "Indoor 3K PB"
            if key not in seen_surface:
                seen_surface.add(key)
                deduped.append(a)
        elif a['type'] == 'age_grade':
            if not seen_ag:
                seen_ag = True
                deduped.append(a)
        else:
            deduped.append(a)
    
    return deduped


def get_milestones_data(df):
    """Compute all-time milestones from master dataframe."""
    df = df.sort_values('date').reset_index(drop=True)
    if len(df) == 0:
        return {'milestones': [], 'next_milestones': [], 'recent_achievements': [], 'summary': {}}
    milestones = []

    # --- CUMULATIVE DISTANCE ---
    cum_dist = df['distance_km'].cumsum()
    for threshold, label, icon, importance in [
        (1000,'1,000 km','\U0001f30d',2),(2000,'2,000 km','\U0001f30d',1),(5000,'5,000 km','\U0001f30f',2),
        (10000,'10,000 km','\U0001f30e',3),(15000,'15,000 km','\U0001f310',2),(20000,'20,000 km','\U0001f6e4\ufe0f',3),
        (25000,'25,000 km','\U0001f3d4\ufe0f',3),(30000,'30,000 km','\U0001f680',3)]:
        idx = cum_dist[cum_dist >= threshold].index
        if len(idx) > 0:
            row = df.loc[idx[0]]
            milestones.append({'date': row['date'].strftime('%Y-%m-%d'), 'category': 'distance',
                'title': f'{label} Total Distance', 'description': f'Reached {label} on run #{idx[0]+1:,}',
                'icon': icon, 'importance': importance, 'value': threshold})

    # --- RUN COUNTS ---
    for threshold, icon, importance in [(100,'\U0001f4af',2),(250,'\U0001f4ca',1),(500,'\U0001f4ca',2),(1000,'\U0001f525',3),
            (1500,'\U0001f525',2),(2000,'\u26a1',3),(2500,'\u26a1',2),(3000,'\U0001f48e',3)]:
        if threshold <= len(df):
            row = df.iloc[threshold - 1]
            milestones.append({'date': row['date'].strftime('%Y-%m-%d'), 'category': 'runs',
                'title': f'Run #{threshold:,}', 'description': f'{row["distance_km"]:.1f}km \u2014 {row.get("activity_name", "")}',
                'icon': icon, 'importance': importance, 'value': threshold})

    # --- RACE PBs ---
    races = df[df['race_flag'] == 1].copy()
    dist_names = {1.0: '1K', 1.5: '1500m', 1.609: 'Mile', 3.0: '3K', 5.0: '5K', 
                  10.0: '10K', 15.0: '15K', 16.09: '10M', 21.097: 'Half Marathon', 42.195: 'Marathon'}
    # Core distances always included if they have data; others included if 2+ races
    core_dists = {3.0, 5.0, 10.0, 21.097, 42.195}
    race_dist_counts = races['official_distance_km'].value_counts()
    pb_dists = sorted(set(
        [d for d in core_dists if d in race_dist_counts.index] +
        [d for d, c in race_dist_counts.items() if c >= 2 and d in dist_names]
    ))
    for dist in pb_dists:
        dist_races = races[races['official_distance_km'] == dist].sort_values('date')
        if len(dist_races) == 0:
            continue
        running_best = dist_races['elapsed_time_s'].cummin()
        pb_runs = dist_races[dist_races['elapsed_time_s'] == running_best]
        for i, (idx, row) in enumerate(pb_runs.iterrows()):
            t = row['elapsed_time_s']
            hrs, mins, secs = int(t // 3600), int((t % 3600) // 60), int(t % 60)
            time_str = f'{hrs}:{mins:02d}:{secs:02d}' if hrs > 0 else f'{mins}:{secs:02d}'
            dist_name = dist_names.get(dist, f'{dist}km')
            is_first, is_latest = (i == 0), (i == len(pb_runs) - 1)
            importance = 3 if is_latest else 1
            name = row.get('activity_name', '')
            if isinstance(name, str) and len(name) > 50:
                name = name[:47] + '...'
            milestones.append({'date': row['date'].strftime('%Y-%m-%d'), 'category': 'pb',
                'title': f'{dist_name} PB: {time_str}', 'description': name if isinstance(name, str) else '',
                'icon': '\U0001f3c5' if is_latest else '\U0001f3af', 'importance': importance,
                'value': t, 'distance_km': dist, 'is_current_pb': is_latest})

    # --- Surface-specific time PBs (indoor/track only, current best) ---
    def _time_surface_group(s):
        if pd.isna(s): return 'road'
        s = str(s).upper()
        if 'INDOOR' in s: return 'indoor'
        if s == 'TRACK': return 'track'
        return 'road'
    
    time_surface_labels = {'indoor': 'Indoor ', 'track': 'Track '}
    if 'surface' in races.columns:
        races_sg = races.copy()
        races_sg['_sgroup'] = races_sg['surface'].apply(_time_surface_group)
        for dist in pb_dists:
            dist_name = dist_names.get(dist, f'{dist}km')
            dist_races = races_sg[races_sg['official_distance_km'] == dist]
            # Find overall best for comparison
            overall_best_t = dist_races['elapsed_time_s'].min() if len(dist_races) > 0 else None
            for sgrp in ['indoor', 'track']:
                surf_races = dist_races[dist_races['_sgroup'] == sgrp].sort_values('date')
                if len(surf_races) < 2:
                    continue  # need 2+ races on this surface to be meaningful
                best_idx = surf_races['elapsed_time_s'].idxmin()
                best = df.loc[best_idx]
                t = best['elapsed_time_s']
                # Skip if this is also the overall PB (already shown)
                if overall_best_t is not None and t <= overall_best_t:
                    continue
                hrs, mins, secs = int(t // 3600), int((t % 3600) // 60), int(t % 60)
                time_str = f'{hrs}:{mins:02d}:{secs:02d}' if hrs > 0 else f'{mins}:{secs:02d}'
                slabel = time_surface_labels[sgrp]
                name = best.get('activity_name', '')
                if isinstance(name, str) and len(name) > 50:
                    name = name[:47] + '...'
                milestones.append({'date': best['date'].strftime('%Y-%m-%d'), 'category': 'pb',
                    'title': f'{slabel}{dist_name} PB: {time_str}', 'description': name if isinstance(name, str) else '',
                    'icon': '\U0001f3c5', 'importance': 2, 'value': t,
                    'distance_km': dist, 'is_current_pb': True})

    # --- AG PBs per distance (current best only, + surface variants) ---
    def _ag_surface_group(s):
        if pd.isna(s): return 'road'
        s = str(s).upper()
        if 'INDOOR' in s: return 'indoor'
        if s == 'TRACK': return 'track'
        return 'road'

    ag_surface_labels = {'road': '', 'indoor': 'Indoor ', 'track': 'Track '}
    if 'age_grade_pct' in df.columns:
        ag_races = races[races['age_grade_pct'].notna()].copy()
        if len(ag_races) > 0:
            ag_races = ag_races.copy()
            ag_races['_sgroup'] = ag_races['surface'].apply(_ag_surface_group) if 'surface' in ag_races.columns else 'road'
            for dist in pb_dists:
                dist_name = dist_names.get(dist, f'{dist}km')
                dist_ag = ag_races[ag_races['official_distance_km'] == dist]
                if len(dist_ag) == 0:
                    continue
                # Overall AG PB for this distance
                best_idx = dist_ag['age_grade_pct'].idxmax()
                best = df.loc[best_idx]
                ag_val = best['age_grade_pct']
                t = best['elapsed_time_s']
                hrs, mins, secs = int(t // 3600), int((t % 3600) // 60), int(t % 60)
                time_str = f'{hrs}:{mins:02d}:{secs:02d}' if hrs > 0 else f'{mins}:{secs:02d}'
                name = best.get('activity_name', '')
                if isinstance(name, str) and len(name) > 50:
                    name = name[:47] + '...'
                milestones.append({'date': best['date'].strftime('%Y-%m-%d'), 'category': 'pb',
                    'title': f'{dist_name} AG: {ag_val:.1f}%', 'description': f'{time_str} \u2014 {name}',
                    'icon': '\U0001f3c6', 'importance': 2, 'value': ag_val,
                    'is_current_pb': True, 'is_ag_pb': True})
                # Surface-specific AG PBs (indoor/track only, if different from overall)
                for sgrp in ['indoor', 'track']:
                    surf_ag = dist_ag[dist_ag['_sgroup'] == sgrp]
                    if len(surf_ag) < 1:
                        continue
                    sbest_idx = surf_ag['age_grade_pct'].idxmax()
                    sbest = df.loc[sbest_idx]
                    sag = sbest['age_grade_pct']
                    if sbest_idx == best_idx:
                        continue  # same as overall, skip
                    st = sbest['elapsed_time_s']
                    shrs, smins, ssecs = int(st // 3600), int((st % 3600) // 60), int(st % 60)
                    stime_str = f'{shrs}:{smins:02d}:{ssecs:02d}' if shrs > 0 else f'{smins}:{ssecs:02d}'
                    sname = sbest.get('activity_name', '')
                    if isinstance(sname, str) and len(sname) > 50:
                        sname = sname[:47] + '...'
                    slabel = ag_surface_labels.get(sgrp, '')
                    milestones.append({'date': sbest['date'].strftime('%Y-%m-%d'), 'category': 'pb',
                        'title': f'{slabel}{dist_name} AG: {sag:.1f}%', 'description': f'{stime_str} \u2014 {sname}',
                        'icon': '\U0001f3c6', 'importance': 1, 'value': sag,
                        'is_current_pb': True, 'is_ag_pb': True})

    # --- FITNESS (RFL_Trend - mode-aware) ---
    _rfl_trend_col = {'gap': 'RFL_gap_Trend', 'sim': 'RFL_sim_Trend'}.get(_cfg_power_mode, 'RFL_Trend')
    if _rfl_trend_col not in df.columns or df[_rfl_trend_col].dropna().empty:
        _rfl_trend_col = 'RFL_Trend'  # fallback
    rfl = df[['date', _rfl_trend_col]].dropna(subset=[_rfl_trend_col])
    if len(rfl) > 0:
        peak_idx = rfl[_rfl_trend_col].idxmax()
        peak_row = df.loc[peak_idx]
        milestones.append({'date': peak_row['date'].strftime('%Y-%m-%d'), 'category': 'fitness',
            'title': f'Peak Fitness: {peak_row[_rfl_trend_col]:.1%}', 'description': 'All-time highest RFL',
            'icon': '\u2b50', 'importance': 3, 'value': peak_row[_rfl_trend_col]})
        for threshold, label, imp in [(0.50,'50%',1),(0.60,'60%',1),(0.70,'70%',2),(0.80,'80%',2),(0.90,'90%',3),(0.95,'95%',3)]:
            above = rfl[rfl[_rfl_trend_col] >= threshold]
            if len(above) > 0:
                first_row = df.loc[above.index[0]]
                milestones.append({'date': first_row['date'].strftime('%Y-%m-%d'), 'category': 'fitness',
                    'title': f'First {label} RFL', 'description': f'Running Fitness Level first reached {label}',
                    'icon': '\U0001f4c8', 'importance': imp, 'value': threshold})

    # --- AGE GRADE ---
    if 'age_grade_pct' in df.columns:
        ag = df[df['age_grade_pct'].notna() & (df['race_flag'] == 1)].copy()
        if len(ag) > 0:
            best_ag = df.loc[ag['age_grade_pct'].idxmax()]
            milestones.append({'date': best_ag['date'].strftime('%Y-%m-%d'), 'category': 'age_grade',
                'title': f'Best Age Grade: {best_ag["age_grade_pct"]:.1f}%',
                'description': best_ag.get('activity_name', ''), 'icon': '\U0001f451', 'importance': 3,
                'value': best_ag['age_grade_pct']})
            for threshold, label, level, imp in [(60,'60%','Regional',1),(65,'65%','Regional+',1),
                    (70,'70%','Local',2),(75,'75%','National',2),(80,'80%','National+',3)]:
                ag_sorted = ag.sort_values('date')
                above = ag_sorted[ag_sorted['age_grade_pct'].cummax() >= threshold]
                if len(above) > 0:
                    first_row = df.loc[above.index[0]]
                    milestones.append({'date': first_row['date'].strftime('%Y-%m-%d'), 'category': 'age_grade',
                        'title': f'First {label} Age Grade', 'description': f'{level} class',
                        'icon': '\U0001f396\ufe0f', 'importance': imp, 'value': threshold})

    # --- YEARLY VOLUME ---
    yearly = df.groupby(df['date'].dt.year).agg(total_dist=('distance_km', 'sum'))
    for threshold, label, imp in [(1000,'1,000 km Year',1),(1500,'1,500 km Year',2),(2000,'2,000 km Year',2),
            (2500,'2,500 km Year',3),(3000,'3,000 km Year',3)]:
        years_above = yearly[yearly['total_dist'] >= threshold]
        if len(years_above) > 0:
            fy = years_above.index[0]
            yr = df[df['date'].dt.year == fy].copy()
            cross = yr['distance_km'].cumsum()
            cidx = cross[cross >= threshold].index
            if len(cidx) > 0:
                cr = df.loc[cidx[0]]
                milestones.append({'date': cr['date'].strftime('%Y-%m-%d'), 'category': 'yearly',
                    'title': f'{label} ({fy})', 'description': f'First year reaching {label}',
                    'icon': '\U0001f4c5', 'importance': imp, 'value': threshold})

    # --- CONSISTENCY ---
    run_dates = sorted(df['date'].dt.date.unique())
    if len(run_dates) > 1:
        streak = max_streak = 1
        max_streak_end = best_start = cur_start = run_dates[0]
        for i in range(1, len(run_dates)):
            if (run_dates[i] - run_dates[i-1]).days == 1:
                streak += 1
                if streak > max_streak:
                    max_streak, max_streak_end, best_start = streak, run_dates[i], cur_start
            else:
                streak, cur_start = 1, run_dates[i]
        if max_streak >= 7:
            milestones.append({'date': max_streak_end.strftime('%Y-%m-%d'), 'category': 'consistency',
                'title': f'{max_streak}-Day Streak',
                'description': f'{best_start.strftime("%d %b")} to {max_streak_end.strftime("%d %b %Y")}',
                'icon': '\U0001f525', 'importance': 2 if max_streak >= 14 else 1, 'value': max_streak})

    df_c = df.copy()
    df_c['iso_year'] = df_c['date'].dt.isocalendar().year.astype(int)
    df_c['iso_week'] = df_c['date'].dt.isocalendar().week.astype(int)
    wc = df_c.groupby(['iso_year', 'iso_week']).size().reset_index(name='runs').sort_values(['iso_year', 'iso_week'])
    for min_r, label in [(4, '4+ runs/week'), (5, '5+ runs/week')]:
        streak = mx = 0; mx_row = None
        for _, wr in wc.iterrows():
            if wr['runs'] >= min_r:
                streak += 1
                if streak > mx: mx, mx_row = streak, wr
            else:
                streak = 0
        if mx >= 10 and mx_row is not None:
            ed = datetime.strptime(f"{int(mx_row['iso_year'])}-W{int(mx_row['iso_week']):02d}-7", "%G-W%V-%u").date()
            milestones.append({'date': ed.strftime('%Y-%m-%d'), 'category': 'consistency',
                'title': f'{mx} Weeks of {label}', 'description': 'Longest streak of consistent training',
                'icon': '\U0001f4c6', 'importance': 2 if mx >= 26 else 1, 'value': mx})

    # --- RACE COUNTS ---
    rc = (df['race_flag'] == 1).cumsum()
    for threshold, label, icon, imp in [(50,'50th Race','\U0001f3c1',1),(100,'100th Race','\U0001f3c1',2),
            (200,'200th Race','\U0001f3c1',2),(300,'300th Race','\U0001f3c1',3)]:
        idx = rc[rc >= threshold].index
        if len(idx) > 0:
            row = df.loc[idx[0]]
            milestones.append({'date': row['date'].strftime('%Y-%m-%d'), 'category': 'races',
                'title': label, 'description': row.get('activity_name', ''),
                'icon': icon, 'importance': imp, 'value': threshold})

    # --- NEXT MILESTONES ---
    next_ms = []
    td, tr, trc = df['distance_km'].sum(), len(df), int((df['race_flag']==1).sum())
    for thr, lab, ic in [(1000,'1,000 km','\U0001f30d'),(2000,'2,000 km','\U0001f30d'),(5000,'5,000 km','\U0001f30f'),
            (10000,'10,000 km','\U0001f30e'),(15000,'15,000 km','\U0001f310'),(20000,'20,000 km','\U0001f6e4\ufe0f'),
            (25000,'25,000 km','\U0001f3d4\ufe0f'),(30000,'30,000 km','\U0001f680')]:
        if td < thr:
            rem = thr - td
            rec = df[df['date'] >= df['date'].max() - pd.Timedelta(days=90)]
            da = rec['distance_km'].sum() / 90 if len(rec) > 0 else 0
            next_ms.append({'category':'distance','title':f'{lab} Total Distance','remaining':f'{rem:.0f} km to go',
                'est_days': int(rem/da) if da > 0 else None, 'icon':ic, 'pct_complete': td/thr*100})
            break
    for thr, ic in [(100,'\U0001f4af'),(250,'\U0001f4ca'),(500,'\U0001f4ca'),(1000,'\U0001f525'),
            (1500,'\U0001f525'),(2000,'\u26a1'),(2500,'\u26a1'),(3000,'\U0001f48e')]:
        if tr < thr:
            next_ms.append({'category':'runs','title':f'Run #{thr:,}','remaining':f'{thr-tr} runs to go',
                'icon':ic,'pct_complete':tr/thr*100})
            break
    for thr, ic in [(50,'\U0001f3c1'),(100,'\U0001f3c1'),(200,'\U0001f3c1'),(300,'\U0001f3c1')]:
        if trc < thr:
            next_ms.append({'category':'races','title':f'{thr}th Race','remaining':f'{thr-trc} races to go',
                'icon':ic,'pct_complete':trc/thr*100})
            break

    milestones.sort(key=lambda m: m['date'])
    recent_achievements = get_recent_achievements(df, lookback_days=60)
    
    # Sanitize numpy types for JSON serialization
    import numpy as np
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj
    milestones = _sanitize(milestones)
    recent_achievements = _sanitize(recent_achievements)
    return {'milestones': milestones, 'next_milestones': next_ms, 'recent_achievements': recent_achievements,
        'summary': {'total_distance_km': td, 'total_runs': tr, 'total_races': trc,
            'years_active': len(df['date'].dt.year.unique()),
            'first_run': df['date'].min().strftime('%Y-%m-%d'), 'latest_run': df['date'].max().strftime('%Y-%m-%d')}}


def _generate_recent_achievements_html(milestone_data):
    """Generate Recent Achievements card (hidden if none)."""
    recent = milestone_data.get('recent_achievements', [])
    if not recent:
        return '<!-- recent achievements: none -->'
    recent_json = json.dumps(recent)
    return f'''
    <div class="chart-container" id="recentAchievementsSection">
        <div class="chart-title">\U0001f525 Recent Achievements</div>
        <div class="chart-desc">Last 60 days</div>
        <div id="recentAchievementsList"></div>
    </div>
    <script>
    (function() {{
        const ra = {recent_json};
        function fmtDate(s) {{ const d=new Date(s); const m=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']; return d.getDate()+' '+m[d.getMonth()]; }}
        document.getElementById('recentAchievementsList').innerHTML = ra.map(a => {{
            const wb = a.window==='all-time'
                ? '<span style="font-size:9px;padding:1px 5px;border-radius:3px;background:#f59e0b22;color:#f59e0b;font-weight:600;">ALL-TIME</span>'
                : `<span style="font-size:9px;padding:1px 5px;border-radius:3px;background:#818cf822;color:#818cf8;font-weight:500;">${{a.window}}</span>`;
            const nm = a.description&&a.description.length>60?a.description.slice(0,57)+'...':(a.description||'');
            return `<div style="padding:8px 10px;border-bottom:1px solid var(--border,#2e3340);display:flex;align-items:center;gap:8px;">
                <span style="font-size:22px;">${{a.icon}}</span>
                <div style="flex:1;min-width:0;">
                    <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
                        <span style="font-weight:600;color:var(--text,#e4e7ef);font-size:13px;">${{a.title}}</span>
                        ${{wb}}
                    </div>
                    <div style="font-size:11px;color:var(--text-dim,#8b8fa3);margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${{nm}}</div>
                </div>
                <div style="font-size:11px;color:var(--text-dim,#8b8fa3);white-space:nowrap;">${{fmtDate(a.date)}}</div>
            </div>`;
        }}).join('');
    }})();
    </script>
    '''


def _generate_milestones_html(milestone_data):
    """Generate all-time Milestones card with toggle filters."""
    milestones = milestone_data.get('milestones', [])
    next_milestones = milestone_data.get('next_milestones', [])
    summary = milestone_data.get('summary', {})
    if not milestones:
        return '<!-- milestones: no data -->'
    mj = json.dumps(milestones)
    nj = json.dumps(next_milestones)
    tr, td, trc, ya = summary.get('total_runs',0), summary.get('total_distance_km',0), summary.get('total_races',0), summary.get('years_active',0)
    return f'''
    <div class="chart-container" id="milestonesSection">
        <div class="chart-title-row">
            <span class="chart-title">\U0001f3c6 Milestones</span>
            <div class="chart-toggle" id="milestoneToggle">
                <button class="active" data-filter="pbs">PBs</button>
                <button data-filter="volume">Volume</button>
                <button data-filter="fitness">Fitness</button>
                <button data-filter="all">All</button>
            </div>
        </div>
        <div class="chart-desc">{tr:,} runs \u00b7 {td:,.0f} km \u00b7 {trc} races \u00b7 {ya} years</div>
        <div id="nextMilestones" style="display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 14px 0;"></div>
        <div id="milestoneTimeline" style="max-height:420px;overflow-y:auto;padding-right:4px;"></div>
    </div>
    <script>
    (function() {{
        const allM={mj}; const nextM={nj};
        const nc=document.getElementById('nextMilestones');
        if(nextM.length>0){{nc.innerHTML=nextM.map(m=>{{const p=Math.min(m.pct_complete,100).toFixed(1);const e=m.est_days?` \u00b7 ~${{m.est_days}}d`:'';return `<div style="flex:1;min-width:140px;background:var(--card-bg,#1a1d27);border:1px solid var(--border,#2e3340);border-radius:8px;padding:8px 10px;"><div style="font-size:11px;color:var(--text-dim,#8b8fa3);margin-bottom:4px;">${{m.icon}} ${{m.title}}</div><div style="background:var(--bg,#0f1117);border-radius:4px;height:6px;overflow:hidden;"><div style="width:${{p}}%;height:100%;background:var(--accent,#818cf8);border-radius:4px;"></div></div><div style="font-size:10px;color:var(--text-dim,#8b8fa3);margin-top:3px;">${{m.remaining}}${{e}}</div></div>`;}}).join('')}}else{{nc.style.display='none'}}
        const cc={{distance:'#818cf8',runs:'#60a5fa',pb:'#f59e0b',fitness:'#4ade80',age_grade:'#f472b6',yearly:'#38bdf8',consistency:'#fb923c',races:'#a78bfa'}};
        const cl={{distance:'Distance',runs:'Runs',pb:'PB',fitness:'Fitness',age_grade:'Age Grade',yearly:'Yearly',consistency:'Streak',races:'Races'}};
        function fd(s){{const d=new Date(s);return d.getDate()+' '+['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][d.getMonth()]+' '+d.getFullYear()}}
        function render(f){{let fl;if(f==='pbs')fl=allM.filter(m=>m.category==='pb').sort((a,b)=>new Date(b.date)-new Date(a.date));else if(f==='volume')fl=allM.filter(m=>['distance','runs','yearly','races'].includes(m.category)).sort((a,b)=>new Date(b.date)-new Date(a.date));else if(f==='fitness')fl=allM.filter(m=>['fitness','age_grade','consistency'].includes(m.category)).sort((a,b)=>new Date(b.date)-new Date(a.date));else fl=[...allM].sort((a,b)=>new Date(b.date)-new Date(a.date));const c=document.getElementById('milestoneTimeline');if(!fl.length){{c.innerHTML='<div style="color:var(--text-dim,#8b8fa3);padding:20px;text-align:center;">No milestones yet</div>';return}}c.innerHTML=fl.map(m=>{{const co=cc[m.category]||'#818cf8';const st=m.importance>=3?'\u2605\u2605\u2605':(m.importance>=2?'\u2605\u2605':'\u2605');const sc=m.importance>=3?'#f59e0b':(m.importance>=2?'#94a3b8':'#475569');const pb=m.is_current_pb?' style="border-left:3px solid #f59e0b;"':'';return `<div class="milestone-item"${{pb}}><div style="display:flex;align-items:center;gap:8px;"><span style="font-size:20px;">${{m.icon}}</span><div style="flex:1;min-width:0;"><div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;"><span style="font-weight:600;color:var(--text,#e4e7ef);font-size:13px;">${{m.title}}</span><span style="font-size:9px;padding:1px 5px;border-radius:3px;background:${{co}}22;color:${{co}};font-weight:500;">${{cl[m.category]||m.category}}</span><span style="font-size:10px;color:${{sc}};">${{st}}</span></div><div style="font-size:11px;color:var(--text-dim,#8b8fa3);margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${{m.description}}</div></div><div style="font-size:11px;color:var(--text-dim,#8b8fa3);white-space:nowrap;">${{fd(m.date)}}</div></div></div>`}}).join('')}}
        document.querySelectorAll('#milestoneToggle button').forEach(b=>{{b.addEventListener('click',function(){{document.querySelectorAll('#milestoneToggle button').forEach(x=>x.classList.remove('active'));this.classList.add('active');render(this.dataset.filter)}})}}); render('pbs');
    }})();
    </script>
    '''


# --- Add milestone-item CSS to existing styles ---
_MILESTONE_CSS = '''
    .milestone-item { padding: 8px 10px; border-bottom: 1px solid var(--border, #2e3340); transition: background 0.15s; }
    .milestone-item:hover { background: rgba(129, 140, 248, 0.04); }
    .milestone-item:last-child { border-bottom: none; }
    #milestoneTimeline::-webkit-scrollbar { width: 4px; }
    #milestoneTimeline::-webkit-scrollbar-track { background: transparent; }
    #milestoneTimeline::-webkit-scrollbar-thumb { background: var(--border, #2e3340); border-radius: 2px; }
'''


def _generate_race_history_html(race_history_data):
    """Generate the Race History comparison section with two selectable card slots."""
    if not race_history_data:
        return ''
    
    import json as _json
    
    # Build distance options from actual race data
    dist_cats = sorted(set(r['dist_cat'] for r in race_history_data),
                       key=lambda x: {'1500m':0,'Mile':1,'3K':2,'5K':3,'10K':4,'10M':5,'HM':6,'30K':7,'Marathon':8,'Other':9}.get(x, 99))
    
    _rh_json = _json.dumps(race_history_data)
    
    return f'''
    <div class="chart-container" id="race-history-section">
        <div class="chart-title-row">
            <span class="chart-title">📊 Race History</span>
            <span class="badge" style="font-size:0.68rem">compare any two races</span>
        </div>
        <div id="rh-cards" style="display:grid;grid-template-columns:1fr;gap:12px;margin-top:8px;max-width:100%;overflow:hidden;">
            <div class="rh-slot" id="rh-slot-0">
                <div class="rh-picker">
                    <select id="rh-dist-0" class="rh-select rh-dist-select" onchange="rhFilterRaces(0)">
                        <option value="all">All distances</option>
                        {"".join(f'<option value="{d}">{d}</option>' for d in dist_cats)}
                    </select>
                    <select id="rh-race-0" class="rh-select rh-race-select" onchange="rhSelectRace(0)">
                        <option value="">Select a race...</option>
                    </select>
                </div>
                <div class="rh-card rh-empty" id="rh-card-0">
                    <div style="text-align:center;color:var(--text-dim);padding:24px 0;font-size:0.8rem">
                        Select a race above
                    </div>
                </div>
            </div>
            <div class="rh-slot" id="rh-slot-1">
                <div class="rh-picker">
                    <select id="rh-dist-1" class="rh-select rh-dist-select" onchange="rhFilterRaces(1)">
                        <option value="all">All distances</option>
                        {"".join(f'<option value="{d}">{d}</option>' for d in dist_cats)}
                    </select>
                    <select id="rh-race-1" class="rh-select rh-race-select" onchange="rhSelectRace(1)">
                        <option value="">Select a race...</option>
                    </select>
                </div>
                <div class="rh-card rh-empty" id="rh-card-1">
                    <div style="text-align:center;color:var(--text-dim);padding:24px 0;font-size:0.8rem">
                        Select a race above
                    </div>
                </div>
            </div>
        </div>
        <div id="rh-delta" style="display:none"></div>
    </div>
    <script>
    const RH_RACES={_rh_json};
    const RH_SEL=[null,null];
    function rhFilterRaces(slot){{
        const distSel=document.getElementById('rh-dist-'+slot);
        const raceSel=document.getElementById('rh-race-'+slot);
        const dist=distSel.value;
        const filtered=dist==='all'?RH_RACES:RH_RACES.filter(r=>r.dist_cat===dist);
        raceSel.innerHTML='<option value="">Select a race...</option>';
        filtered.forEach((r,i)=>{{
            const idx=RH_RACES.indexOf(r);
            const opt=document.createElement('option');
            opt.value=idx;
            const dLabel=r.dist_cat==='Other'?(r.dist_km?r.dist_km.toFixed(1)+'km':'Other'):r.dist_cat;
            opt.textContent=r.date_display+' · '+r.name+(r.name?' · ':'')+dLabel+' · '+r.time;
            raceSel.appendChild(opt);
        }});
        // Clear card
        RH_SEL[slot]=null;
        document.getElementById('rh-card-'+slot).innerHTML='<div style="text-align:center;color:var(--text-dim);padding:24px 0;font-size:0.8rem">Select a race above</div>';
        document.getElementById('rh-card-'+slot).classList.add('rh-empty');
        rhUpdateDelta();
    }}
    function rhFmtTSB(v){{if(v===null||v===undefined)return'-';return(v>=0?'+':'')+v.toFixed(1);}}
    function rhFmtVal(v,unit){{if(v===null||v===undefined)return'-';return v+((unit)?'<span style="font-size:0.7rem;color:var(--text-dim)">'+unit+'</span>':'');}}
    function rhSelectRace(slot){{
        const raceSel=document.getElementById('rh-race-'+slot);
        const idx=parseInt(raceSel.value);
        if(isNaN(idx)){{RH_SEL[slot]=null;rhUpdateDelta();return;}}
        const r=RH_RACES[idx];
        RH_SEL[slot]=r;
        const card=document.getElementById('rh-card-'+slot);
        card.classList.remove('rh-empty');
        const surfBadge=r.surface!=='road'?'<span class="rh-surf">'+r.surface.toUpperCase()+'</span>':'';
        const tempStr=r.temp!==null?r.temp+'°C':'-';
        const solarBoost=(r.solar&&r.solar>0)?r.solar/200.0:0;
        const tempEff=r.temp!==null?Math.round(r.temp+solarBoost):null;
        const sunIcon=r.solar>400?'☀️':r.solar>200?'🌤️':'';
        const tempDisplay=tempEff!==null?(sunIcon+tempEff+'°C'):'-';
        const tempTip=r.temp!==null?(solarBoost>0.5?r.temp+'°C shade + '+solarBoost.toFixed(0)+'°C solar = '+tempEff+'°C effective':r.temp+'°C (low/no solar)'):'No temperature data';
        const gainKm=(r.elev_gain&&r.dist_km>0)?r.elev_gain/r.dist_km:0;
        const und=r.undulation||0;
        const terrStr=gainKm<=5?'flat':gainKm>12?(und>6?'hilly & rolling':'hilly'):(und>6?'rolling':'undulating');
        const terrDetail=r.elev_gain?r.elev_gain+'m gain ('+gainKm.toFixed(0)+'m/km)'+(und>0?', undulation '+und:''):'';
        const condTip=tempTip+(terrDetail?' · '+terrDetail:'');
        const tsbCol=r.tsb!==null?(r.tsb>=0?'#4ade80':'#fbbf24'):'var(--text-dim)';
        const rflVal=(typeof currentMode!=='undefined'&&currentMode==='gap'&&r.rfl_gap!==null)?r.rfl_gap:r.rfl;
        // Format prediction
        function fmtSecs(s){{if(!s||s<=0)return'-';const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sc=Math.floor(s%60);return h>0?h+':'+String(m).padStart(2,'0')+':'+String(sc).padStart(2,'0'):m+':'+String(sc).padStart(2,'0');}}
        const predStr=r.pred_s?fmtSecs(r.pred_s):null;
        const predSub=predStr?'pred '+predStr:'';
        const predDiff=r.pred_s&&r.time_s?r.time_s-r.pred_s:null;
        const predTip=predDiff!==null?fmtSecs(Math.abs(predDiff))+(predDiff>0?' slower':' faster')+' than predicted':'';
        card.innerHTML=`
            <div class="rh-header">
                <div class="rh-name">${{r.name}}${{surfBadge}}</div>
                <div class="rh-date">${{r.date_display}} · ${{r.dist_km}}km (${{r.dist_cat}})</div>
            </div>
            <div class="rh-row">
                <div class="ws-tip rh-metric"><div class="rh-val" style="color:var(--accent)">${{r.time}}</div><div class="rh-label">Time</div><div class="rh-sub">${{predSub}}</div><div class="tip">Elapsed finish time${{predTip?' · '+predTip:''}}</div></div>
                <div class="ws-tip rh-metric"><div class="rh-val">${{r.pace}}</div><div class="rh-label">/km</div><div class="tip">Average pace per km</div></div>
                <div class="ws-tip rh-metric"><div class="rh-val">${{rhFmtVal(r.hr)}}</div><div class="rh-label">Avg HR</div><div class="tip">Average heart rate during race</div></div>
                <div class="ws-tip rh-metric"><div class="rh-val">${{rhFmtVal(r.nag,'%')}}</div><div class="rh-label">nAG</div><div class="rh-sub">${{r.raw_ag?'raw '+r.raw_ag+'%':''}}</div><div class="tip">Normalised age grade (adjusted for temp, terrain, surface)</div></div>
            </div>
            <div class="rh-sep"></div>
            <div class="rh-row">
                <div class="ws-tip rh-metric"><div class="rh-val">${{rhFmtVal(r.ctl)}}</div><div class="rh-label">CTL</div><div class="tip">Chronic training load (42-day) morning of race</div></div>
                <div class="ws-tip rh-metric"><div class="rh-val">${{rhFmtVal(r.atl)}}</div><div class="rh-label">ATL</div><div class="tip">Acute training load (7-day) morning of race</div></div>
                <div class="ws-tip rh-metric"><div class="rh-val" style="color:${{tsbCol}}">${{rhFmtTSB(r.tsb)}}</div><div class="rh-label">TSB</div><div class="tip">Training stress balance (CTL − ATL) morning of race</div></div>
                <div class="ws-tip rh-metric"><div class="rh-val">${{rhFmtVal(r.tss)}}</div><div class="rh-label">TSS</div><div class="tip">Training stress score for this race</div></div>
            </div>
            <div class="rh-sep"></div>
            <div class="rh-row">
                <div class="ws-tip rh-metric"><div class="rh-val" style="color:#4ade80">${{rflVal!==null?rflVal+'%':'-'}}</div><div class="rh-label">RFL</div><div class="tip">Relative fitness level coming into the race</div></div>
                <div class="ws-tip rh-metric"><div class="rh-val">${{r.e14}}<span class="rh-unit">min</span></div><div class="rh-label">Effort 14d</div><div class="tip">Minutes at ${{r.dist_cat}} race effort in 14 days before race</div></div>
                <div class="ws-tip rh-metric"><div class="rh-val">${{r.e42}}<span class="rh-unit">min</span></div><div class="rh-label">Effort 42d</div><div class="tip">Minutes at ${{r.dist_cat}} race effort in 42 days before race</div></div>
                <div class="ws-tip rh-metric"><div class="rh-val">${{tempDisplay}}</div><div class="rh-label">${{terrStr}}</div><div class="tip">${{condTip}}</div></div>
            </div>
            ${{r.dist_km >= 21.0 ? `<div class="rh-sep"></div>
            <div class="rh-row">
                <div class="ws-tip rh-metric"><div class="rh-val" style="color:#a78bfa">${{r.lr_total_14}}<span class="rh-unit">min</span></div><div class="rh-label">≥${{r.lr_threshold}}m 14d</div><div class="tip">Total running time beyond ${{r.lr_threshold}} min in last 14 days</div></div>
                <div class="ws-tip rh-metric"><div class="rh-val" style="color:#a78bfa">${{r.lr_total_42}}<span class="rh-unit">min</span></div><div class="rh-label">≥${{r.lr_threshold}}m 42d</div><div class="tip">Total running time beyond ${{r.lr_threshold}} min in last 42 days</div></div>
                <div class="ws-tip rh-metric"><div class="rh-val" style="color:#fb923c">${{r.lr_z3_14}}<span class="rh-unit">min</span></div><div class="rh-label">Z3+ tail 14d</div><div class="tip">Time at Z3+ HR beyond ${{r.lr_threshold}} min mark in last 14 days</div></div>
                <div class="ws-tip rh-metric"><div class="rh-val" style="color:#fb923c">${{r.lr_z3_42}}<span class="rh-unit">min</span></div><div class="rh-label">Z3+ tail 42d</div><div class="tip">Time at Z3+ HR beyond ${{r.lr_threshold}} min mark in last 42 days</div></div>
            </div>` : ''}}`;
        rhUpdateDelta();
    }}
    function rhUpdateDelta(){{
        const el=document.getElementById('rh-delta');
        const a=RH_SEL[0],b=RH_SEL[1];
        if(!a||!b){{el.style.display='none';return;}}
        el.style.display='block';
        // Compare
        const rows=[];
        function d(label,va,vb,unit,invert){{
            if(va===null||va===undefined||vb===null||vb===undefined)return;
            const diff=vb-va;
            const sign=diff>0?'+':'';
            const col=invert?(diff<0?'#4ade80':(diff>0?'#ef4444':'var(--text-dim)')):(diff>0?'#4ade80':(diff<0?'#ef4444':'var(--text-dim)'));
            rows.push(`<div class="rh-delta-row"><span class="rh-delta-label">${{label}}</span><span style="color:${{col}};font-family:'JetBrains Mono';font-weight:600">${{sign}}${{typeof diff==='number'?diff.toFixed(1):diff}}${{unit||''}}</span></div>`);
        }}
        // Time diff
        if(a.time_s&&b.time_s&&a.dist_cat===b.dist_cat){{
            const ds=b.time_s-a.time_s;
            const sign=ds>0?'+':'-';
            const abs=Math.abs(ds);
            const m=Math.floor(abs/60),s=Math.round(abs%60);
            const col=ds<0?'#4ade80':(ds>0?'#ef4444':'var(--text-dim)');
            rows.push(`<div class="rh-delta-row"><span class="rh-delta-label">Time</span><span style="color:${{col}};font-family:'JetBrains Mono';font-weight:600">${{sign}}${{m}}:${{String(s).padStart(2,'0')}}</span></div>`);
        }}
        d('nAG',a.nag,b.nag,'%',false);
        d('CTL',a.ctl,b.ctl,'',false);
        d('TSB',a.tsb,b.tsb,'',false);
        d('14d effort',a.e14,b.e14,' min',false);
        d('42d effort',a.e42,b.e42,' min',false);
        if(a.dist_km>=21.0||b.dist_km>=21.0){{d('Long 14d',a.lr_total_14,b.lr_total_14,' min',false);
        d('Long 42d',a.lr_total_42,b.lr_total_42,' min',false);}}
        d('Avg HR',a.hr,b.hr,' bpm',true);
        if(rows.length===0){{el.style.display='none';return;}}
        el.innerHTML='<div class="rh-delta-title">Δ '+a.date_display+' → '+b.date_display+'</div>'+rows.join('');
    }}
    // Init: populate race selects
    [0,1].forEach(slot=>rhFilterRaces(slot));
    // Position fixed tooltips inside rh-cards on hover
    document.getElementById('rh-cards').addEventListener('mouseover', function(e) {{
        const wsTip = e.target.closest('.ws-tip');
        if (!wsTip) return;
        const tip = wsTip.querySelector('.tip');
        if (!tip) return;
        const rect = wsTip.getBoundingClientRect();
        tip.style.left = (rect.left + rect.width / 2) + 'px';
        tip.style.top = (rect.top - 4) + 'px';
        tip.style.transform = 'translate(-50%, -100%)';
    }});
    </script>
    '''


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


def _build_upcoming_sessions_html(sessions):
    """Build HTML for the Upcoming Sessions card. Returns empty string if no sessions."""
    if not sessions:
        return ''

    # Actual TSS already completed earlier in the current week
    completed_tss = sessions[0].pop('_completed_tss_this_week', 0) if sessions else 0

    # Group by week (ISO week) to show weekly TSS totals
    from collections import OrderedDict
    weeks = OrderedDict()
    first_week = True
    for s in sessions:
        # Get ISO week from date
        dt = datetime.strptime(s['date_iso'], '%Y-%m-%d')
        week_key = dt.strftime('%Y-W%V')
        if week_key not in weeks:
            # First week includes already-completed TSS
            initial_tss = completed_tss if first_week else 0
            weeks[week_key] = {'sessions': [], 'tss': initial_tss, 'completed_tss': initial_tss}
            first_week = False
        weeks[week_key]['sessions'].append(s)
        weeks[week_key]['tss'] += s['tss']

    # Detect hard sessions (F/L prefix = fartlek/long run with quality)
    def _is_hard_session(desc):
        d = desc.strip()
        return bool(d) and d[0] in ('F', 'L') and len(d) > 1 and d[1] in '0123456789:'

    # Smart truncation: preserve session type prefix (e.g. "F9:" or "L4:")
    def _truncate_desc(desc, max_len=50):
        if len(desc) <= max_len:
            return desc
        # Find first colon — keep the prefix label intact
        colon = desc.find(':')
        if 0 < colon < 5:
            return desc[:max_len - 1] + '\u2026'
        return desc[:max_len - 1] + '\u2026'

    def _week_total_row(week_data):
        total = week_data['tss']
        done = week_data.get('completed_tss', 0)
        if done > 0:
            label = f'Week total ({done} done + {total - done} planned)'
        else:
            label = 'Week total'
        return f'<tr class="upcoming-week-total"><td></td><td>{label}</td><td style="text-align:right">{total}</td><td colspan="3"></td></tr>'

    # Build rows
    rows = []
    current_week = None
    for s in sessions:
        dt = datetime.strptime(s['date_iso'], '%Y-%m-%d')
        week_key = dt.strftime('%Y-W%V')

        # Week separator
        if week_key != current_week:
            if current_week is not None:
                rows.append(_week_total_row(weeks[current_week]))
            current_week = week_key

        # Session row
        is_rest = s['tss'] == 0
        is_race = s.get('is_race', False)
        is_hard = _is_hard_session(s['description'])
        classes = []
        if is_race:
            classes.append('upcoming-race')
        elif is_rest:
            classes.append('upcoming-rest')
        elif is_hard:
            classes.append('upcoming-hard')
        row_class = ' '.join(classes)
        tss_str = str(s['tss']) if s['tss'] > 0 else ''
        desc = _truncate_desc(s['description'])

        # CTL/ATL/TSB values (projected after planned session)
        ctl_str = f'{s["ctl"]:.0f}' if s.get('ctl') is not None else ''
        atl_str = f'{s["atl"]:.0f}' if s.get('atl') is not None else ''
        tsb_val = s.get('tsb')
        if tsb_val is not None:
            tsb_color = '#4ade80' if tsb_val >= 0 else '#f87171'
            tsb_str = f'<span style="color:{tsb_color}">{tsb_val:+.0f}</span>'
        else:
            tsb_str = ''

        rows.append(
            f'<tr class="{row_class}">'
            f'<td style="white-space:nowrap;opacity:0.5;font-size:0.85em">{s["date_str"]}</td>'
            f'<td class="upcoming-desc">{desc}</td>'
            f'<td style="text-align:right;opacity:0.5">{tss_str}</td>'
            f'<td style="text-align:right;opacity:0.4;font-size:0.85em">{ctl_str}</td>'
            f'<td style="text-align:right;opacity:0.4;font-size:0.85em">{atl_str}</td>'
            f'<td style="text-align:right;font-size:0.85em">{tsb_str}</td>'
            f'</tr>'
        )

    # Final week total
    if current_week:
        rows.append(_week_total_row(weeks[current_week]))

    # Race countdown
    race_sessions = [s for s in sessions if s.get('is_race')]
    race_line = ''
    if race_sessions:
        r = race_sessions[0]
        days_to = (datetime.strptime(r['date_iso'], '%Y-%m-%d') - datetime.now()).days + 1
        day_word = 'day' if days_to == 1 else 'days'
        race_line = f'<div style="margin-top:10px;font-size:0.85em;opacity:0.7">Race: {r["description"]} — {r["date_str"]} ({days_to} {day_word})</div>'

    # Weekly TSS taper summary
    week_tss_parts = []
    for wk, wdata in weeks.items():
        week_tss_parts.append(str(wdata['tss']))
    taper_line = ''
    if len(week_tss_parts) > 1:
        taper_line = f'<div style="font-size:0.85em;opacity:0.7">Weekly TSS: {" → ".join(week_tss_parts)}</div>'

    return f'''<div class="chart-container">
        <div class="chart-header">
            <div class="chart-title">📋 Upcoming Sessions</div>
        </div>
        <div class="chart-desc">Planned training from uploaded schedule. TSS estimated from session type and duration.</div>
        <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse;font-size:0.9em">
        <thead><tr style="border-bottom:1px solid rgba(255,255,255,0.1)">
            <th style="text-align:left;padding:4px 8px;opacity:0.6">Date</th>
            <th style="text-align:left;padding:4px 8px;opacity:0.6">Session</th>
            <th style="text-align:right;padding:4px 8px;opacity:0.6">TSS</th>
            <th style="text-align:right;padding:4px 8px;opacity:0.4;font-size:0.85em">CTL</th>
            <th style="text-align:right;padding:4px 8px;opacity:0.4;font-size:0.85em">ATL</th>
            <th style="text-align:right;padding:4px 8px;opacity:0.4;font-size:0.85em">TSB</th>
        </tr></thead>
        <tbody>{"".join(rows)}</tbody>
        </table>
        </div>
        {race_line}
        {taper_line}
    </div>'''


def generate_html(stats, rf_data, volume_data, ctl_atl_data, ctl_atl_lookup, rfl_trend_dates, rfl_trend_values, rfl_trendline, rfl_projection, rfl_ci_upper, rfl_ci_lower, alltime_rfl_dates, alltime_rfl_values, recent_runs, top_races, alert_data=None, weight_data=None, prediction_data=None, ag_data=None, zone_data=None, rfl_trend_gap=None, rfl_trend_sim=None, milestone_data=None, race_history_data=None, upcoming_sessions=None):
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
        _init_ag = stats['race_predictions'].get('_ag_gap') or stats.get('age_grade')
    elif _init_mode == 'sim':
        _init_rfl = stats.get('latest_rfl_sim') or stats.get('latest_rfl')
        _init_delta = stats.get('rfl_14d_delta_sim') or stats.get('rfl_14d_delta')
        _init_rfl_label = 'RFL (Sim)'
        _init_preds = stats['race_predictions'].get('_mode_sim', dict())
        _init_ag = stats['race_predictions'].get('_ag_sim')
    else:
        _init_rfl = stats.get('latest_rfl')
        _init_delta = stats.get('rfl_14d_delta')
        _init_rfl_label = 'RFL Trend'
        _init_preds = stats['race_predictions']
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
        if len(data) == 7:
            return data  # dates, ctl, atl, ctl_proj, atl_proj, tsb_proj, planned_markers
        elif len(data) == 5:
            return data[0], data[1], data[2], data[3], data[4], [], []  # legacy 5-element
        elif len(data) == 3:
            return data[0], data[1], data[2], [], [], [], []  # dates, ctl, atl, empty proj
        else:
            return [], [], [], [], [], [], []
    
    ctl_atl_dates_90, ctl_values_90, atl_values_90, ctl_proj_90, atl_proj_90, tsb_proj_90, planned_90 = unpack_ctl_atl(ctl_atl_90)
    ctl_atl_dates_180, ctl_values_180, atl_values_180, ctl_proj_180, atl_proj_180, tsb_proj_180, planned_180 = unpack_ctl_atl(ctl_atl_180)
    ctl_atl_dates_365, ctl_values_365, atl_values_365, ctl_proj_365, atl_proj_365, tsb_proj_365, planned_365 = unpack_ctl_atl(ctl_atl_365)
    ctl_atl_dates_730, ctl_values_730, atl_values_730, ctl_proj_730, atl_proj_730, tsb_proj_730, planned_730 = unpack_ctl_atl(ctl_atl_730)
    ctl_atl_dates_1095, ctl_values_1095, atl_values_1095, ctl_proj_1095, atl_proj_1095, tsb_proj_1095, planned_1095 = unpack_ctl_atl(ctl_atl_1095)
    ctl_atl_dates_1825, ctl_values_1825, atl_values_1825, ctl_proj_1825, atl_proj_1825, tsb_proj_1825, planned_1825 = unpack_ctl_atl(ctl_atl_1825)
    
    # Build planned races JSON for JS injection (outside f-string to avoid dict/brace conflicts)
    _planned_races_json = json.dumps([
        {'name': r['name'], 'date': r['date'], 'priority': r.get('priority', 'B'),
         'distance_key': r['distance_key'], 'distance_km': r.get('distance_km', 5.0),
         'surface': r.get('surface', 'road')}
        for r in PLANNED_RACES_DASH
    ])
    
    # Race week plan: pre-race TSB lookup and chart plan data
    _pre_race_tsb_json = json.dumps(zone_data.get('pre_race_tsb', {}) if zone_data else {})
    
    _rwp_plans_list = []
    if zone_data:
        _DOW_JS3 = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        for _ri3, _race3 in enumerate(PLANNED_RACES_DASH):
            _rpri3 = _race3.get('priority', 'B')
            if _rpri3 == 'C':
                continue
            try:
                _rdt3 = datetime.strptime(_race3['date'], '%Y-%m-%d').date()
                from datetime import date as _dtd3
                _dto3 = (_rdt3 - _dtd3.today()).days
            except (ValueError, KeyError):
                continue
            if _dto3 < 1 or _dto3 > 7:
                continue
            _dkm3 = _race3.get('distance_km', 5.0)
            _dcat3 = 'Mara' if _dkm3 >= 35 else ('HM' if _dkm3 > 12 else ('10K' if _dkm3 > 5.5 else '5K'))
            _c03 = zone_data.get('current_ctl', 0)
            _a03 = zone_data.get('current_atl', 0)
            _tsb03 = _c03 - _a03
            _recent3 = {r['date']: r for r in zone_data.get('recent_tss', [])}
            
            # Use shared solver
            _plan3, _results3 = _solve_taper(
                _c03, _a03, _dto3, _c03, _dcat3, _rpri3,
                _dtd3.today(), _recent3, _race3['name'], dist_km=_dkm3
            )
            
            # Build chart points from results (includes lookback + forward)
            _pts3 = []
            for _r3 in _results3:
                _is_now = (_r3['date'] == _dtd3.today())
                _pts3.append({
                    'tsb': round(_r3['tsb'], 1),
                    'l': 'Today' if _is_now else _DOW_JS3[_r3['date'].isoweekday() % 7],
                    't': _r3['tag'][0] if _r3['tag'] != 'race' else 'r',
                    'r': _r3['is_race'],
                    'now': _is_now,
                })
            
            _race_ctl3 = _results3[-1]['ctl']
            _pcts3 = _rwp_interp_targets(_dkm3).get(_rpri3, (2, 18))
            _rwp_plans_list.append({
                'canvas': f'tsbChart_{_ri3}',
                'pts': _pts3,
                'tsbLo': round(_race_ctl3 * _pcts3[0] / 100, 1),
                'tsbHi': round(_race_ctl3 * _pcts3[1] / 100, 1),
            })
    _rwp_plans_json = json.dumps(_rwp_plans_list)
    
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
        .stat-card.ws-tip {{ cursor: help; }}
        .stat-card.ws-tip:hover {{ border-color: var(--accent-dim, rgba(129,140,248,0.3)); }}
        
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
        .ws-tip {{ position: relative; cursor: default; }}
        .ws-tip .tip {{ display: none; position: fixed; background: #1e2130; border: 1px solid var(--border); padding: 5px 9px; border-radius: 5px; font-size: 0.67rem; z-index: 9999; color: var(--text); pointer-events: none; max-width: 280px; white-space: normal; text-align: center; line-height: 1.4; }}
        .ws-tip:hover .tip {{ display: block; }}
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
        
        /* Race Week Plan */
        .rwp {{ margin-top: 12px; border-top: 1px solid var(--border); padding-top: 12px; }}
        .rwp-label {{ font-size: 0.72rem; font-weight: 600; color: var(--accent); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }}
        .dh {{ display: grid; grid-template-columns: 72px 1fr 42px 42px 42px 48px; gap: 4px; font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600; padding-bottom: 5px; border-bottom: 1px solid var(--border); margin-bottom: 2px; }}
        .dh span:nth-child(n+3) {{ text-align: right; }}
        .dr {{ display: grid; grid-template-columns: 72px 1fr 42px 42px 42px 48px; align-items: center; padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.025); font-size: 0.78rem; gap: 4px; }}
        .dr:last-child {{ border-bottom: none; }}
        .dr-race {{ background: rgba(74,222,128,0.06); border-radius: 6px; padding-left: 6px; margin-left: -6px; padding-right: 6px; margin-right: -6px; }}
        .dd {{ font-family: 'JetBrains Mono'; font-size: 0.72rem; color: var(--text-dim); }}
        .dd b {{ font-weight: 600; color: var(--text); font-size: 0.75rem; }}
        .ds {{ font-weight: 500; font-size: 0.76rem; }}
        .dt {{ display: inline-block; font-size: 0.56rem; padding: 1px 5px; border-radius: 3px; font-weight: 600; margin-left: 4px; vertical-align: middle; }}
        .dt-d {{ background: rgba(74,222,128,0.15); color: #4ade80; }}
        .dt-p {{ background: rgba(129,140,248,0.12); color: var(--accent); }}
        .dt-t {{ background: rgba(251,191,36,0.15); color: #fbbf24; }}
        .dt-r {{ background: rgba(251,191,36,0.18); color: #fbbf24; }}
        .dv {{ text-align: right; font-family: 'JetBrains Mono'; font-size: 0.73rem; }}
        .dv-t {{ color: #fbbf24; }}
        .dv-c {{ color: #3b82f6; }}
        .dv-a {{ color: #f97316; }}
        .dv-s {{ font-weight: 700; }}
        .tsb-v {{ display: inline-block; font-size: 0.72rem; padding: 3px 8px; border-radius: 5px; font-weight: 600; margin-top: 10px; }}
        .tsb-v-ok {{ background: rgba(74,222,128,0.12); color: #4ade80; }}
        .tsb-v-lo {{ background: rgba(251,191,36,0.12); color: #fbbf24; }}
        .tsb-v-hi {{ background: rgba(59,130,246,0.12); color: #3b82f6; }}
        .tsb-cw {{ margin-top: 10px; height: 90px; position: relative; }}
        .tsb-cw canvas {{ width: 100%; height: 100%; display: block; }}
        @media (max-width: 600px) {{
            .dr, .dh {{ grid-template-columns: 54px 1fr 38px 38px 38px 44px; }}
            .dv {{ font-size: 0.68rem; }}
        }}
        
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
        
        /* Milestones */
        .milestone-item {{ padding: 8px 10px; border-bottom: 1px solid var(--border); transition: background 0.15s; }}
        .milestone-item:hover {{ background: rgba(129, 140, 248, 0.04); }}
        .milestone-item:last-child {{ border-bottom: none; }}
        #milestoneTimeline::-webkit-scrollbar {{ width: 4px; }}
        #milestoneTimeline::-webkit-scrollbar-track {{ background: transparent; }}
        #milestoneTimeline::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}
        
        /* Race History */
        .rh-slot {{ min-width: 0; }}
        .rh-picker {{ display: flex; gap: 6px; margin-bottom: 8px; }}
        .rh-select {{ background: var(--surface2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 5px 8px; font-size: 0.74rem; font-family: 'DM Sans'; flex: 1; cursor: pointer; min-width: 0; overflow: hidden; text-overflow: ellipsis; }}
        .rh-select:focus {{ outline: none; border-color: var(--accent); }}
        .rh-dist-select {{ flex: 0 0 auto; min-width: 110px; }}
        .rh-card {{ background: var(--surface2); border-radius: 8px; padding: 14px 16px; transition: all 0.2s; overflow: hidden; }}
        .rh-card.rh-empty {{ border: 1px dashed var(--border); background: transparent; min-height: 50px; display: flex; align-items: center; justify-content: center; padding: 12px; }}
        .rh-card .ws-tip .tip {{ position: fixed; bottom: auto; left: auto; transform: none; z-index: 9999; }}
        .rh-header {{ margin-bottom: 8px; }}
        .rh-name {{ font-weight: 600; font-size: 0.85rem; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .rh-surf {{ font-size: 0.58rem; padding: 1px 5px; border-radius: 3px; background: #f59e0b22; color: #f59e0b; font-weight: 600; margin-left: 6px; vertical-align: middle; letter-spacing: 0.03em; }}
        .rh-date {{ font-size: 0.7rem; color: var(--text-dim); font-family: 'JetBrains Mono'; margin-top: 2px; }}
        .rh-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; padding: 4px 0; }}
        .rh-sep {{ height: 1px; background: var(--border); opacity: 0.5; margin: 4px 0; }}
        .rh-metric {{ text-align: center; padding: 2px 0; position: relative; }}
        .rh-val {{ font-size: 1.1rem; font-weight: 700; font-family: 'JetBrains Mono'; }}
        .rh-unit {{ font-size: 0.75rem; color: var(--text-dim); margin-left: 1px; }}
        .rh-label {{ font-size: 0.68rem; color: var(--text-dim); margin-top: 1px; }}
        .rh-sub {{ font-size: 0.62rem; color: var(--text-dim); }}
        #rh-delta {{ background: var(--surface2); border-radius: 8px; padding: 10px 14px; overflow: hidden; margin-top: 12px; }}
        .rh-delta-title {{ font-size: 0.72rem; font-weight: 600; color: var(--accent); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.04em; }}
        .rh-delta-row {{ display: flex; justify-content: space-between; padding: 3px 0; font-size: 0.78rem; }}
        .rh-delta-label {{ color: var(--text-dim); }}
        
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
        
        /* Upcoming sessions table */
        .upcoming-rest td {{ opacity: 0.3; font-style: italic; }}
        .upcoming-race td {{ color: #FFD700; font-weight: 600; }}
        .upcoming-hard td.upcoming-desc {{ color: #93c5fd; }}
        .upcoming-week-total td {{
            border-top: 1px solid rgba(255,255,255,0.15);
            padding-top: 6px !important;
            padding-bottom: 8px !important;
            font-weight: 600;
            font-size: 0.82em;
            letter-spacing: 0.02em;
        }}
        .upcoming-week-total + tr td {{ padding-top: 8px !important; }}
        .chart-container tbody tr td {{ padding: 3px 8px; }}

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
        <div class="stat-card ws-tip">
            <div class="stat-value" id="ctl-value">{stats['ctl'] if stats['ctl'] else '-'}</div>
            <div class="stat-label">CTL</div>
            <div class="stat-sub">fitness</div>
            <div class="tip">Chronic Training Load — your rolling 42-day fitness. Higher = fitter. Built from consistent training over weeks.</div>
        </div>
        <div class="stat-card ws-tip">
            <div class="stat-value" id="atl-value">{stats['atl'] if stats['atl'] else '-'}</div>
            <div class="stat-label">ATL</div>
            <div class="stat-sub">fatigue</div>
            <div class="tip">Acute Training Load — your rolling 7-day fatigue. Spikes after hard training, drops with rest.</div>
        </div>
        <div class="stat-card ws-tip">
            <div class="stat-value" id="tsb-value">{stats['tsb'] if stats['tsb'] else '-'}</div>
            <div class="stat-label">TSB</div>
            <div class="stat-sub">form</div>
            <div class="tip">Training Stress Balance (CTL − ATL). Positive = fresh, negative = fatigued. Race-ready form is typically 5–25% of your CTL depending on distance and priority.</div>
        </div>
        <div class="stat-card ws-tip">
            <div class="stat-value">{f"{stats['weight']}kg" if stats['weight'] else '-'}</div>
            <div class="stat-label">Weight</div>
            <div class="stat-sub">7d average</div>
            <div class="tip">Your 7-day smoothed weight from the most recent data available.</div>
        </div>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card ws-tip">
            <div class="stat-value" id="rfl-value">{_init_rfl}%</div>
            <div class="stat-label" id="rfl-label">{_init_rfl_label}</div>
            <div class="stat-sub">vs peak</div>
            <div class="tip">Relative Fitness Level — your current running fitness as a percentage of your all-time peak. 100% = best ever.</div>
        </div>
        <div class="stat-card ws-tip">
            <div class="stat-value" id="rfl-delta">{f"{'+' if _init_delta > 0 else ''}{_init_delta}%" if _init_delta is not None else '-'}</div>
            <div class="stat-label">RFL 14d</div>
            <div class="stat-sub">change</div>
            <div class="tip">Change in RFL over the last 14 days. Positive = fitness improving. Negative = fitness declining.</div>
        </div>
        <div class="stat-card ws-tip">
            <div class="stat-value" id="ag-value">{_init_ag if _init_ag else '-'}%</div>
            <div class="stat-label">Age Grade</div>
            <div class="stat-sub">5k estimate</div>
            <div class="tip">Age-graded performance — compares your estimated 5K to the world record for your age and sex. 60%+ is good club level, 70%+ is competitive.</div>
        </div>
        <div class="stat-card ws-tip">
            <div class="stat-value" id="ag-rfl-value">{f"{stats['ag_rfl']}%" if stats.get('ag_rfl') is not None else '-%'}</div>
            <div class="stat-label">AG RFL</div>
            <div class="stat-sub">vs peak</div>
            <div class="tip">Age-graded Relative Fitness Level — your current predicted age grade ({stats.get('current_pred_ag', '-')}%) divided by your best ever race age grade ({stats.get('best_race_ag', '-')}%). 100% = you are at your age-adjusted peak.</div>
        </div>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card ws-tip">
            <div class="stat-value" id="pred-5k">{format_race_time(_init_preds.get('5k', '-'))}</div>
            <div class="stat-label">5k</div>
            <div class="stat-sub">predicted</div>
            <div class="tip">Predicted 5K race time based on your current fitness trend and critical power.</div>
        </div>
        <div class="stat-card ws-tip">
            <div class="stat-value" id="pred-10k">{format_race_time(_init_preds.get('10k', '-'))}</div>
            <div class="stat-label">10k</div>
            <div class="stat-sub">predicted</div>
            <div class="tip">Predicted 10K race time based on your current fitness trend and critical power.</div>
        </div>
        <div class="stat-card ws-tip">
            <div class="stat-value" id="pred-hm">{format_race_time(_init_preds.get('Half Marathon', '-'))}</div>
            <div class="stat-label">Half</div>
            <div class="stat-sub">predicted</div>
            <div class="tip">Predicted half marathon time based on your current fitness trend and critical power.</div>
        </div>
        <div class="stat-card ws-tip">
            <div class="stat-value" id="pred-mara">{format_race_time(_init_preds.get('Marathon', '-'))}</div>
            <div class="stat-label">Marathon</div>
            <div class="stat-sub">predicted</div>
            <div class="tip">Predicted marathon time based on your current fitness trend and critical power.</div>
        </div>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card ws-tip">
            <div class="stat-value">{stats['week_km']}</div>
            <div class="stat-label">Last 7 Days</div>
            <div class="stat-sub">km</div>
            <div class="tip">Total running distance in the last 7 days.</div>
        </div>
        <div class="stat-card ws-tip">
            <div class="stat-value">{stats['month_km']}</div>
            <div class="stat-label">Last 30 Days</div>
            <div class="stat-sub">km</div>
            <div class="tip">Total running distance in the last 30 days.</div>
        </div>
        <div class="stat-card ws-tip">
            <div class="stat-value">{int(stats['year_km']):,}</div>
            <div class="stat-label">Last 12 Months</div>
            <div class="stat-sub">km</div>
            <div class="tip">Total running distance in the last 12 months.</div>
        </div>
        <div class="stat-card ws-tip">
            <div class="stat-value">{stats['total_runs']:,}</div>
            <div class="stat-label">Total Runs</div>
            <div class="stat-sub">since {stats['first_year']}</div>
            <div class="tip">Total number of runs in your training history.</div>
        </div>
    </div>
    
    <!-- RF Trend Chart -->
    
    <!-- Recent Achievements (first thing after stats) -->
    {_generate_recent_achievements_html(milestone_data) if milestone_data else ''}
    
    <!-- Training Zones Section -->
    {_generate_zone_html(zone_data, stats, upcoming_sessions=upcoming_sessions)}
    
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
        <div class="chart-desc">CTL (blue) = fitness. ATL (red) = fatigue. Dashed lines show projection based on planned training sessions.</div>
        <div class="chart-wrapper">
            <canvas id="ctlAtlChart"></canvas>
        </div>
    </div>

    <!-- Upcoming Sessions -->
    {_build_upcoming_sessions_html(upcoming_sessions)}

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
    
    <!-- Race History Comparison -->
    {_generate_race_history_html(race_history_data) if race_history_data else ''}
    
    <!-- All-Time Milestones (career retrospective at bottom) -->
    {_generate_milestones_html(milestone_data) if milestone_data else ''}
    
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
                cp: {stats['race_predictions'].get('_cp_gap') or (round(PEAK_CP_WATTS_DASH * _gap_rfl_num / 100) if _gap_rfl_num else 0)},
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
            '90': {{ dates: {json.dumps(ctl_atl_dates_90)}, ctl: {json.dumps(ctl_values_90)}, atl: {json.dumps(atl_values_90)}, ctlProj: {json.dumps(ctl_proj_90)}, atlProj: {json.dumps(atl_proj_90)}, tsbProj: {json.dumps(tsb_proj_90)}, planned: {json.dumps(planned_90)} }},
            '180': {{ dates: {json.dumps(ctl_atl_dates_180)}, ctl: {json.dumps(ctl_values_180)}, atl: {json.dumps(atl_values_180)}, ctlProj: {json.dumps(ctl_proj_180)}, atlProj: {json.dumps(atl_proj_180)}, tsbProj: {json.dumps(tsb_proj_180)}, planned: {json.dumps(planned_180)} }},
            '365': {{ dates: {json.dumps(ctl_atl_dates_365)}, ctl: {json.dumps(ctl_values_365)}, atl: {json.dumps(atl_values_365)}, ctlProj: {json.dumps(ctl_proj_365)}, atlProj: {json.dumps(atl_proj_365)}, tsbProj: {json.dumps(tsb_proj_365)}, planned: {json.dumps(planned_365)} }},
            '730': {{ dates: {json.dumps(ctl_atl_dates_730)}, ctl: {json.dumps(ctl_values_730)}, atl: {json.dumps(atl_values_730)}, ctlProj: {json.dumps(ctl_proj_730)}, atlProj: {json.dumps(atl_proj_730)}, tsbProj: {json.dumps(tsb_proj_730)}, planned: {json.dumps(planned_730)} }},
            '1095': {{ dates: {json.dumps(ctl_atl_dates_1095)}, ctl: {json.dumps(ctl_values_1095)}, atl: {json.dumps(atl_values_1095)}, ctlProj: {json.dumps(ctl_proj_1095)}, atlProj: {json.dumps(atl_proj_1095)}, tsbProj: {json.dumps(tsb_proj_1095)}, planned: {json.dumps(planned_1095)} }},
            '1825': {{ dates: {json.dumps(ctl_atl_dates_1825)}, ctl: {json.dumps(ctl_values_1825)}, atl: {json.dumps(atl_values_1825)}, ctlProj: {json.dumps(ctl_proj_1825)}, atlProj: {json.dumps(atl_proj_1825)}, tsbProj: {json.dumps(tsb_proj_1825)}, planned: {json.dumps(planned_1825)} }}
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
        
        // Pre-race morning training load lookup (used by top races, recent runs, pred chart, AG chart)
        const _preRaceTSB = {_pre_race_tsb_json};
        
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
                let rowTitle = '';
                if (_preRaceTSB && race.date_iso && _preRaceTSB[race.date_iso]) {{
                    const pr = _preRaceTSB[race.date_iso];
                    const sign = pr.tsb_pre >= 0 ? '+' : '';
                    rowTitle = `CTL ${{pr.ctl_pre.toFixed(0)}} · ATL ${{pr.atl_pre.toFixed(0)}} · TSB ${{sign}}${{pr.tsb_pre.toFixed(1)}}`;
                }}
                return `
                <tr title="${{rowTitle}}">
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
        
        function conditionsTooltip(temp, solar, elevGain, undulation, surface, distKm) {{
            const lines = [];
            // Temperature (includes solar boost silently)
            if (temp !== null && temp !== undefined) {{
                const solarBoost = (solar && solar > 0) ? solar / 200.0 : 0;
                const tempEff = Math.round(temp + solarBoost);
                const sunIcon = solar > 400 ? '☀️ ' : solar > 200 ? '🌤️ ' : '';
                let tIcon = sunIcon;
                if (!tIcon) {{
                    if (tempEff >= 25) tIcon = '🥵 ';
                    else if (tempEff >= 20) tIcon = '☀️ ';
                    else if (tempEff <= 0) tIcon = '🥶 ';
                }}
                lines.push(tIcon + tempEff + '°C');
            }}
            // Terrain from VAM + undulation
            const gainKm = (elevGain && distKm > 0) ? elevGain / distKm : 0;
            if (gainKm > 5) {{
                const und = undulation || 0;
                const label = gainKm > 12 ? (und > 6 ? 'hilly & rolling' : 'hilly') : (und > 6 ? 'rolling' : 'undulating');
                lines.push('⛰️ ' + label + ' (' + elevGain + 'm gain, ' + gainKm.toFixed(0) + 'm/km)');
            }}
            // Surface
            if (surface && surface !== 'road') {{
                const surfIcons = {{ 'SNOW': '❄️', 'HEAVY_SNOW': '🌨️', 'TRAIL': '🌲', 'TRACK': '🏟️', 'INDOOR_TRACK': '🏟️' }};
                lines.push((surfIcons[surface] || '') + ' ' + surface);
            }}
            return lines;
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
            const predArr = d[predKey] && d[predKey].some(v => v !== null) ? d[predKey] : d.predicted;
            const predicted = indices.map(i => predArr[i]);
            const actual = indices.map(i => d.actual[i]);
            const names = indices.map(i => d.names[i]);
            const temps = indices.map(i => d.temps[i]);
            const surfaces = indices.map(i => d.surfaces[i]);
            const isParkrun = indices.map(i => d.is_parkrun[i]);
            const tempAdjs = indices.map(i => (d.temp_adjs || [])[i] || 1.0);
            const surfaceAdjs = indices.map(i => (d.surface_adjs || [])[i] || 1.0);
            const reAdjs = indices.map(i => (d.re_adjs || [])[i] || 1.0);
            const predSolar = indices.map(i => (d.solar || [])[i]);
            const predElev = indices.map(i => (d.elev_gain || [])[i]);
            const predUnd = indices.map(i => (d.undulation || [])[i]);
            const predDists = indices.map(i => (d.dists || [])[i]);
            
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
                    // RE_Adj: accounts for race-day RE vs prediction RE (Stryd only, 1.0 for GAP)
                    const reFactor = reAdjs[i];
                    return Math.round(p * tempFactor * surfFactor * reFactor);
                }});
            }}
            
            // Colour actual dots: parkruns lighter, non-parkrun races darker
            const dotColors = isParkrun.map(p => p === 1 ? 'rgba(252, 165, 165, 0.8)' : 'rgba(239, 68, 68, 0.9)');
            
            // Build data points for time axis (fall back to actual time if prediction missing)
            const predPoints = datesISO.map((dt, i) => ({{ x: dt, y: displayPredicted[i] !== null ? displayPredicted[i] : actual[i] }}));
            const actualPoints = datesISO.map((dt, i) => ({{ x: dt, y: actual[i] }}));
            
            // Full prediction trend line (weekly smoothed, for distances with few races)
            const trendKey = currentMode === 'gap' ? 'trend_dates_iso_gap' : currentMode === 'sim' ? 'trend_dates_iso_sim' : 'trend_dates_iso';
            const trendValKey = currentMode === 'gap' ? 'trend_values_gap' : currentMode === 'sim' ? 'trend_values_sim' : 'trend_values';
            const trendDates = (d[trendKey] && d[trendKey].length > 0) ? d[trendKey] : (d.trend_dates_iso || []);
            const trendVals = (d[trendValKey] && d[trendValKey].length > 0) ? d[trendValKey] : (d.trend_values || []);
            const trendPoints = trendDates.map((dt, i) => ({{ x: dt, y: trendVals[i] }}));
            const singleRace = actualPoints.length < 2;
            
            // Compute x-axis bounds from race dates
            const raceTimes = datesISO.map(d => new Date(d).getTime());
            const firstRace = Math.min(...raceTimes);
            const lastRace = Math.max(...raceTimes);
            let xDataMin = firstRace;
            let xDataMax = lastRace;
            if (singleRace) {{
                // Extend to present so you see current fitness relative to sparse races
                xDataMax = Math.max(xDataMax, Date.now());
            }}
            const xSpan = Math.max(xDataMax - xDataMin, 180 * 86400000);  // minimum 6 months
            const xPad = xSpan * 0.05;
            const xMin = new Date(xDataMin - xPad).toISOString().slice(0, 10);
            const xMax = new Date(xDataMax + xPad).toISOString().slice(0, 10);
            
            // Compute stable y-axis range covering both adjusted and unadjusted data
            const allYVals = [];
            for (let i = 0; i < actual.length; i++) {{ if (actual[i]) allYVals.push(actual[i]); }}
            for (let i = 0; i < predicted.length; i++) {{ if (predicted[i]) allYVals.push(predicted[i]); }}
            if (singleRace) {{ for (let i = 0; i < trendVals.length; i++) {{ if (trendVals[i]) allYVals.push(trendVals[i]); }} }}
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
                    // Background trend line (only for single-race distances)
                    ...(singleRace && trendPoints.length > 0 ? [{{
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
                        borderWidth: singleRace ? 0 : 2,
                        pointRadius: 3,
                        pointBackgroundColor: (function() {{
                            const modeBase = currentMode === 'gap' ? 'rgba(74, 222, 128,' : currentMode === 'sim' ? 'rgba(249, 115, 22,' : 'rgba(129, 140, 248,';
                            return modeBase + ' 0.8)';
                        }})(),
                        fill: false,
                        tension: 0.3,
                        showLine: !singleRace,
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
                                            const reFactor = reAdjs[i];
                                            compPred = Math.round(predicted[i] * tempFactor * surfFactor * reFactor);
                                        }}
                                        const gap = actual[i] - compPred;
                                        if (Math.abs(gap) < 2) {{
                                            lines.push('Hit prediction exactly');
                                        }} else {{
                                            lines.push(formatPredTime(Math.abs(gap)) + (gap > 0 ? ' slower than prediction' : ' faster than prediction'));
                                        }}
                                    }}
                                    // Condition adjustments breakdown
                                    if (adjustConditions && (tempAdjs[i] !== 1.0 || surfaceAdjs[i] !== 1.0 || reAdjs[i] !== 1.0)) {{
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
                                        if (reAdjs[i] !== 1.0) {{
                                            const rePct = ((reAdjs[i] - 1.0) * 100).toFixed(1);
                                            parts.push('RE ' + (reAdjs[i] > 1 ? '+' : '') + rePct + '%');
                                        }}
                                        if (parts.length > 0) lines.push('Adj: ' + parts.join(', '));
                                    }}
                                    // Conditions (temp with solar, terrain, surface)
                                    const cond = conditionsTooltip(temps[i], predSolar[i], predElev[i], predUnd[i], surfaces[i], predDists[i]);
                                    lines.push(...cond);
                                    // Pre-race morning training load (reverse Banister)
                                    if (_preRaceTSB && isoDate && _preRaceTSB[isoDate]) {{
                                        const pr = _preRaceTSB[isoDate];
                                        const sign = pr.tsb_pre >= 0 ? '+' : '';
                                        lines.push('CTL ' + pr.ctl_pre.toFixed(0) + ' · ATL ' + pr.atl_pre.toFixed(0) + ' · TSB ' + sign + pr.tsb_pre.toFixed(1));
                                    }}
                                    return lines;
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            type: 'time',
                            min: xMin,
                            max: xMax,
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
            // Grey out distance tabs with no races
            document.querySelectorAll('#predToggle button[data-dist]').forEach(btn => {{
                const dist = btn.dataset.dist;
                const d = predData[dist];
                if (!d || !d.dates || d.dates.length === 0) {{
                    btn.style.opacity = '0.3';
                    btn.style.pointerEvents = 'none';
                    btn.title = 'No races at this distance';
                }}
            }});
            // Auto-select first distance with races
            let initDist = '5k';
            for (const dist of ['5k', '10k', 'hm', 'marathon']) {{
                const d = predData[dist];
                if (d && d.dates && d.dates.length > 0) {{ initDist = dist; break; }}
            }}
            document.querySelectorAll('#predToggle button').forEach(b => b.classList.remove('active'));
            const initBtn = document.querySelector(`#predToggle button[data-dist="${{initDist}}"]`);
            if (initBtn) initBtn.classList.add('active');
            currentPredDist = initDist;
            renderPredChart(initDist, showParkruns);
            
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
        const agAllTemps = {json.dumps(ag_data.get('temps', []))};
        const agAllSolar = {json.dumps(ag_data.get('solar', []))};
        const agAllElev = {json.dumps(ag_data.get('elev_gain', []))};
        const agAllUndulation = {json.dumps(ag_data.get('undulation', []))};
        const agAllSurfaces = {json.dumps(ag_data.get('surfaces', []))};
        const agAllDists = {json.dumps(ag_data.get('dists', []))};
        const agAllNames = {json.dumps(ag_data.get('names', []))};
        
        const agCtx = document.getElementById('agChart');
        let agChart = null;
        
        function getAgSlice(days) {{
            if (days >= 99999) return {{ datesISO: agAllDatesISO, dates: agAllDates, values: agAllValues, labels: agAllLabels, colors: agAllColors, sizes: agAllSizes, parkrun: agAllParkrun, rolling: agAllRolling, temps: agAllTemps, solar: agAllSolar, elev_gain: agAllElev, undulation: agAllUndulation, surfaces: agAllSurfaces, dists: agAllDists, names: agAllNames }};
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
                rolling: indices.map(i => agAllRolling[i]),
                temps: indices.map(i => agAllTemps[i]),
                solar: indices.map(i => agAllSolar[i]),
                elev_gain: indices.map(i => agAllElev[i]),
                undulation: indices.map(i => agAllUndulation[i]),
                surfaces: indices.map(i => agAllSurfaces[i]),
                dists: indices.map(i => agAllDists[i]),
                names: indices.map(i => agAllNames[i])
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
                                    if (i < 0 || i >= d.dates.length) return '';
                                    const name = d.names && d.names[i] ? d.names[i] : '';
                                    return d.dates[i] + (name ? ' — ' + name : '');
                                }},
                                label: function(ctx) {{ 
                                    if (ctx.datasetIndex !== 1) return null;
                                    const i = ctx.dataIndex;
                                    if (i < 0 || i >= d.values.length) return '';
                                    const prefix = d.parkrun[i] ? '🅿️ parkrun ' : '';
                                    return prefix + d.labels[i] + ': ' + d.values[i] + '%'; 
                                }},
                                afterBody: function(ctx) {{
                                    if (!ctx.length || ctx[0].datasetIndex !== 1) return [];
                                    const i = ctx[0].dataIndex;
                                    const isoDate = d.datesISO[i];
                                    const lines = [];
                                    // Conditions
                                    const cond = conditionsTooltip(d.temps?d.temps[i]:null, d.solar?d.solar[i]:null, d.elev_gain?d.elev_gain[i]:null, d.undulation?d.undulation[i]:null, d.surfaces?d.surfaces[i]:null, d.dists?d.dists[i]:null);
                                    lines.push(...cond);
                                    // Training load
                                    if (_preRaceTSB && isoDate && _preRaceTSB[isoDate]) {{
                                        const pr = _preRaceTSB[isoDate];
                                        const sign = pr.tsb_pre >= 0 ? '+' : '';
                                        lines.push('CTL ' + pr.ctl_pre.toFixed(0) + ' · ATL ' + pr.atl_pre.toFixed(0) + ' · TSB ' + sign + pr.tsb_pre.toFixed(1));
                                    }}
                                    return lines;
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
                let rowTitle = '';
                if (run.race && _preRaceTSB && run.date_iso && _preRaceTSB[run.date_iso]) {{
                    const pr = _preRaceTSB[run.date_iso];
                    const sign = pr.tsb_pre >= 0 ? '+' : '';
                    rowTitle = `CTL ${{pr.ctl_pre.toFixed(0)}} · ATL ${{pr.atl_pre.toFixed(0)}} · TSB ${{sign}}${{pr.tsb_pre.toFixed(1)}}`;
                }}
                return `
                <tr title="${{rowTitle}}">
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
        const _stdKm = {{'5K':5.0,'10K':10.0,'HM':21.097,'Mara':42.195}};
        // Ordered StepB prediction keys for Riegel interpolation
        const _predOrder = [['5K',5.0,'pred5k_s'],['10K',10.0,'pred10k_s'],['HM',21.097,'predHm_s'],['Mara',42.195,'predMara_s']];
        function _riegelInterp(dist, ms) {{
            // Find bracketing StepB predictions and interpolate via Riegel exponent
            const pts = _predOrder.map(p => ({{d:p[1], t:ms[p[2]]||0}})).filter(p => p.t > 0);
            if (pts.length < 2) return 0;
            // Find bracket
            let lo = pts[0], hi = pts[pts.length - 1];
            for (let i = 0; i < pts.length - 1; i++) {{
                if (dist >= pts[i].d && dist <= pts[i+1].d) {{ lo = pts[i]; hi = pts[i+1]; break; }}
            }}
            if (dist < pts[0].d) {{ lo = pts[0]; hi = pts[1]; }}
            if (dist > pts[pts.length-1].d) {{ lo = pts[pts.length-2]; hi = pts[pts.length-1]; }}
            if (lo.d === hi.d) return lo.t;
            const e = Math.log(hi.t / lo.t) / Math.log(hi.d / lo.d);
            return Math.round(lo.t * Math.pow(dist / lo.d, e));
        }}
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
            // For non-standard distances, interpolate from StepB predictions via Riegel
            const stepbKey = _stepbKeyMap[race.distance_key];
            const isStdDist = _stdKm[race.distance_key] && Math.abs(dist - _stdKm[race.distance_key]) < 0.1;
            const stepbSecs = (stepbKey && isStdDist && (race.surface||'road')==='road') ? ms[stepbKey] : 0;
            let t;
            if (stepbSecs && stepbSecs > 0) {{
                t = stepbSecs;
            }} else if ((race.surface||'road') === 'road') {{
                // Interpolate from mode-specific StepB predictions (respects GAP/SIM/Stryd)
                t = _riegelInterp(dist, ms);
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
    
    // ── Race Week Plan: TSB mini-charts ──
    const _rwpPlans = {_rwp_plans_json};
    
    function drawRWPChart(canvasId, plan) {{
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        const W = rect.width, H = rect.height;
        
        const pts = plan.pts;
        const tsbLo = plan.tsbLo, tsbHi = plan.tsbHi;
        const allVals = pts.map(p => p.tsb).concat([tsbLo, tsbHi, 0]);
        const dMin = Math.min(...allVals), dMax = Math.max(...allVals);
        const yPad = Math.max(3, (dMax - dMin) * 0.18);
        const yMin = dMin - yPad, yMax = dMax + yPad;
        
        const rawStep = (yMax - yMin) / 4;
        const niceStep = rawStep <= 3 ? 2 : rawStep <= 6 ? 5 : rawStep <= 12 ? 10 : 20;
        const ticks = [];
        for (let t = Math.ceil(yMin / niceStep) * niceStep; t <= yMax; t += niceStep) ticks.push(t);
        if (!ticks.includes(0) && yMin < 0 && yMax > 0) ticks.push(0);
        ticks.sort((a, b) => a - b);
        
        const pad = {{ l: 38, r: 16, t: 10, b: 22 }};
        const pW = W - pad.l - pad.r, pH = H - pad.t - pad.b;
        const x = i => pad.l + (i / Math.max(pts.length - 1, 1)) * pW;
        const y = v => pad.t + pH - ((v - yMin) / (yMax - yMin)) * pH;
        
        ctx.fillStyle = 'rgba(0,0,0,0.12)';
        ctx.beginPath(); ctx.roundRect(0, 0, W, H, 6); ctx.fill();
        
        // Target zone band
        ctx.fillStyle = 'rgba(74, 222, 128, 0.07)';
        const zT = Math.max(y(tsbHi), pad.t), zB = Math.min(y(tsbLo), pad.t + pH);
        if (zB > zT) ctx.fillRect(pad.l, zT, pW, zB - zT);
        
        // Grid + labels
        ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        ctx.font = '500 10px "JetBrains Mono", monospace';
        ticks.forEach(t => {{
            const yy = y(t);
            if (yy < pad.t - 2 || yy > pad.t + pH + 2) return;
            ctx.strokeStyle = t === 0 ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.04)';
            ctx.lineWidth = t === 0 ? 1 : 0.5;
            ctx.setLineDash(t === 0 ? [] : [3, 3]);
            ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(W - pad.r, yy); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = 'rgba(139, 144, 160, 0.8)';
            ctx.fillText((t >= 0 ? '+' : '') + t, pad.l - 4, yy);
        }});
        
        // Zone dashed borders
        ctx.strokeStyle = 'rgba(74, 222, 128, 0.3)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
        [tsbLo, tsbHi].forEach(v => {{ const yy = y(v); if (yy >= pad.t && yy <= pad.t + pH) {{ ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(W - pad.r, yy); ctx.stroke(); }} }});
        ctx.setLineDash([]);
        
        // TSB line: solid for done/today, dashed for planned
        const nowIdx = pts.findIndex(p => p.now);
        const lastActual = nowIdx >= 0 ? nowIdx : pts.reduce((a, p, i) => p.t === 'd' ? i : a, 0);
        ctx.beginPath(); ctx.strokeStyle = '#818cf8'; ctx.lineWidth = 2;
        for (let i = 0; i <= Math.min(lastActual, pts.length - 1); i++) {{ if (i === 0) ctx.moveTo(x(i), y(pts[i].tsb)); else ctx.lineTo(x(i), y(pts[i].tsb)); }}
        ctx.stroke();
        if (lastActual < pts.length - 1) {{
            ctx.beginPath(); ctx.strokeStyle = 'rgba(129, 140, 248, 0.5)'; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
            ctx.moveTo(x(lastActual), y(pts[lastActual].tsb));
            for (let i = lastActual + 1; i < pts.length; i++) ctx.lineTo(x(i), y(pts[i].tsb));
            ctx.stroke(); ctx.setLineDash([]);
        }}
        
        // Dots
        pts.forEach((p, i) => {{
            const isActual = (p.t === 'd' || p.now);
            const r = p.r ? 5 : (p.now ? 4.5 : 3.5);
            ctx.beginPath(); ctx.arc(x(i), y(p.tsb), r, 0, Math.PI * 2);
            ctx.fillStyle = p.r ? '#4ade80' : isActual ? '#818cf8' : 'rgba(129, 140, 248, 0.45)';
            ctx.fill(); ctx.strokeStyle = 'rgba(26, 29, 39, 0.8)'; ctx.lineWidth = 1.5; ctx.stroke();
        }});
        
        // X labels
        ctx.font = '500 10px "DM Sans", sans-serif'; ctx.fillStyle = '#8b90a0'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        pts.forEach((p, i) => ctx.fillText(p.l, x(i), pad.t + pH + 6));
        
        // Race morning callout
        const last = pts[pts.length - 1];
        ctx.font = '700 11px "JetBrains Mono", monospace';
        ctx.fillStyle = (last.tsb >= tsbLo && last.tsb <= tsbHi) ? '#4ade80' : (last.tsb < tsbLo ? '#fbbf24' : '#60a5fa');
        const lbl = (last.tsb >= 0 ? '+' : '') + last.tsb.toFixed(1);
        const lx = x(pts.length - 1);
        if (lx + 40 > W - pad.r) {{ ctx.textAlign = 'right'; ctx.fillText(lbl, lx - 8, y(last.tsb) - 5); }}
        else {{ ctx.textAlign = 'left'; ctx.fillText(lbl, lx + 8, y(last.tsb) - 5); }}
    }}
    
    // Draw all TSB charts
    const _DOW = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
    _rwpPlans.forEach(p => {{ if (p.canvas) drawRWPChart(p.canvas, p); }});
    window.addEventListener('resize', () => _rwpPlans.forEach(p => {{ if (p.canvas) drawRWPChart(p.canvas, p); }}));
    
    // Tooltip positioning — flip below if too close to top of viewport
    document.querySelectorAll('.ws-tip').forEach(el => {{
        el.addEventListener('mouseenter', function() {{
            const tip = this.querySelector('.tip');
            if (!tip) return;
            const rect = this.getBoundingClientRect();
            const tipH = tip.offsetHeight || 60;
            let top = rect.top - tipH - 6;
            let left = rect.left + rect.width / 2 - 140;
            if (top < 4) top = rect.bottom + 6;
            left = Math.max(4, Math.min(left, window.innerWidth - 284));
            tip.style.top = top + 'px';
            tip.style.left = left + 'px';
        }});
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
    
    # v53: Load data-driven race factors from Master (overrides hardcoded defaults)
    _load_race_factors_from_master(df)
    
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
    
    # AG RFL: current predicted AG / best ever race AG
    # Current = predicted AG from the athlete's configured power mode
    # Best = max age_grade_pct across all races in master
    _current_ag = None
    if age_grade and age_grade > 30:
        _current_ag = age_grade
    elif _cfg_power_mode in ('gap', 'sim'):
        _mode_ag = race_predictions.get(f'_ag_{_cfg_power_mode}')
        if _mode_ag and _mode_ag > 30:
            _current_ag = _mode_ag
    
    _best_race_ag = None
    race_col = 'race_flag' if 'race_flag' in df.columns else 'Race'
    if 'age_grade_pct' in df.columns and race_col in df.columns:
        _ag_races = df[df[race_col] == 1].copy()
        # Exclude virtual/Zwift races — unreliable treadmill distance
        if 'strava_activity_type' in _ag_races.columns:
            _virtual = _ag_races['strava_activity_type'].str.contains('virtual', case=False, na=False)
        elif 'activity_name' in _ag_races.columns:
            # Fallback: name-based detection for masters without strava_activity_type
            _virtual = _ag_races['activity_name'].str.contains(
                r'\bzwift\b|virtual|\(E\)\s*$|ZLDR|watopia', case=False, na=False, regex=True)
        else:
            _virtual = pd.Series(False, index=_ag_races.index)
            if _virtual.any():
                print(f"  AG RFL: excluded {_virtual.sum()} virtual/Zwift races from peak AG")
                _ag_races = _ag_races[~_virtual]
        _race_ags = _ag_races['age_grade_pct'].dropna()
        _race_ags = _race_ags[(_race_ags >= 30) & (_race_ags < 120)]
        if len(_race_ags) > 0:
            _best_race_ag = round(float(_race_ags.max()), 2)
    
    _ag_rfl = min(round(_current_ag / _best_race_ag * 100, 1), 100.0) if _current_ag and _best_race_ag and _best_race_ag > 0 else None
    stats['ag_rfl'] = _ag_rfl
    stats['best_race_ag'] = _best_race_ag
    stats['current_pred_ag'] = round(_current_ag, 1) if _current_ag else None
    
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
    
    # Milestones data
    print("Computing milestones...")
    milestone_data = get_milestones_data(df)
    n_ms = len(milestone_data.get('milestones', []))
    n_ra = len(milestone_data.get('recent_achievements', []))
    print(f"  {n_ms} milestones, {n_ra} recent achievements")
    
    # Upcoming sessions from planned_sessions.yml (via Daily sheet)
    print("Loading upcoming sessions...")
    _upcoming_result = get_upcoming_sessions(MASTER_FILE)
    if _upcoming_result:
        upcoming_sessions, _completed_tss = _upcoming_result
        # Attach completed TSS as metadata so card builder can include it in first week total
        if upcoming_sessions:
            upcoming_sessions[0]['_completed_tss_this_week'] = _completed_tss
        print(f"  {len(upcoming_sessions)} upcoming planned sessions (completed this week: {_completed_tss} TSS)")
    else:
        upcoming_sessions = None

    # Race History data
    print("Computing race history...")
    race_history_data = get_race_history_data(df, ctl_atl_lookup, zone_data=zone_data)
    
    # Generate HTML
    print("Generating HTML...")
    html = generate_html(stats, rf_data, volume_data, ctl_atl_data, ctl_atl_lookup, rfl_trend_dates, rfl_trend_values, rfl_trendline, rfl_projection, rfl_ci_upper, rfl_ci_lower, alltime_rfl_dates, alltime_rfl_values, recent_runs, top_races, alert_data=alert_data, weight_data=weight_data, prediction_data=prediction_data, ag_data=ag_data, zone_data=zone_data, rfl_trend_gap={'values': rfl_trend_gap_values, 'trendline': rfl_trendline_gap, 'projection': rfl_proj_gap, 'ci_upper': rfl_ci_upper_gap, 'ci_lower': rfl_ci_lower_gap}, rfl_trend_sim={'values': rfl_trend_sim_values, 'trendline': rfl_trendline_sim, 'projection': rfl_proj_sim, 'ci_upper': rfl_ci_upper_sim, 'ci_lower': rfl_ci_lower_sim}, milestone_data=milestone_data, race_history_data=race_history_data, upcoming_sessions=upcoming_sessions)
    
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
