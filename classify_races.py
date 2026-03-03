#!/usr/bin/env python3
"""
classify_races.py — Smart race/training classifier for activity_overrides.xlsx

Detects races from Master file using HR intensity as the primary signal,
with distance-specific HR thresholds (marathons are run at lower HR than 5Ks).

Strategy:
  1. Find all runs at standard race distances (3K–Marathon) with HR data
  2. Classify using HR-for-distance model (sustained 90%+ LTHR at marathon = race)
  3. Boost/demote using activity name keywords (expanded for international events)
  4. Detect parkruns (name, or Saturday ~9am + ~5km)
  5. Detect surface from name keywords (indoor, track)

Usage:
    python classify_races.py \\
        --master athletes/A005/output/Master_FULL.xlsx \\
        --overrides athletes/A005/activity_overrides.xlsx \\
        --athlete-yml athletes/A005/athlete.yml \\
        --from-master
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Font, PatternFill

# ─── Race detection patterns ──────────────────────────────────────────────

# Keywords that strongly indicate a race (expanded for international events)
RACE_KEYWORDS = re.compile(
    r'parkrun|park run|marathon|half marathon|\brace\b|championship|league|'
    r'serpentine|LFOTM|cross country|\bXC\b|assembly|relays?\b|'
    r'mob match|interclub|'
    r'vitality|bupa|great run|great north|big half|'
    # Swedish / European race names
    r'loppet|varvet|rusket|midnattsloppet|vinthund|premiärhalvan|'
    # Common race series / organisers
    r"RunThrough|Brooks 5k|CityRun|STHLM 10|Winter 10k|Stockholm's Bästa|"
    r'Harry Hawkes|BMC\b|\bTT\b|mästerskap|'
    # PB/SB mentions (strong race signal)
    r'\bPB[!\s]|\bSB[!\s]',
    re.I
)

# Keywords that indicate training (not a race)
ANTI_KEYWORDS = re.compile(
    r'tempo|recovery|recover|steady|easy|\bwarm.?up\b|cool.?down|treadmill|'
    r'\bjog\b|fartlek|interval|threshold|long run|out and back|club run|'
    r'zwift|progression|hills?\b|repeat|session|drill|commute|travel|'
    r'shakeout|pre.?match|lunch run|morning run|\bWU\b|\bCD\b|'
    r'sandwich|streak|relaxed|gentle|slow|brisk|progressive|buggy|'
    r'FK Studenterna|FKS\b|\bprep\b|preamble|calibrat',
    re.I
)

# Specific race name matches (abbreviations, event names)
RACE_NAME_OVERRIDES = re.compile(
    r'\bVLM\b|London Marathon|Copenhagen Marathon|Stockholm Marathon|'
    r'Hackney Half|Battersea Park|Djurgårdsvarvet|Höstrusket|Hässelbyloppet|'
    r'Lidingöloppet|Midnattsloppet|2 sjöar',
    re.I
)

# Race keywords excluding parkrun — used to check if "parkrun" was the only match
# at non-5K distances (sandwich runs, longer runs containing a parkrun)
NON_PARKRUN_RACE_KW = re.compile(
    r'marathon|half marathon|\brace\b|championship|league|'
    r'serpentine|LFOTM|cross country|\bXC\b|assembly|relays?\b|'
    r'mob match|interclub|vitality|bupa|great run|great north|big half|'
    r'loppet|varvet|rusket|midnattsloppet|vinthund|premiärhalvan|'
    r"RunThrough|Brooks 5k|CityRun|STHLM 10|Winter 10k|Stockholm's Bästa|"
    r'Harry Hawkes|BMC\b|\bTT\b|\bPB[!\s]|\bSB[!\s]',
    re.I
)

# Surface detection from activity name
INDOOR_KEYWORDS = re.compile(r'\bindoor\b|\btreadmill\b|\bgym\b', re.I)
TRACK_KEYWORDS = re.compile(r'\btrack\b|\b[345]000m\b|\b1500m\b|\bmile\b.*\btrack\b', re.I)

# Standard race distances: (min_km, max_km, official_km, label)
RACE_DISTANCES = [
    (2.8, 3.2, 3.0, "3K"),
    (4.8, 5.5, 5.0, "5K"),
    (9.5, 10.8, 10.0, "10K"),
    (14.5, 16.5, 15.534, "10M"),
    (20.5, 22.0, 21.097, "HM"),
    (29.5, 31.5, 30.0, "30K"),
    (41.0, 43.5, 42.195, "Marathon"),
]

# ─── HR thresholds by distance ────────────────────────────────────────────
# Default race HR thresholds as %LTHR. Loaded from athlete.yml if present,
# otherwise these defaults apply. Athletes should tune to their own data.
# Named races (keyword/override match) get threshold - 5%.
DEFAULT_RACE_HR_THRESHOLDS = {
    "3K":   1.01,   # Near max — first lap drag on short races
    "5K":   1.00,   # First km drags avg down from 180+ effort
    "10K":  1.00,   # Sustained near LTHR
    "10M":  0.98,   # Interpolated
    "HM":   0.97,   # ~172 bpm for LTHR 178
    "30K":  0.94,   # Interpolated
    "Marathon": 0.93,  # ~165 bpm for LTHR 178
}


def load_athlete_config(yml_path: str) -> dict:
    """Load LTHR, max_hr, timezone, and race HR thresholds from athlete.yml."""
    try:
        import yaml
        with open(yml_path) as f:
            cfg = yaml.safe_load(f)
        result = {
            "lthr": cfg["athlete"]["lthr"],
            "max_hr": cfg["athlete"]["max_hr"],
            "timezone": cfg["athlete"].get("timezone", "UTC"),
        }
        # Race HR thresholds: merge athlete overrides with defaults
        athlete_thresholds = cfg["athlete"].get("race_hr_thresholds_pct", {})
        merged = dict(DEFAULT_RACE_HR_THRESHOLDS)
        merged.update(athlete_thresholds)
        result["race_hr_thresholds"] = merged
        return result
    except Exception as e:
        print(f"Warning: Could not read {yml_path}: {e}")
        return {"race_hr_thresholds": dict(DEFAULT_RACE_HR_THRESHOLDS)}


def detect_surface(name: str) -> str:
    """Detect surface type from activity name. Returns None if undetected."""
    if not name:
        return None
    if INDOOR_KEYWORDS.search(name):
        return "indoor"
    if TRACK_KEYWORDS.search(name):
        return "track"
    return None


def is_parkrun(name: str, date_val, dist_km: float, start_hour: float = None) -> bool:
    """Detect parkrun from name, day-of-week, time, and distance.
    
    Parkrun sandwiches (longer run containing a parkrun) are NOT flagged
    as parkrun races — the distance check filters them out.
    """
    if name and re.search(r'parkrun|park run', name, re.I):
        if dist_km and float(dist_km) > 5.1:
            return False  # Sandwich or longer run
        return True
    
    # Saturday + ~5km + morning
    if date_val and dist_km:
        try:
            if isinstance(date_val, str):
                dt = datetime.strptime(str(date_val)[:10], '%Y-%m-%d')
            elif hasattr(date_val, 'weekday'):
                dt = date_val
            else:
                return False
            if dt.weekday() == 5 and 4.8 <= float(dist_km) <= 5.5:
                if start_hour is not None and not (8.0 <= start_hour <= 10.5):
                    return False
                return True
        except (ValueError, TypeError):
            pass
    
    return False


def classify_run(name: str, avg_hr: float, lthr: float, max_hr: float,
                 distance_km: float, duration_min: float,
                 race_type: str,
                 race_hr_thresholds: dict = None) -> tuple:
    """Classify a single candidate as race/training/uncertain.
    
    Primary signal: HR intensity vs distance-specific threshold.
    Secondary signal: activity name keywords.
    
    race_hr_thresholds: dict of {distance_label: pct_lthr} from athlete.yml.
    """
    if race_hr_thresholds is None:
        race_hr_thresholds = DEFAULT_RACE_HR_THRESHOLDS
    
    hr_pct = avg_hr / lthr if lthr > 0 and avg_hr > 0 else 0
    
    has_race_kw = bool(RACE_KEYWORDS.search(name))
    has_race_name = bool(RACE_NAME_OVERRIDES.search(name))
    has_anti_kw = bool(ANTI_KEYWORDS.search(name))
    
    # "parkrun" in a non-5K run is a sandwich/longer run — don't treat as race keyword
    if has_race_kw and race_type != "5K":
        if not NON_PARKRUN_RACE_KW.search(name):
            has_race_kw = False  # Only "parkrun" matched — not a race at this distance
    
    min_hr = race_hr_thresholds.get(race_type, 1.00)
    # Named races get a lower HR bar (name is strong evidence)
    named_hr = min_hr - 0.05
    
    # ── Decision tree ──
    
    # 1. Specific race name override (VLM, Stockholm Marathon etc)
    if has_race_name and not has_anti_kw:
        if hr_pct >= named_hr:
            return ('race', 'high', f'Named race + HR {hr_pct*100:.0f}%LTHR')
        else:
            return ('race', 'low', f'Named race but easy HR {hr_pct*100:.0f}%LTHR — DNS/jogged?')
    
    # 2. Race keyword + no anti-keyword — use relaxed HR threshold
    if has_race_kw and not has_anti_kw:
        if hr_pct >= named_hr:
            return ('race', 'high', f'Race keyword + HR {hr_pct*100:.0f}%LTHR')
        else:
            return ('race', 'low', f'Race keyword but HR {hr_pct*100:.0f}%LTHR — easy effort')
    
    # 3. Anti-keyword only → training (regardless of HR — sessions can have high HR)
    if has_anti_kw and not has_race_kw and not has_race_name:
        return ('training', 'high', f'Training keyword: "{_first_match(ANTI_KEYWORDS, name)}", HR {hr_pct*100:.0f}%LTHR')
    
    # 4. Both race + anti keywords → check context
    if has_race_kw and has_anti_kw:
        # Decisive race keywords override any anti-keyword (e.g. FKS championship)
        decisive_race = re.search(r'championship|mästerskap', name, re.I)
        if decisive_race and hr_pct >= min_hr:
            return ('race', 'high', f'Decisive race keyword: "{decisive_race.group()}" + HR {hr_pct*100:.0f}%LTHR')
        # Decisive anti-keywords override race keywords
        decisive_anti = re.search(
            r'sandwich|calibrat|shakeout|pre.?match|preamble|'
            r'\btempo\b|\bsession\b|\binterval|FK Studenterna|FKS\b|'
            r'\bWU\b.*\bCD\b|\bCD\b.*\bWU\b|warm.?up.*cool.?down|'
            r'\bstreaks?\b|\bprep\b|buggy',
            name, re.I)
        if decisive_anti:
            return ('training', 'high', f'Decisive training keyword: "{decisive_anti.group()}"')
        if hr_pct >= min_hr:
            return ('race', 'medium', f'Mixed keywords, HR {hr_pct*100:.0f}%LTHR suggests race')
        else:
            return ('training', 'medium', f'Mixed keywords, HR {hr_pct*100:.0f}%LTHR suggests training')
    
    # 5. No keywords — rely entirely on calibrated HR thresholds
    if hr_pct >= min_hr:
        return ('race', 'medium', f'HR {hr_pct*100:.0f}%LTHR ≥ {min_hr*100:.0f}% for {race_type} — REVIEW')
    else:
        return ('training', 'medium', f'HR {hr_pct*100:.0f}%LTHR < {min_hr*100:.0f}% for {race_type}')


def _first_match(pattern, text):
    """Return the first regex match from text, for readable reasons."""
    m = pattern.search(text)
    return m.group(0) if m else ""


def detect_candidates_from_master(master_df: pd.DataFrame, lthr: float,
                                  max_hr: float) -> pd.DataFrame:
    """Find ALL runs at standard race distances with HR above 75% LTHR.
    
    No pace filter. The classify_run() step handles the actual classification.
    """
    df = master_df.copy()
    valid = df[df['avg_hr'].notna() & df['distance_km'].notna()].copy()
    
    candidates = []
    for _, row in valid.iterrows():
        dist = row['distance_km']
        hr = row['avg_hr']
        
        matched = False
        official_dist = None
        race_type = None
        for dmin, dmax, odist, label in RACE_DISTANCES:
            if dmin <= dist <= dmax:
                matched = True
                official_dist = odist
                race_type = label
                break
        
        if not matched:
            continue
        
        # Minimum HR floor — below 82% LTHR nothing is a race in calibration data
        # (Easy parkruns at 83-88% are caught by parkrun name detection instead)
        if lthr > 0 and hr < lthr * 0.82:
            continue
        
        candidates.append({
            'file': row['file'],
            'official_distance_km': official_dist,
            'race_type': race_type,
        })
    
    return pd.DataFrame(candidates)


def enrich_and_classify(master_path: str, overrides_path: str,
                        lthr: float, max_hr: float,
                        from_master: bool = False,
                        timezone: str = "UTC",
                        race_hr_thresholds: dict = None) -> str:
    """Main function: detect, classify, enrich, and write overrides."""
    print(f"Loading Master: {master_path}")
    master = pd.read_excel(master_path)
    print(f"  {len(master)} runs, {master['date'].min().date()} to {master['date'].max().date()}")
    
    master_lookup = master.set_index('file').to_dict('index')
    
    # ── Load or generate overrides ──
    if from_master:
        print(f"\nDetecting race candidates from Master...")
        candidates = detect_candidates_from_master(master, lthr, max_hr)
        print(f"  {len(candidates)} candidates at race distances")
        
        ov = candidates[['file', 'official_distance_km', 'race_type']].copy()
        ov['race_flag'] = 1
        ov['parkrun'] = 0
        ov['surface'] = None
        ov['surface_adj'] = None
        ov['temp_override'] = None
        ov['power_override_w'] = None
        ov['official_time_s'] = None
        ov['notes'] = ''
    else:
        print(f"\nLoading overrides: {overrides_path}")
        ov = pd.read_excel(overrides_path, dtype={'file': str})
        print(f"  {len(ov)} rows")
        
        rename_map = {}
        if 'race' in ov.columns and 'race_flag' not in ov.columns:
            rename_map['race'] = 'race_flag'
        if 'distance_km' in ov.columns and 'official_distance_km' not in ov.columns:
            rename_map['distance_km'] = 'official_distance_km'
        if 'temp_c' in ov.columns and 'temp_override' not in ov.columns:
            rename_map['temp_c'] = 'temp_override'
        if rename_map:
            ov.rename(columns=rename_map, inplace=True)
        
        if 'race_flag' in ov.columns:
            ov['race_flag'] = ov['race_flag'].astype(int)
        if 'parkrun' in ov.columns:
            ov['parkrun'] = ov['parkrun'].fillna(0).astype(int)
        
        if 'race_type' not in ov.columns:
            ov['race_type'] = None
            for i, row in ov.iterrows():
                dist = row.get('official_distance_km', 0)
                if dist:
                    for _, _, odist, label in RACE_DISTANCES:
                        if abs(dist - odist) < 0.5:
                            ov.at[i, 'race_type'] = label
                            break
    
    # ── Classify ──
    print(f"\nClassifying {len(ov)} candidates...")
    enriched_cols = {
        'date': [], 'activity_name': [], 'actual_dist_km': [],
        'avg_hr': [], 'hr_pct_lthr': [], 'duration_min': [],
        'verdict': [], 'confidence': [], 'reason': []
    }
    
    race_count = training_count = uncertain_count = 0
    
    for _, row in ov.iterrows():
        fname = row['file']
        m = master_lookup.get(fname, {})
        
        date = m.get('date')
        name = str(m.get('activity_name', '') or '')
        dist = m.get('distance_km', 0) or 0
        hr = m.get('avg_hr', 0) or 0
        elapsed = m.get('elapsed_time_s', 0) or 0
        duration = elapsed / 60
        race_type = row.get('race_type', '') or ''
        
        enriched_cols['date'].append(
            date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10] if date else '')
        enriched_cols['activity_name'].append(name)
        enriched_cols['actual_dist_km'].append(round(dist, 2) if dist else None)
        enriched_cols['avg_hr'].append(int(hr) if hr else None)
        enriched_cols['hr_pct_lthr'].append(round(hr / lthr * 100, 0) if hr and lthr else None)
        enriched_cols['duration_min'].append(round(duration, 1) if duration else None)
        
        verdict, conf, reason = classify_run(name, hr, lthr, max_hr, dist, duration, race_type,
                                               race_hr_thresholds=race_hr_thresholds)
        
        enriched_cols['verdict'].append(f"{verdict} ({conf})")
        enriched_cols['confidence'].append(conf)
        enriched_cols['reason'].append(reason)
        
        if verdict == 'race':
            race_count += 1
        elif verdict == 'training':
            training_count += 1
        else:
            uncertain_count += 1
    
    for col, values in enriched_cols.items():
        ov[col] = values
    
    # ── Apply verdicts to race_flag ──
    for i, row in ov.iterrows():
        v = row['verdict']
        if 'training' in v:
            ov.at[i, 'race_flag'] = 0
        elif 'race (high)' in v:
            ov.at[i, 'race_flag'] = 1
        elif 'race (low)' in v:
            # Race keyword present but HR below race threshold → didn't race it
            ov.at[i, 'race_flag'] = 0
            ov.at[i, 'notes'] = f"REVIEW: {row['reason']}"
        elif 'race (medium)' in v:
            ov.at[i, 'race_flag'] = 1
            if 'REVIEW' in row.get('reason', ''):
                ov.at[i, 'notes'] = f"REVIEW: {row['reason']}"
        elif 'uncertain' in v:
            ov.at[i, 'race_flag'] = 0
            ov.at[i, 'notes'] = f"REVIEW: {row['reason']}"
    
    # ── Detect parkruns ──
    # parkrun=1 marks the event (always set if it's a parkrun).
    # race_flag is set independently by the HR-based classification above.
    # A run can be parkrun=1, race_flag=0 (jogged/buggy parkrun).
    parkrun_count = 0
    for i, row in ov.iterrows():
        fname = row['file']
        m = master_lookup.get(fname, {})
        name = str(m.get('activity_name', '') or '')
        date_val = m.get('date')
        dist = m.get('distance_km')
        start_hour = None
        if hasattr(date_val, 'hour'):
            start_hour = date_val.hour + date_val.minute / 60
        if is_parkrun(name, date_val, dist, start_hour):
            ov.at[i, 'parkrun'] = 1
            # race_flag already set by classify_run — don't override
            parkrun_count += 1
    
    # ── Detect surface from activity name ──
    surface_count = 0
    for i, row in ov.iterrows():
        if pd.notna(row.get('surface')) and row['surface']:
            continue
        fname = row['file']
        m = master_lookup.get(fname, {})
        name = str(m.get('activity_name', '') or '')
        detected = detect_surface(name)
        if detected:
            ov.at[i, 'surface'] = detected
            surface_count += 1
    
    # ── Sort by date ──
    ov.sort_values('date', inplace=True)
    
    # ── Summary ──
    final_races = (ov['race_flag'] == 1).sum()
    final_training = (ov['race_flag'] == 0).sum()
    review_count = ov['notes'].str.contains('REVIEW', na=False).sum()
    
    print(f"\n{'='*60}")
    print(f"  Classification results")
    print(f"{'='*60}")
    print(f"  Race (high confidence):     {race_count}")
    print(f"  Training:                   {training_count}")
    print(f"  Uncertain:                  {uncertain_count}")
    print(f"{'─'*60}")
    print(f"  Final race_flag=1:          {final_races}")
    print(f"  Final race_flag=0:          {final_training}")
    print(f"  Parkruns detected:          {parkrun_count}")
    print(f"  Surface detected:           {surface_count}")
    print(f"  Rows marked REVIEW:         {review_count}")
    print(f"{'='*60}")
    
    # ── Write output ──
    output_path = overrides_path
    core_cols = ['file', 'race_flag', 'parkrun', 'official_distance_km',
                 'surface', 'surface_adj', 'temp_override', 'power_override_w', 'official_time_s']
    enrich_cols = ['date', 'activity_name', 'actual_dist_km', 'avg_hr',
                   'hr_pct_lthr', 'duration_min', 'verdict', 'reason', 'notes']
    
    for c in core_cols + enrich_cols:
        if c not in ov.columns:
            ov[c] = None
    
    out_cols = core_cols + enrich_cols
    ov_out = ov[out_cols].copy()
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "overrides"
    
    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    review_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    race_fill = PatternFill(start_color="D5F5E3", end_color="D5F5E3", fill_type="solid")
    training_fill = PatternFill(start_color="FADBD8", end_color="FADBD8", fill_type="solid")
    
    for col_idx, header in enumerate(out_cols, 1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = Font(bold=True)
        cell.fill = header_fill
    
    for row_idx, (_, row) in enumerate(ov_out.iterrows(), 2):
        for col_idx, col in enumerate(out_cols, 1):
            val = row[col]
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val) if not np.isnan(val) else None
            elif pd.isna(val):
                val = None
            ws.cell(row=row_idx, column=col_idx, value=val)
        
        verdict = str(row.get('verdict', ''))
        notes = str(row.get('notes', '') or '')
        if 'REVIEW' in notes:
            for c in range(1, len(out_cols) + 1):
                ws.cell(row=row_idx, column=c).fill = review_fill
        elif 'race' in verdict:
            for c in range(1, len(out_cols) + 1):
                ws.cell(row=row_idx, column=c).fill = race_fill
        elif 'training' in verdict:
            for c in range(1, len(out_cols) + 1):
                ws.cell(row=row_idx, column=c).fill = training_fill
    
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 3, 60)
    
    ws.freeze_panes = 'A2'
    ws.auto_filter.ref = ws.dimensions
    
    wb.save(output_path)
    print(f"\nSaved: {output_path}")
    print(f"  Green = race, Red = training, Yellow = REVIEW needed")
    
    return output_path


def _backfill_parkruns(ov, master_path, overrides_path):
    """Lightweight parkrun detection — runs even when full classification is skipped."""
    if 'parkrun' not in ov.columns:
        ov['parkrun'] = 0
    
    try:
        master = pd.read_excel(master_path)
    except Exception as e:
        print(f"  Could not load Master for parkrun backfill: {e}")
        return
    
    master_lookup = {}
    for _, row in master.iterrows():
        fname = str(row.get('file', ''))
        if fname:
            master_lookup[fname] = {
                'activity_name': str(row.get('activity_name', '') or ''),
                'date': row.get('date', ''),
                'distance_km': row.get('distance_km', None),
            }
    
    filled = 0
    for i, row in ov.iterrows():
        if row.get('parkrun', 0) == 1:
            continue
        fname = str(row.get('file', ''))
        m = master_lookup.get(fname, {})
        name = m.get('activity_name', '')
        date_val = m.get('date', '')
        dist = m.get('distance_km', None)
        if is_parkrun(name, date_val, dist):
            ov.at[i, 'parkrun'] = 1
            if 'race_flag' in ov.columns:
                ov.at[i, 'race_flag'] = 1
            filled += 1
    
    if filled > 0:
        ov.to_excel(overrides_path, index=False)
        print(f"  Parkrun backfill: marked {filled} new parkrun(s).")
    else:
        print(f"  Parkrun backfill: no new parkruns found.")


def main():
    p = argparse.ArgumentParser(description="Smart race classifier for activity overrides")
    p.add_argument("--master", required=True, help="Path to Master_FULL.xlsx")
    p.add_argument("--overrides", required=True, help="Path to activity_overrides.xlsx")
    p.add_argument("--athlete-yml", help="Path to athlete.yml (for LTHR/max HR)")
    p.add_argument("--lthr", type=int, help="Lactate threshold HR (overrides athlete.yml)")
    p.add_argument("--max-hr", type=int, help="Max HR (overrides athlete.yml)")
    p.add_argument("--from-master", action="store_true",
                   help="Generate candidates from Master instead of reading existing overrides")
    p.add_argument("--skip-if-classified", action="store_true",
                   help="Skip if overrides already has verdict column (preserves athlete edits)")
    args = p.parse_args()
    
    if args.skip_if_classified and Path(args.overrides).exists():
        try:
            existing = pd.read_excel(args.overrides)
            skip = False
            if 'verdict' in existing.columns:
                print(f"Overrides already classified (has 'verdict' column) — skipping.")
                skip = True
            elif 'race_flag' in existing.columns and existing['race_flag'].notna().any():
                n_races = (existing['race_flag'] == 1).sum()
                print(f"Overrides already has {n_races} race flags set — skipping.")
                skip = True
            if skip:
                _backfill_parkruns(existing, args.master, args.overrides)
                return
        except Exception:
            pass
    
    lthr = args.lthr
    max_hr = args.max_hr
    timezone = "UTC"
    race_hr_thresholds = None
    
    if args.athlete_yml and (lthr is None or max_hr is None):
        cfg = load_athlete_config(args.athlete_yml)
        if lthr is None:
            lthr = cfg.get("lthr", 160)
        if max_hr is None:
            max_hr = cfg.get("max_hr", 185)
        timezone = cfg.get("timezone", "UTC")
        race_hr_thresholds = cfg.get("race_hr_thresholds")
    
    if lthr is None:
        lthr = 160
        print(f"Warning: using default LTHR={lthr}")
    if max_hr is None:
        max_hr = 185
        print(f"Warning: using default max_hr={max_hr}")
    
    print(f"HR thresholds: LTHR={lthr}, max HR={max_hr}")
    if race_hr_thresholds:
        print(f"Race HR thresholds (from athlete.yml): { {k: f'{v:.0%}' for k,v in race_hr_thresholds.items()} }")
    else:
        print(f"Race HR thresholds: defaults")
    
    enrich_and_classify(args.master, args.overrides, lthr, max_hr,
                        from_master=args.from_master, timezone=timezone,
                        race_hr_thresholds=race_hr_thresholds)


if __name__ == "__main__":
    main()
