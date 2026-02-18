"""
Auto-detect races from Strava activity names and generate activity_overrides.xlsx.

Strategy:
  1. Parkruns: always flag (keyword "parkrun" is unambiguous)
  2. Named events: match known race name patterns (London Marathon, etc.)
  3. Distance keywords (5k, 10k, HM, marathon, mile): only flag if NO training
     keywords are also present (tempo, session, easy, reps, etc.)

Usage:
    python auto_flag_races.py activities.csv [--output activity_overrides.xlsx] [--merge] [--dry-run]
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Training keywords that suppress race detection ──────────────────────
TRAINING_SUPPRESSORS = re.compile(
    r'\b(?:'
    r'tempo|session|training|steady|easy|easyish|recovery|warm[- ]?up|cool[- ]?down|'
    r'jog|interval|rep[s]?\b|alt\b|progression|strides|fartlek|'
    r'PE\b|lesson|aborted|supporting|spectating|'
    r'efforts?\b|reps?\b|rest\b|off\s+\d|'
    r'continuous|float|speed\s+session|track\s+session|'
    r'treadmill|marathon\s+group|morning\s+run|LSR|brisk|testing|'
    r'laps?\b|2nd\s+half|min/mile|min\s+mile'
    r')\b',
    re.I
)

PACE_REFERENCE = re.compile(r'@\s*(?:5k|10k|HM|marathon|mile|MP|HMP)\s*pace', re.I)
AT_PACE = re.compile(r'\bat\s+(?:5k|10k|HM|marathon|mile|MP|HMP)\s*pace', re.I)
# "@ MP" or "at HMP" (e.g., "Mile at HMP", "70 mins @ MP")
AT_MP = re.compile(r'(?:@|at)\s*(?:MP|HMP)\b', re.I)
# "building towards X pace"
BUILDING_TOWARDS = re.compile(r'building\s+towards', re.I)
# Approximate distance "~ 5k" = casual
APPROX_DIST = re.compile(r'~\s*\d+\s*[kK]', re.I)
# "to HM distance" = training run at that distance, not a race
TO_DIST = re.compile(r'to\s+(?:HM|marathon|5k|10k)\s+distance', re.I)
# "HM Pace" or "5k pace" as standalone phrase
PACE_MENTION = re.compile(r'\b(?:HM|marathon|5k|10k|mile)\s+[Pp]ace\b')
# "slow" / "crap" = clearly not racing
CASUAL_RUN = re.compile(r'\b(?:slow|crap|gentle|shuffle)\b', re.I)

# ── Definite race patterns (always flag, ignore suppressors) ────────────
DEFINITE_RACE_PATTERNS = [
    (re.compile(r'\bparkrun\b', re.I), 'parkrun', 5.0),
    (re.compile(r'\bpark\s+run\b', re.I), 'parkrun', 5.0),
    (re.compile(r'(?:London|Brighton|Stockholm|Berlin|Chicago|New York|Boston|Paris|Tokyo)\s+Marathon', re.I), 'marathon', 42.195),
    (re.compile(r'(?:Paddock Wood|Great North|Reading|Bath|Copenhagen|Örebro|Oxford)\s+(?:Half|HM)', re.I), 'half_marathon', 21.097),
    (re.compile(r'\bSurrey League\s+XC\b', re.I), 'xc', None),
    (re.compile(r'\bKent (?:Masters\s+)?XC\b', re.I), 'xc', None),
    (re.compile(r'\bSEAA\s+Masters\b', re.I), 'xc', None),
    (re.compile(r'\bNational XC\b', re.I), 'xc', None),
    (re.compile(r'\bSri Chinmoy\b', re.I), 'race', None),
    (re.compile(r'\bMark Hayes Mile\b', re.I), 'mile', 1.609),
    (re.compile(r'\bBeer Mile\b', re.I), 'mile', 1.609),
    (re.compile(r'\bSoar Mile\b', re.I), 'mile', 1.609),
    (re.compile(r'\bTracksmith.*Mile\s+Race\b', re.I), 'mile', 1.609),
    (re.compile(r'\bBannister.*Mile\b', re.I), 'mile', 1.609),
    (re.compile(r'\bMile Race\b', re.I), 'mile', 1.609),
    (re.compile(r'\bVirtual Marathon\b', re.I), 'marathon', 42.195),
    (re.compile(r'\bLidingö', re.I), 'trail_race', 30.0),
    (re.compile(r'\bMidnattsloppet\b', re.I), '10k', 10.0),
    (re.compile(r'\bChamps?\b', re.I), 'race', None),
]

# ── Conditional distance patterns (suppressed by training context) ──────
CONDITIONAL_DISTANCE_PATTERNS = [
    (re.compile(r'\bmarathon\b(?!.*\bhalf\b)', re.I), 'marathon', 42.195),
    (re.compile(r'\bhalf\s*marathon\b', re.I), 'half_marathon', 21.097),
    (re.compile(r'\bhalvmarathon\b', re.I), 'half_marathon', 21.097),
    (re.compile(r'\bHM\b'), 'half_marathon', 21.097),
    (re.compile(r'\b10\s*[kK](?:m|M)?\b'), '10k', 10.0),
    (re.compile(r'\b5\s*[kK](?:m|M)?\b'), '5k', 5.0),
    (re.compile(r'\b3\s*[kK]\b'), '3k', 3.0),
    (re.compile(r'\b1500\s*[mM]\b'), '1500m', 1.5),
    (re.compile(r'\bmile\b', re.I), 'mile', 1.609),
    (re.compile(r'\blopp\b', re.I), 'race', None),
    (re.compile(r'\btävling\b', re.I), 'race', None),
    (re.compile(r'\brace\b', re.I), 'race', None),
    (re.compile(r'\bXC\b'), 'xc', None),
    (re.compile(r'\bcross\s*country\b', re.I), 'xc', None),
    (re.compile(r'\btime\s*trial\b', re.I), 'time_trial', None),
    (re.compile(r'\brelay\b', re.I), 'relay', None),
]

EVENT_NAME_PATTERN = re.compile(
    r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+'
    r'(?:10[kK]|5[kK]|HM|Half|Marathon|Mile|XC)',
    re.UNICODE
)

STANDARD_DISTANCES = [
    (1.5, 1.7, 'mile', 1.609),
    (2.9, 3.2, '3k', 3.0),
    (4.7, 5.3, '5k', 5.0),
    (9.5, 10.5, '10k', 10.0),
    (20.5, 21.5, 'half_marathon', 21.097),
    (41.0, 43.0, 'marathon', 42.195),
]


def is_training_context(name: str) -> bool:
    """Check if the name contains training/workout indicators."""
    if TRAINING_SUPPRESSORS.search(name):
        return True
    if PACE_REFERENCE.search(name):
        return True
    if AT_PACE.search(name):
        return True
    if AT_MP.search(name):
        return True
    if BUILDING_TOWARDS.search(name):
        return True
    if APPROX_DIST.search(name):
        return True
    if TO_DIST.search(name):
        return True
    if PACE_MENTION.search(name):
        return True
    if CASUAL_RUN.search(name):
        return True
    if re.search(r'\d+\s*[xX×]\s*\d+', name):
        return True
    if re.match(r'Day\s+\d+', name, re.I):
        return True
    return False


def detect_race(name: str, distance_km: float = None) -> dict:
    """Check if activity name indicates a race."""
    if not name or not isinstance(name, str):
        return None

    # Phase 1: Definite races (always flag)
    for pattern, race_type, official_dist in DEFINITE_RACE_PATTERNS:
        if pattern.search(name):
            if race_type == 'parkrun' and re.search(r'\b(?:post|pre|after|before|missed)\s+parkrun\b', name, re.I):
                return None
            # "Sri Chinmoy warm up" — not the race itself
            if re.search(r'\bwarm[- ]?up\b', name, re.I):
                return None
            result = {
                'race_flag': 1,
                'race_type': race_type,
                'official_distance_km': official_dist,
                'match_source': 'definite',
            }
            if race_type == 'parkrun':
                result['parkrun'] = 1
            if official_dist is None and distance_km and np.isfinite(distance_km):
                for lo, hi, dtype, dval in STANDARD_DISTANCES:
                    if lo <= distance_km <= hi:
                        result['official_distance_km'] = dval
                        break
            return result

    # Phase 2: Event name pattern (proper noun + distance)
    if EVENT_NAME_PATTERN.match(name) and not is_training_context(name):
        for pattern, race_type, official_dist in CONDITIONAL_DISTANCE_PATTERNS:
            if pattern.search(name):
                result = {
                    'race_flag': 1,
                    'race_type': race_type,
                    'official_distance_km': official_dist,
                    'match_source': 'event_name',
                }
                if official_dist is None and distance_km and np.isfinite(distance_km):
                    for lo, hi, dtype, dval in STANDARD_DISTANCES:
                        if lo <= distance_km <= hi:
                            result['official_distance_km'] = dval
                            break
                return result

    # Phase 3: Conditional keywords (suppressed by training context)
    if is_training_context(name):
        return None

    for pattern, race_type, official_dist in CONDITIONAL_DISTANCE_PATTERNS:
        if pattern.search(name):
            result = {
                'race_flag': 1,
                'race_type': race_type,
                'official_distance_km': official_dist,
                'match_source': 'keyword',
            }
            if official_dist is None and distance_km and np.isfinite(distance_km):
                for lo, hi, dtype, dval in STANDARD_DISTANCES:
                    if lo <= distance_km <= hi:
                        result['official_distance_km'] = dval
                        break
            return result

    return None


def load_strava_activities(csv_path: str) -> pd.DataFrame:
    """Load Strava activities.csv, filter to runs."""
    for enc in ('utf-8', 'cp1252', 'latin-1'):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        raise ValueError(f"Could not read {csv_path} with any encoding")

    type_col = 'Activity Type' if 'Activity Type' in df.columns else 'type'
    df = df[df[type_col].astype(str).str.lower() == 'run'].copy()

    name_col = 'Activity Name' if 'Activity Name' in df.columns else 'name'
    date_col = 'Activity Date' if 'Activity Date' in df.columns else 'date'
    dist_col = 'Distance' if 'Distance' in df.columns else 'distance'
    file_col = 'Filename' if 'Filename' in df.columns else 'filename'
    hr_col = 'Average Heart Rate' if 'Average Heart Rate' in df.columns else None
    time_col = 'Moving Time' if 'Moving Time' in df.columns else None

    df['_name'] = df[name_col].astype(str).fillna('')
    df['_date'] = pd.to_datetime(df[date_col], errors='coerce')
    df['_dist_km'] = pd.to_numeric(df.get(dist_col), errors='coerce')
    if df['_dist_km'].median(skipna=True) > 1000:
        df['_dist_km'] = df['_dist_km'] / 1000.0
    
    if hr_col:
        df['_avg_hr'] = pd.to_numeric(df[hr_col], errors='coerce')
    else:
        df['_avg_hr'] = np.nan
    
    if time_col:
        df['_moving_s'] = pd.to_numeric(df[time_col], errors='coerce')
    else:
        df['_moving_s'] = np.nan

    if file_col in df.columns:
        df['_file'] = df[file_col].astype(str).str.strip()
    else:
        df['_file'] = df['_date'].dt.strftime('%Y-%m-%d')

    return df


def generate_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Scan activities and generate override rows for detected races."""
    rows = []
    for _, act in df.iterrows():
        result = detect_race(act['_name'], act['_dist_km'])
        if result is None:
            continue
        # Pace for review
        _pace_str = ''
        if pd.notna(act['_dist_km']) and act['_dist_km'] > 0 and pd.notna(act.get('_moving_s')) and act['_moving_s'] > 0:
            _pace = act['_moving_s'] / act['_dist_km']
            _pace_str = f"{int(_pace//60)}:{int(_pace%60):02d}/km"
        
        row = {
            'file': act['_file'],
            'activity_name': act['_name'],
            'date': act['_date'].strftime('%Y-%m-%d') if pd.notna(act['_date']) else '',
            'race_flag': 1,
            'parkrun': result.get('parkrun', 0),
            'official_distance_km': result.get('official_distance_km'),
            'race_type': result.get('race_type', ''),
            'avg_hr': int(act['_avg_hr']) if pd.notna(act.get('_avg_hr')) and act['_avg_hr'] > 0 else '',
            'pace': _pace_str,
            'surface': '',
            'surface_adj': '',
            'temp_override': '',
            'notes': f"Auto: {result.get('match_source', '')}",
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out = out.sort_values('date').reset_index(drop=True)
    return out


def merge_with_existing(new_df: pd.DataFrame, existing_path: str) -> pd.DataFrame:
    """Merge new detections with existing overrides. Existing entries win."""
    try:
        existing = pd.read_excel(existing_path)
    except Exception:
        return new_df
    if len(existing) == 0:
        return new_df
    existing_files = set(existing['file'].astype(str).str.strip())
    new_only = new_df[~new_df['file'].astype(str).str.strip().isin(existing_files)]
    merged = pd.concat([existing, new_only], ignore_index=True)
    merged = merged.sort_values('date' if 'date' in merged.columns else 'file').reset_index(drop=True)
    return merged


def main():
    parser = argparse.ArgumentParser(description='Auto-detect races from Strava activity names')
    parser.add_argument('activities_csv', help='Path to Strava activities.csv')
    parser.add_argument('--output', '-o', default='activity_overrides.xlsx',
                        help='Output file (default: activity_overrides.xlsx)')
    parser.add_argument('--merge', action='store_true',
                        help='Merge with existing overrides (existing entries take priority)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show detections without writing file')
    args = parser.parse_args()

    print(f"Loading {args.activities_csv}...")
    df = load_strava_activities(args.activities_csv)
    print(f"  {len(df)} run activities loaded")

    print("Scanning for races...")
    overrides = generate_overrides(df)
    n_races = len(overrides)
    n_parkrun = (overrides.get('parkrun', 0) == 1).sum() if n_races > 0 else 0
    print(f"  {n_races} races detected ({n_parkrun} parkruns)")

    if n_races > 0:
        type_counts = overrides['race_type'].value_counts()
        for rtype, count in type_counts.items():
            print(f"    {rtype}: {count}")

    if args.dry_run:
        print("\n--- Dry run ---")
        if n_races > 0:
            for _, r in overrides.iterrows():
                src = r.get('notes', '')
                print(f"  {r['date']}  {r['race_type']:15s}  {str(r.get('official_distance_km','')):>6}  {r['activity_name'][:70]}  [{src}]")
        return

    if args.merge and os.path.exists(args.output):
        print(f"Merging with existing {args.output}...")
        existing_count = len(pd.read_excel(args.output))
        overrides = merge_with_existing(overrides, args.output)
        new_count = len(overrides) - existing_count
        print(f"  {existing_count} existing + {new_count} new = {len(overrides)} total")

    overrides.to_excel(args.output, index=False)
    print(f"\nWritten {len(overrides)} rows to {args.output}")
    print("Review and edit — add surface, surface_adj, temp_override where needed.")
    print("Remove any false positives (training runs incorrectly flagged).")


if __name__ == '__main__':
    main()
