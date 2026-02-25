#!/usr/bin/env python3
"""
classify_races.py — Smart race/training classifier for activity_overrides.xlsx

Uses the Master file's activity names, HR data, and RF trends to distinguish
real races from fast training runs. Enriches the overrides file with date,
activity name, HR stats, and a confidence-based race classification.

Usage:
    python classify_races.py \
        --master athletes/SteveDavies/output/Master_FULL_post.xlsx \
        --overrides athletes/SteveDavies/activity_overrides.xlsx \
        --lthr 157 --max-hr 173

    # Or auto-detect from athlete.yml:
    python classify_races.py \
        --master athletes/SteveDavies/output/Master_FULL_post.xlsx \
        --overrides athletes/SteveDavies/activity_overrides.xlsx \
        --athlete-yml athletes/SteveDavies/athlete.yml

    # Generate fresh overrides from Master (skip onboard_athlete scan):
    python classify_races.py \
        --master athletes/SteveDavies/output/Master_FULL_post.xlsx \
        --overrides athletes/SteveDavies/activity_overrides.xlsx \
        --athlete-yml athletes/SteveDavies/athlete.yml \
        --from-master

Output:
    Overwrites activity_overrides.xlsx with enriched columns and smart verdicts.
    High-confidence races get race_flag=1, high-confidence training get race_flag=0.
    Uncertain rows get race_flag=1 with a "REVIEW" note for the athlete.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Font, PatternFill

# ─── Race detection patterns ──────────────────────────────────────────────

RACE_KEYWORDS = re.compile(
    r'parkrun|park run|marathon|half marathon|\brace\b|championship|league|'
    r'serpentine 5k|LFOTM|cross country|\bXC\b|assembly|relays?|'
    r'\b10k\b|\b5k\b|\bhm\b|mob match|interclub|'
    r'vitality|bupa|great run|great north|big half',
    re.I
)

ANTI_KEYWORDS = re.compile(
    r'tempo|recovery|recover|steady|easy|\bwarm.?up\b|cool.?down|treadmill|'
    r'\bjog\b|fartlek|interval|threshold|long run|out and back|club run|'
    r'zwift|progression|hills?|repeat|session|drill|commute|travel|'
    r'shakeout|pre.?match|lunch run|morning run',
    re.I
)

# Standard race distances for auto-detection from Master
RACE_DISTANCES = [
    (2.8, 3.2, 3.0, "3K"),
    (4.8, 5.5, 5.0, "5K"),
    (9.5, 10.8, 10.0, "10K"),
    (14.5, 16.5, 15.534, "10M"),
    (20.5, 22.0, 21.097, "HM"),
    (29.5, 31.5, 30.0, "30K"),
    (41.0, 43.5, 42.195, "Marathon"),
]


def load_athlete_config(yml_path: str) -> dict:
    """Load LTHR and max_hr from athlete.yml."""
    try:
        import yaml
        with open(yml_path) as f:
            cfg = yaml.safe_load(f)
        return {
            "lthr": cfg["athlete"]["lthr"],
            "max_hr": cfg["athlete"]["max_hr"],
        }
    except Exception as e:
        print(f"Warning: Could not read {yml_path}: {e}")
        return {}


def classify_run(name: str, avg_hr: float, lthr: float, max_hr: float,
                 distance_km: float, pace_min_km: float,
                 pace_p10: float, pace_median: float) -> tuple[str, str, str]:
    """Classify a single run as race/training/uncertain.
    
    Returns (verdict, confidence, reason) where:
        verdict:    'race', 'training', 'uncertain'
        confidence: 'high', 'medium', 'low'
        reason:     human-readable explanation
    """
    hr_pct = avg_hr / lthr if lthr > 0 and avg_hr > 0 else 0
    hr_pct_max = avg_hr / max_hr if max_hr > 0 and avg_hr > 0 else 0
    
    has_race_kw = bool(RACE_KEYWORDS.search(name))
    has_anti_kw = bool(ANTI_KEYWORDS.search(name))
    
    # Is this a standard race distance?
    is_race_dist = False
    for dmin, dmax, _, label in RACE_DISTANCES:
        if dmin <= distance_km <= dmax:
            is_race_dist = True
            break
    
    # ── Decision tree ──
    
    # 1. Anti-keyword WITHOUT race keyword → training
    if has_anti_kw and not has_race_kw:
        return ('training', 'high', f'Training keyword in name, HR {hr_pct*100:.0f}%LTHR')
    
    # 2. Race keyword + high HR → race
    if has_race_kw and hr_pct > 0.88:
        return ('race', 'high', f'Race keyword + HR {hr_pct*100:.0f}%LTHR')
    
    # 3. Race keyword + low HR → easy race (jogged parkrun etc)
    if has_race_kw and hr_pct <= 0.88:
        return ('race', 'low', f'Race keyword but easy HR {hr_pct*100:.0f}%LTHR — jogged?')
    
    # 4. Both keywords (conflicting) → trust HR
    if has_anti_kw and has_race_kw and hr_pct > 0.90:
        return ('race', 'medium', f'Conflicting keywords, HR {hr_pct*100:.0f}%LTHR suggests race')
    
    # 5. No keywords — use HR and pace (strict: only flag as race with very high HR)
    if hr_pct > 0.97 and is_race_dist and pace_min_km < pace_p10 * 1.05:
        return ('race', 'medium', f'No name clue, but HR {hr_pct*100:.0f}%LTHR + race distance + fast pace — REVIEW')
    
    if hr_pct < 0.90:
        return ('training', 'medium', f'No race keyword, HR {hr_pct*100:.0f}%LTHR')
    
    # 6. Middle ground — default to training, flag for review
    return ('uncertain', 'low', f'HR {hr_pct*100:.0f}%LTHR, no clear signal — REVIEW')


def detect_candidates_from_master(master_df: pd.DataFrame, lthr: float,
                                  max_hr: float) -> pd.DataFrame:
    """Detect race candidates from Master file (replaces onboard FIT scanning).
    
    Uses pace distribution + HR + distance to find candidate races.
    """
    df = master_df.copy()
    
    # Need pace
    df['pace_min_km'] = df['elapsed_time_s'] / 60 / df['distance_km']
    valid = df[df['pace_min_km'].between(2.5, 10) & df['avg_hr'].notna()].copy()
    
    pace_p10 = valid['pace_min_km'].quantile(0.10)
    pace_median = valid['pace_min_km'].median()
    
    candidates = []
    for _, row in valid.iterrows():
        pace = row['pace_min_km']
        dist = row['distance_km']
        hr = row['avg_hr']
        
        # Must be faster than median AND at race-ish distance
        if pace >= pace_median * 0.95:
            continue
        
        is_race_dist = False
        official_dist = None
        race_type = None
        for dmin, dmax, odist, label in RACE_DISTANCES:
            if dmin <= dist <= dmax:
                is_race_dist = True
                official_dist = odist
                race_type = label
                break
        
        if not is_race_dist:
            continue
        
        # Must have elevated HR (> 85% LTHR) to be a candidate
        if lthr > 0 and hr < lthr * 0.85:
            continue
        
        candidates.append({
            'file': row['file'],
            'official_distance_km': official_dist,
            'race_type': race_type,
        })
    
    return pd.DataFrame(candidates)


def enrich_and_classify(master_path: str, overrides_path: str,
                        lthr: float, max_hr: float,
                        from_master: bool = False) -> str:
    """Main function: enrich overrides with names + smart classification.
    
    Returns path to output file.
    """
    print(f"Loading Master: {master_path}")
    master = pd.read_excel(master_path)
    print(f"  {len(master)} runs, {master['date'].min().date()} to {master['date'].max().date()}")
    
    # Pace stats for calibration
    master['_pace'] = master['elapsed_time_s'] / 60 / master['distance_km']
    valid_paces = master['_pace'][master['_pace'].between(2.5, 10)]
    pace_p10 = valid_paces.quantile(0.10)
    pace_median = valid_paces.median()
    print(f"  Pace P10={pace_p10:.2f}/km, median={pace_median:.2f}/km")
    
    master_lookup = master.set_index('file').to_dict('index')
    
    # ── Load or generate overrides ──
    if from_master:
        print(f"\nDetecting race candidates from Master...")
        candidates = detect_candidates_from_master(master, lthr, max_hr)
        print(f"  {len(candidates)} candidates found")
        
        # Build overrides DataFrame
        ov = candidates[['file', 'official_distance_km']].copy()
        ov['race_flag'] = 1
        ov['parkrun'] = 0
        ov['surface'] = None
        ov['surface_adj'] = None
        ov['temp_override'] = None
        ov['power_override_w'] = None
        ov['notes'] = ''
    else:
        print(f"\nLoading overrides: {overrides_path}")
        ov = pd.read_excel(overrides_path, dtype={'file': str})
        print(f"  {len(ov)} rows")
        
        # Fix column names if needed (onboard_athlete.py bug)
        rename_map = {}
        if 'race' in ov.columns and 'race_flag' not in ov.columns:
            rename_map['race'] = 'race_flag'
        if 'distance_km' in ov.columns and 'official_distance_km' not in ov.columns:
            rename_map['distance_km'] = 'official_distance_km'
        if 'temp_c' in ov.columns and 'temp_override' not in ov.columns:
            rename_map['temp_c'] = 'temp_override'
        if rename_map:
            ov.rename(columns=rename_map, inplace=True)
            print(f"  Fixed column names: {rename_map}")
        
        # Ensure race_flag and parkrun are integer (onboard script may write True/False)
        if 'race_flag' in ov.columns:
            ov['race_flag'] = ov['race_flag'].astype(int)
        if 'parkrun' in ov.columns:
            ov['parkrun'] = ov['parkrun'].fillna(0).astype(int)
    
    # ── Enrich with Master data ──
    print(f"\nEnriching with activity names and HR data...")
    enriched_cols = {
        'date': [], 'activity_name': [], 'actual_dist_km': [],
        'avg_hr': [], 'hr_pct_lthr': [], 'verdict': [],
        'confidence': [], 'reason': []
    }
    
    race_count = 0
    training_count = 0
    uncertain_count = 0
    
    for _, row in ov.iterrows():
        fname = row['file']
        m = master_lookup.get(fname, {})
        
        date = m.get('date')
        name = str(m.get('activity_name', '') or '')
        dist = m.get('distance_km', 0) or 0
        hr = m.get('avg_hr', 0) or 0
        pace = m.get('_pace', 0) or 0
        
        enriched_cols['date'].append(
            date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10] if date else '')
        enriched_cols['activity_name'].append(name)
        enriched_cols['actual_dist_km'].append(round(dist, 2) if dist else None)
        enriched_cols['avg_hr'].append(int(hr) if hr else None)
        enriched_cols['hr_pct_lthr'].append(
            round(hr / lthr * 100, 0) if hr and lthr else None)
        
        verdict, conf, reason = classify_run(
            name, hr, lthr, max_hr, dist, pace, pace_p10, pace_median)
        
        enriched_cols['verdict'].append(f"{verdict} ({conf})")
        enriched_cols['confidence'].append(conf)
        enriched_cols['reason'].append(reason)
        
        if verdict == 'race':
            race_count += 1
        elif verdict == 'training':
            training_count += 1
        else:
            uncertain_count += 1
    
    # Add enriched columns
    for col, values in enriched_cols.items():
        ov[col] = values
    
    # ── Apply verdicts to race_flag ──
    # High-confidence: auto-set
    # Medium/low/uncertain: flag for review
    for i, row in ov.iterrows():
        v = row['verdict']
        if 'training (high)' in v:
            ov.at[i, 'race_flag'] = 0
        elif 'race (high)' in v:
            ov.at[i, 'race_flag'] = 1
        elif 'race (low)' in v:
            ov.at[i, 'race_flag'] = 1
        elif 'training (medium)' in v:
            ov.at[i, 'race_flag'] = 0
        elif 'race (medium)' in v:
            ov.at[i, 'race_flag'] = 1
            ov.at[i, 'notes'] = f"REVIEW: {row['reason']}"
        elif 'uncertain' in v:
            ov.at[i, 'race_flag'] = 0  # default to training, flag for review
            ov.at[i, 'notes'] = f"REVIEW: {row['reason']}"
    
    # ── Detect parkruns ──
    _detect_parkruns(ov, master_lookup)
    
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
    print(f"  Training (high confidence): {training_count}")
    print(f"  Uncertain / medium:         {uncertain_count + (race_count + training_count - (ov['confidence']=='high').sum())}")
    print(f"{'─'*60}")
    print(f"  Final race_flag=1:          {final_races}")
    print(f"  Final race_flag=0:          {final_training}")
    print(f"  Rows marked REVIEW:         {review_count}")
    print(f"{'='*60}")
    
    # ── Write output ──
    output_path = overrides_path
    
    # Column order: core fields first, then enrichment
    core_cols = ['file', 'race_flag', 'parkrun', 'official_distance_km',
                 'surface', 'surface_adj', 'temp_override', 'power_override_w']
    enrich_cols = ['date', 'activity_name', 'actual_dist_km', 'avg_hr',
                   'hr_pct_lthr', 'verdict', 'reason', 'notes']
    
    # Ensure all columns exist
    for c in core_cols + enrich_cols:
        if c not in ov.columns:
            ov[c] = None
    
    out_cols = core_cols + enrich_cols
    ov_out = ov[out_cols].copy()
    
    # Write Excel
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "overrides"
    
    # Headers
    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    review_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")  # yellow
    race_fill = PatternFill(start_color="D5F5E3", end_color="D5F5E3", fill_type="solid")    # green
    training_fill = PatternFill(start_color="FADBD8", end_color="FADBD8", fill_type="solid") # red-ish
    
    for col_idx, header in enumerate(out_cols, 1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = Font(bold=True)
        cell.fill = header_fill
    
    # Data rows
    for row_idx, (_, row) in enumerate(ov_out.iterrows(), 2):
        for col_idx, col in enumerate(out_cols, 1):
            val = row[col]
            # Convert numpy types
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val) if not np.isnan(val) else None
            elif pd.isna(val):
                val = None
            ws.cell(row=row_idx, column=col_idx, value=val)
        
        # Color-code rows
        verdict = str(row.get('verdict', ''))
        notes = str(row.get('notes', '') or '')
        if 'REVIEW' in notes:
            for col_idx in range(1, len(out_cols) + 1):
                ws.cell(row=row_idx, column=col_idx).fill = review_fill
        elif 'race' in verdict:
            for col_idx in range(1, len(out_cols) + 1):
                ws.cell(row=row_idx, column=col_idx).fill = race_fill
        elif 'training' in verdict:
            for col_idx in range(1, len(out_cols) + 1):
                ws.cell(row=row_idx, column=col_idx).fill = training_fill
    
    # Auto-width
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 3, 60)
    
    # Freeze header row
    ws.freeze_panes = 'A2'
    
    # Auto-filter
    ws.auto_filter.ref = ws.dimensions
    
    wb.save(output_path)
    print(f"\nSaved: {output_path}")
    print(f"  Green = race, Red = training, Yellow = REVIEW needed")
    
    return output_path


def _is_parkrun(name, date_val, dist_km):
    """Detect parkrun from name, day-of-week, and distance.
    Returns True if:
      - Activity name contains 'parkrun' or 'park run', OR
      - It's a Saturday AND distance is 4.8-5.5 km (typical parkrun GPS range)
    """
    # Name match — strongest signal
    if name and re.search(r'parkrun|park run', name, re.I):
        return True
    
    # Saturday + ~5km — very likely parkrun
    if date_val and dist_km:
        try:
            from datetime import datetime
            if isinstance(date_val, str):
                dt = datetime.strptime(str(date_val)[:10], '%Y-%m-%d')
            elif hasattr(date_val, 'weekday'):
                dt = date_val
            else:
                return False
            if dt.weekday() == 5 and 4.8 <= float(dist_km) <= 5.5:
                return True
        except (ValueError, TypeError):
            pass
    
    return False


def _detect_parkruns(ov, master_lookup):
    """Set parkrun=1 for all detected parkruns in the overrides DataFrame.
    Uses activity name, date (Saturday), and distance (~5km).
    Only sets — never clears an existing parkrun=1."""
    
    if 'parkrun' not in ov.columns:
        ov['parkrun'] = 0
    
    count = 0
    for i, row in ov.iterrows():
        if row.get('parkrun', 0) == 1:
            continue
        
        fname = str(row.get('file', ''))
        m = master_lookup.get(fname, {}) if isinstance(master_lookup, dict) else {}
        
        # Get activity name from enriched column or Master
        name = str(row.get('activity_name', '') or m.get('activity_name', '') or '')
        date_val = row.get('date', '') or m.get('date', '')
        dist = row.get('actual_dist_km', None) or m.get('distance_km', None)
        
        if _is_parkrun(name, date_val, dist):
            ov.at[i, 'parkrun'] = 1
            # Ensure race_flag is also 1
            if 'race_flag' in ov.columns:
                ov.at[i, 'race_flag'] = 1
            count += 1
    
    if count > 0:
        print(f"  Parkrun detection: marked {count} parkrun(s).")


def _backfill_parkruns(ov, master_path, overrides_path):
    """Lightweight parkrun detection — runs even when full classification is skipped.
    Looks up activity names/dates/distances from Master and sets parkrun=1 where detected.
    Only touches rows where parkrun is 0 or NaN (never clears an existing 1)."""
    
    if 'parkrun' not in ov.columns:
        ov['parkrun'] = 0
    
    # Load Master for activity names + dates + distances
    try:
        master = pd.read_excel(master_path)
    except Exception as e:
        print(f"  Could not load Master for parkrun backfill: {e}")
        return
    
    # Build lookup: filename -> {activity_name, date, distance_km}
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
        
        if _is_parkrun(name, date_val, dist):
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
    p.add_argument("--master", required=True, help="Path to Master_FULL_post.xlsx")
    p.add_argument("--overrides", required=True, help="Path to activity_overrides.xlsx")
    p.add_argument("--athlete-yml", help="Path to athlete.yml (for LTHR/max HR)")
    p.add_argument("--lthr", type=int, help="Lactate threshold HR (overrides athlete.yml)")
    p.add_argument("--max-hr", type=int, help="Max HR (overrides athlete.yml)")
    p.add_argument("--from-master", action="store_true",
                   help="Generate candidates from Master instead of reading existing overrides")
    p.add_argument("--skip-if-classified", action="store_true",
                   help="Skip if overrides already has verdict column or any race_flag values set (preserves athlete edits)")
    args = p.parse_args()
    
    # Check skip condition — skip if already classified OR if athlete has manually edited race flags
    if args.skip_if_classified and Path(args.overrides).exists():
        try:
            existing = pd.read_excel(args.overrides)
            skip = False
            if 'verdict' in existing.columns:
                print(f"Overrides already classified (has 'verdict' column) — skipping full classification.")
                print(f"  To re-classify, remove --skip-if-classified or delete the verdict column.")
                skip = True
            elif 'race_flag' in existing.columns and existing['race_flag'].notna().any():
                n_races = (existing['race_flag'] == 1).sum()
                print(f"Overrides already has {n_races} race flags set — skipping to preserve athlete edits.")
                print(f"  To re-classify, remove --skip-if-classified.")
                skip = True
            
            if skip:
                # Still do a lightweight parkrun backfill from Master activity names
                _backfill_parkruns(existing, args.master, args.overrides)
                return
        except Exception:
            pass
    
    # Get HR thresholds
    lthr = args.lthr
    max_hr = args.max_hr
    
    if args.athlete_yml and (lthr is None or max_hr is None):
        cfg = load_athlete_config(args.athlete_yml)
        if lthr is None:
            lthr = cfg.get("lthr", 160)
        if max_hr is None:
            max_hr = cfg.get("max_hr", 185)
    
    if lthr is None:
        lthr = 160
        print(f"Warning: using default LTHR={lthr}")
    if max_hr is None:
        max_hr = 185
        print(f"Warning: using default max_hr={max_hr}")
    
    print(f"HR thresholds: LTHR={lthr}, max HR={max_hr}")
    
    enrich_and_classify(args.master, args.overrides, lthr, max_hr,
                        from_master=args.from_master)


if __name__ == "__main__":
    main()
