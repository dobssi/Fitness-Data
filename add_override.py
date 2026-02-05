#!/usr/bin/env python3
"""
add_override.py - Add overrides by date, date #N, or filename.

Usage:
    python add_override.py "2026-02-04" SNOW,1.05
    python add_override.py "2026-02-04 #1" SNOW,1.05
    python add_override.py "2026-02-04 #2" INDOOR_TRACK,20
    python add_override.py "2026-02-04_10-30-00.FIT" "RACE,PARKRUN,5.0"
    python add_override.py                              (interactive mode)

Override values (comma-separated):
    RACE          - Mark as race (race_flag=1)
    PARKRUN       - Mark as parkrun (implies RACE)
    TRACK         - Track surface
    INDOOR_TRACK  - Indoor track surface
    TRAIL         - Trail surface
    SNOW          - Snow conditions
    HEAVY_SNOW    - Heavy snow conditions
    <0.8-1.2>     - Surface adjustment (e.g., 1.05)
    ADJ=<val>     - Explicit surface adjustment
    TEMP=<val>    - Temperature override (°C)
    <number>      - Official distance in km if >1.5 (e.g., 5.0, 42.195)

Examples:
    python add_override.py "2026-02-04" "SNOW,1.05"
    python add_override.py "2026-02-04 #2" "INDOOR_TRACK,TEMP=18"
    python add_override.py "2025-12-25" "RACE,PARKRUN,5.0"
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
OVERRIDE_FILE = r"C:\Users\paul.collyer\Dropbox\Running and Cycling\DataPipeline\activity_overrides.xlsx"

VALID_SURFACES = {'TRACK', 'INDOOR_TRACK', 'TRAIL', 'SNOW', 'HEAVY_SNOW'}
FLAG_KEYWORDS = {'RACE', 'PARKRUN'} | VALID_SURFACES


def parse_flags(flags_str: str) -> dict:
    """Parse comma-separated flags string into override fields."""
    result = {
        'race_flag': 0,
        'parkrun': 0,
        'surface': '',
        'surface_adj': None,
        'official_distance_km': None,
        'temp_override': None,
    }
    
    if not flags_str:
        return result
    
    for flag in [f.strip() for f in flags_str.split(',')]:
        upper = flag.upper()
        
        if upper == 'RACE':
            result['race_flag'] = 1
        elif upper == 'PARKRUN':
            result['race_flag'] = 1
            result['parkrun'] = 1
            if result['official_distance_km'] is None:
                result['official_distance_km'] = 5.0
        elif upper in VALID_SURFACES:
            result['surface'] = upper
        elif upper.startswith('ADJ=') or upper.startswith('SADJ='):
            try:
                result['surface_adj'] = float(flag.split('=')[1])
            except (ValueError, IndexError):
                print(f"WARNING: Invalid surface_adj '{flag}' - ignoring")
        elif upper.startswith('TEMP='):
            try:
                result['temp_override'] = float(flag.split('=')[1])
            except (ValueError, IndexError):
                print(f"WARNING: Invalid temp_override '{flag}' - ignoring")
        else:
            try:
                val = float(flag)
                # Distinguish: 0.8-1.2 = surface_adj, >1.5 = distance
                if 0.5 < val < 1.5:
                    result['surface_adj'] = val
                elif val > 1.5:
                    result['official_distance_km'] = val
            except ValueError:
                print(f"WARNING: Unknown flag '{flag}' - ignoring")
    
    return result


def add_override(file_or_date: str, flags: dict):
    """Add or update an override entry."""
    if not os.path.exists(OVERRIDE_FILE):
        df = pd.DataFrame(columns=['file', 'race_flag', 'parkrun', 'official_distance_km',
                                    'surface', 'surface_adj', 'temp_override', 'notes'])
    else:
        df = pd.read_excel(OVERRIDE_FILE)
    
    # Remove existing entry for this identifier
    df = df[df['file'].astype(str) != file_or_date]
    
    new_row = pd.DataFrame([{
        'file': file_or_date,
        'race_flag': flags['race_flag'],
        'parkrun': flags['parkrun'],
        'official_distance_km': flags['official_distance_km'],
        'surface': flags['surface'],
        'surface_adj': flags.get('surface_adj'),
        'temp_override': flags.get('temp_override'),
        'notes': f"Added via add_override.py on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    }])
    
    if len(df) == 0:
        df = new_row
    else:
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(OVERRIDE_FILE, index=False)
    
    # Summary
    parts = [file_or_date]
    if flags['race_flag']: parts.append('RACE')
    if flags['parkrun']: parts.append('PARKRUN')
    if flags['surface']: parts.append(f"surface={flags['surface']}")
    if flags.get('surface_adj') is not None: parts.append(f"surface_adj={flags['surface_adj']}")
    if flags.get('temp_override') is not None: parts.append(f"temp={flags['temp_override']}°C")
    if flags['official_distance_km']: parts.append(f"distance={flags['official_distance_km']}km")
    print(f"  Override added: {', '.join(parts)}")


def show_overrides():
    """Show current overrides."""
    if not os.path.exists(OVERRIDE_FILE):
        print("No override file found.")
        return
    
    df = pd.read_excel(OVERRIDE_FILE)
    if len(df) == 0:
        print("Override file is empty.")
        return
    
    print(f"\n{len(df)} overrides in {OVERRIDE_FILE}:\n")
    for _, row in df.iterrows():
        parts = [str(row['file'])]
        if row.get('race_flag', 0) == 1: parts.append('RACE')
        if row.get('parkrun', 0) == 1: parts.append('PARKRUN')
        s = str(row.get('surface', '')).strip()
        if s and s != 'nan': parts.append(f"surface={s}")
        sa = row.get('surface_adj')
        if pd.notna(sa): parts.append(f"adj={sa}")
        to = row.get('temp_override')
        if pd.notna(to): parts.append(f"temp={to}°C")
        d = row.get('official_distance_km')
        if pd.notna(d): parts.append(f"dist={d}km")
        print(f"  {', '.join(parts)}")


def interactive_mode():
    """Interactive override entry."""
    print("\nAdd overrides (empty to finish):\n")
    
    while True:
        file_or_date = input("File or date (e.g. 2026-02-04 or 2026-02-04 #1): ").strip()
        if not file_or_date:
            break
        
        flags_str = input("  Flags (e.g. SNOW,1.05 or RACE,PARKRUN,5.0): ").strip()
        if not flags_str:
            print("  Skipped (no flags)")
            continue
        
        flags = parse_flags(flags_str)
        add_override(file_or_date, flags)
        print()


def main():
    print("=" * 60)
    print("Add Override")
    print("=" * 60)
    
    if len(sys.argv) == 1:
        # No args: show current + interactive
        show_overrides()
        interactive_mode()
    elif len(sys.argv) == 2 and sys.argv[1].lower() in ('--list', '-l', 'list', 'show'):
        show_overrides()
    elif len(sys.argv) >= 3:
        file_or_date = sys.argv[1]
        flags_str = sys.argv[2]
        flags = parse_flags(flags_str)
        add_override(file_or_date, flags)
    else:
        print("Usage: python add_override.py <file_or_date> <flags>")
        print("       python add_override.py                  (interactive)")
        print("       python add_override.py --list           (show current)")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
