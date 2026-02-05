#!/usr/bin/env python3
"""
verify_weather_migration.py - Check that all indoor runs from weather_overrides.csv
have temp_override entries in activity_overrides.xlsx.

Run from DataPipeline folder:
    python verify_weather_migration.py
"""

import pandas as pd
import os
import sys

WEATHER_CSV = "weather_overrides.csv"
OVERRIDE_XLSX = "activity_overrides.xlsx"

def main():
    if not os.path.exists(WEATHER_CSV):
        print(f"No {WEATHER_CSV} found — nothing to verify.")
        return 0

    if not os.path.exists(OVERRIDE_XLSX):
        print(f"ERROR: {OVERRIDE_XLSX} not found.")
        return 1

    # Read weather overrides
    wov = pd.read_csv(WEATHER_CSV, dtype=str)
    print(f"\n{WEATHER_CSV}: {len(wov)} rows")
    print(f"  Columns: {list(wov.columns)}")
    print()
    for _, row in wov.iterrows():
        parts = [f"  {col}={row[col]}" for col in wov.columns if pd.notna(row[col])]
        print("  " + ", ".join(parts))
    print()

    weather_files = set(str(x).strip() for x in wov['file'].dropna() if str(x).strip())

    # Read activity overrides
    odf = pd.read_excel(OVERRIDE_XLSX, dtype={'file': str})
    print(f"{OVERRIDE_XLSX}: {len(odf)} rows")

    if 'temp_override' not in odf.columns:
        print(f"  WARNING: No 'temp_override' column — none of the indoor runs are covered!")
        missing = weather_files
    else:
        # Find which have temp_override set
        indoor_overrides = odf[odf['temp_override'].notna()]
        override_files = set(str(x).strip() for x in indoor_overrides['file'].dropna())

        covered = weather_files & override_files
        missing = weather_files - override_files

        print(f"  Rows with temp_override: {len(indoor_overrides)}")
        print()

        if covered:
            print(f"COVERED ({len(covered)} of {len(weather_files)}):")
            for f in sorted(covered):
                temp = indoor_overrides[indoor_overrides['file'].str.strip() == f]['temp_override'].iloc[0]
                print(f"  ✓ {f}  (temp_override={temp})")
            print()

    if missing:
        print(f"MISSING ({len(missing)} — need to add temp_override to {OVERRIDE_XLSX}):")
        for f in sorted(missing):
            row = wov[wov['file'].str.strip() == f].iloc[0]
            temp = row.get('override_temp_c', '?')
            print(f"  ✗ {f}  (weather_overrides temp={temp})")
        print()
        print("ACTION: Add these to activity_overrides.xlsx with the temp_override values shown above.")
        print("You can use:  python add_override.py <file> TEMP=<value>")
        return 1
    else:
        print("ALL COVERED — safe to archive weather_overrides.csv")
        return 0


if __name__ == "__main__":
    sys.exit(main())
