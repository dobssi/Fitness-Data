#!/usr/bin/env python3
"""
zip_add_fits.py - Safely add FIT files to TotalHistory.zip with deduplication.

Compares by start timestamp (from filename) against existing zip contents
to avoid adding duplicate runs under different filenames.

Usage:
    python zip_add_fits.py                          (add all from FIT_downloads/)
    python zip_add_fits.py --fit-dir FIT_downloads  (explicit dir)
    python zip_add_fits.py --dry-run                (preview only)
"""

import os
import sys
import argparse
import zipfile
from pathlib import Path
from datetime import datetime


TOTAL_HISTORY_ZIP = r"TotalHistory.zip"
DEFAULT_FIT_DIR = r"FIT_downloads"


def parse_timestamp_from_filename(filename: str) -> str:
    """
    Extract a normalised timestamp string from a FIT filename.
    
    Handles: "2026-02-04_13-56-27.FIT" -> "2026-02-04T13:56:27"
    Returns empty string if filename doesn't match date pattern.
    """
    stem = Path(filename).stem
    # Pattern: YYYY-MM-DD_HH-MM-SS
    if len(stem) >= 19 and stem[4] == '-' and stem[7] == '-' and stem[10] == '_':
        try:
            dt = datetime.strptime(stem[:19], "%Y-%m-%d_%H-%M-%S")
            return dt.isoformat()
        except ValueError:
            pass
    return ""


def get_zip_timestamps(zip_path: str, master_path: str = None) -> set:
    """
    Get set of start timestamps for all runs already in the zip.
    
    Uses the Master XLSX to map old-named files to their start times.
    Falls back to filename parsing for date-named files.
    """
    timestamps = set()
    
    # First: build filename->date lookup from Master if available
    file_to_date = {}
    if master_path and os.path.exists(master_path):
        try:
            import pandas as pd
            df = pd.read_excel(master_path, engine='openpyxl')
            for _, row in df.iterrows():
                fname = str(row.get('file', '')).strip().lower()
                dt = row.get('date')
                if fname and not pd.isna(dt):
                    ts = pd.Timestamp(dt).isoformat()
                    file_to_date[fname] = ts
        except Exception as e:
            print(f"  Warning: Could not read Master for dedup: {e}")
    
    # Now scan the zip
    if not os.path.exists(zip_path):
        return timestamps
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if not name.lower().endswith('.fit'):
                continue
            
            basename = os.path.basename(name).lower()
            
            # Try Master lookup first
            if basename in file_to_date:
                timestamps.add(file_to_date[basename])
                continue
            
            # Try parsing from filename
            ts = parse_timestamp_from_filename(name)
            if ts:
                timestamps.add(ts)
    
    return timestamps


def main():
    parser = argparse.ArgumentParser(description="Add FIT files to TotalHistory.zip with dedup")
    parser.add_argument("--fit-dir", default=DEFAULT_FIT_DIR, help="Directory with new FIT files")
    parser.add_argument("--zip", default=TOTAL_HISTORY_ZIP, help="Target zip file")
    parser.add_argument("--master", default="Master_FULL_GPSQ_ID_post.xlsx", help="Master XLSX for timestamp lookup")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()
    
    fit_dir = Path(args.fit_dir)
    if not fit_dir.exists():
        print(f"No FIT_downloads directory found — nothing to add.")
        return 0
    
    new_fits = list(fit_dir.glob("*.FIT")) + list(fit_dir.glob("*.fit"))
    if not new_fits:
        print("No new FIT files to add.")
        return 0
    
    print(f"Found {len(new_fits)} FIT file(s) in {args.fit_dir}")
    
    # Get existing timestamps from zip
    print(f"Checking existing runs in {args.zip}...")
    existing_timestamps = get_zip_timestamps(args.zip, args.master)
    print(f"  {len(existing_timestamps)} existing run timestamps")
    
    # Also check existing filenames (exact match)
    existing_names = set()
    if os.path.exists(args.zip):
        with zipfile.ZipFile(args.zip, 'r') as zf:
            existing_names = {os.path.basename(n).lower() for n in zf.namelist()}
    
    # Determine which files to add
    to_add = []
    skipped_dup_name = 0
    skipped_dup_time = 0
    
    for fit_path in sorted(new_fits):
        basename = fit_path.name
        
        # Skip if exact filename already in zip
        if basename.lower() in existing_names:
            skipped_dup_name += 1
            continue
        
        # Skip if same timestamp already in zip (different filename, same run)
        ts = parse_timestamp_from_filename(basename)
        if ts and ts in existing_timestamps:
            skipped_dup_time += 1
            continue
        
        to_add.append(fit_path)
    
    # Report
    if skipped_dup_name:
        print(f"  Skipped {skipped_dup_name} (filename already in zip)")
    if skipped_dup_time:
        print(f"  Skipped {skipped_dup_time} (same start time as existing run)")
    
    if not to_add:
        print("No new runs to add.")
        return 0
    
    print(f"\n  Adding {len(to_add)} new run(s):")
    for f in to_add:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name} ({size_kb:.0f} KB)")
    
    if args.dry_run:
        print("\n  [DRY RUN — no changes made]")
        return 0
    
    # Add to zip
    with zipfile.ZipFile(args.zip, 'a', zipfile.ZIP_DEFLATED) as zf:
        for fit_path in to_add:
            zf.write(fit_path, fit_path.name)
    
    print(f"\n  [OK] Added {len(to_add)} FIT file(s) to {args.zip}")
    
    # Signal to Daily_Update.bat that new runs were added
    try:
        with open('_new_runs_added.tmp', 'w') as f:
            f.write(str(len(to_add)))
    except Exception:
        pass
    
    # Verify
    with zipfile.ZipFile(args.zip, 'r') as zf:
        total = sum(1 for n in zf.namelist() if n.lower().endswith('.fit'))
        print(f"  Zip now contains {total} FIT files")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
