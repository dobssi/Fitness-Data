#!/usr/bin/env python3
"""
merge_user_data.py — Merge athlete-uploaded data into pipeline data stores.

Athletes drop files into their Dropbox user_data/ folder:
  user_data/fits/           ← FIT files (Zwift, manual exports, etc.)
  user_data/activities.csv  ← updated Strava export (re-upload anytime)
  user_data/weight.csv      ← date,weight_kg CSV for weight history

This script downloads new files and merges them into the canonical stores:
  - FIT files → data/fits.zip (dedup by timestamp from filename)
  - activities.csv → data/activities.csv (replace if newer)
  - weight.csv → athlete_data.csv (append new dates, dedup)

Usage:
  python ci/merge_user_data.py --athlete-dir athletes/A002

Environment:
  DROPBOX_TOKEN / DROPBOX_REFRESH_TOKEN + APP_KEY + APP_SECRET
  DB_BASE — Dropbox base path (e.g. /Running and Cycling/DataPipeline/athletes/A002)
"""

import argparse
import csv
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

# Add parent dir to path so we can import from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dropbox_sync import get_token, dropbox_list_folder, dropbox_download


# ─── FIT merge ──────────────────────────────────────────────────────────────

def parse_timestamp_from_filename(filename: str) -> str | None:
    """Extract normalised timestamp from FIT filename.
    
    Handles patterns like:
      2026-02-04_13-56-27.FIT → 2026-02-04T13:56:27
      2026-02-04_13-56-27.fit → 2026-02-04T13:56:27
      anything.fit → None (if no parseable timestamp)
    """
    import re
    stem = Path(filename).stem
    # Try: YYYY-MM-DD_HH-MM-SS or YYYY-MM-DD_HHMMSS
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})[_T](\d{2})-?(\d{2})-?(\d{2})', stem)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:{m.group(6)}"
    return None


def get_existing_timestamps(zip_path: str) -> set:
    """Get set of normalised timestamps already in the fits.zip."""
    timestamps = set()
    if not os.path.exists(zip_path):
        return timestamps
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            for name in z.namelist():
                ts = parse_timestamp_from_filename(name)
                if ts:
                    timestamps.add(ts)
    except zipfile.BadZipFile:
        print(f"  ⚠ Bad zip file: {zip_path}")
    return timestamps


def merge_fits(user_fits_dir: str, fits_zip_path: str, dry_run: bool = False) -> int:
    """Add new FIT files from user_data/fits/ into data/fits.zip.
    
    Deduplicates by timestamp extracted from filename.
    Returns count of files added.
    """
    if not os.path.isdir(user_fits_dir):
        return 0
    
    # Find FIT files in user_data/fits/
    fit_files = []
    for f in Path(user_fits_dir).iterdir():
        if f.suffix.lower() in ('.fit', '.fit.gz'):
            fit_files.append(f)
    
    if not fit_files:
        return 0
    
    print(f"  Found {len(fit_files)} FIT file(s) in user_data/fits/")
    
    # Get existing timestamps from zip
    existing = get_existing_timestamps(fits_zip_path)
    print(f"  Existing fits.zip has {len(existing)} timestamped entries")
    
    # Filter to new files only
    new_files = []
    for f in fit_files:
        ts = parse_timestamp_from_filename(f.name)
        if ts and ts in existing:
            continue  # Already in zip
        new_files.append(f)
    
    if not new_files:
        print(f"  No new FIT files to add (all {len(fit_files)} already in zip)")
        return 0
    
    print(f"  Adding {len(new_files)} new FIT file(s) to {fits_zip_path}")
    
    if dry_run:
        for f in new_files:
            print(f"    [dry-run] Would add: {f.name}")
        return len(new_files)
    
    # Ensure zip exists
    if not os.path.exists(fits_zip_path):
        Path(fits_zip_path).parent.mkdir(parents=True, exist_ok=True)
        # Create empty zip
        with zipfile.ZipFile(fits_zip_path, "w") as z:
            pass
    
    # Append to zip
    with zipfile.ZipFile(fits_zip_path, "a", compression=zipfile.ZIP_DEFLATED) as z:
        for f in new_files:
            z.write(str(f), f.name)
            print(f"    + {f.name}")
    
    return len(new_files)


# ─── Activities.csv merge ───────────────────────────────────────────────────

def merge_activities_csv(user_csv: str, pipeline_csv: str, dry_run: bool = False) -> bool:
    """Replace pipeline's activities.csv if user uploaded a newer one.
    
    Returns True if replaced.
    """
    if not os.path.exists(user_csv):
        return False
    
    user_size = os.path.getsize(user_csv)
    pipeline_size = os.path.getsize(pipeline_csv) if os.path.exists(pipeline_csv) else 0
    
    if user_size == 0:
        print(f"  User activities.csv is empty — skipping")
        return False
    
    # Count rows for comparison
    def count_rows(path):
        try:
            with open(path, encoding='utf-8', errors='replace') as f:
                return sum(1 for _ in f) - 1  # Subtract header
        except Exception:
            return 0
    
    user_rows = count_rows(user_csv)
    pipeline_rows = count_rows(pipeline_csv) if os.path.exists(pipeline_csv) else 0
    
    print(f"  User activities.csv: {user_rows} rows ({user_size:,} bytes)")
    print(f"  Pipeline activities.csv: {pipeline_rows} rows ({pipeline_size:,} bytes)")
    
    if user_rows <= pipeline_rows and user_size <= pipeline_size:
        print(f"  User file is not larger — skipping (upload a newer export to replace)")
        return False
    
    if dry_run:
        print(f"  [dry-run] Would replace activities.csv ({pipeline_rows} → {user_rows} rows)")
        return True
    
    # Replace
    import shutil
    Path(pipeline_csv).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(user_csv, pipeline_csv)
    print(f"  ✓ Replaced activities.csv ({pipeline_rows} → {user_rows} rows)")
    return True


# ─── Weight merge ───────────────────────────────────────────────────────────

def merge_weight(user_weight_csv: str, athlete_data_csv: str, dry_run: bool = False) -> int:
    """Append new weight entries from user_data/weight.csv into athlete_data.csv.
    
    Expected format: date,weight_kg (with or without header).
    Deduplicates by date.
    Returns count of new entries added.
    """
    if not os.path.exists(user_weight_csv):
        return 0
    
    # Parse user weight data
    user_weights = {}
    with open(user_weight_csv, encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            date_str = row[0].strip()
            weight_str = row[1].strip()
            # Skip header
            if date_str.lower() in ('date', 'date_local', 'day'):
                continue
            try:
                # Validate date
                datetime.strptime(date_str, "%Y-%m-%d")
                weight = float(weight_str)
                if 30 <= weight <= 200:  # Sanity check
                    user_weights[date_str] = weight
            except (ValueError, TypeError):
                continue
    
    if not user_weights:
        print(f"  No valid weight entries in user_data/weight.csv")
        return 0
    
    print(f"  User weight.csv: {len(user_weights)} valid entries")
    
    # Load existing athlete_data.csv
    existing_dates = set()
    existing_rows = []
    header = None
    weight_col = None
    
    if os.path.exists(athlete_data_csv):
        with open(athlete_data_csv, encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    # Handle comment headers (# prefix)
                    if row and row[0].startswith('#'):
                        header = [c.lstrip('#').strip() for c in row]
                    else:
                        header = row
                    # Find weight column
                    for j, col in enumerate(header):
                        if col.lower() in ('weight_kg', 'weight'):
                            weight_col = j
                            break
                    existing_rows.append(row)  # Keep original header row
                    continue
                if row and row[0].startswith('#'):
                    existing_rows.append(row)
                    continue
                existing_rows.append(row)
                if row:
                    existing_dates.add(row[0].strip())
    
    # If no athlete_data.csv exists, create with minimal header
    if header is None:
        header = ['date_local', 'weight_kg', 'non_running_tss']
        weight_col = 1
        existing_rows = [header]
    
    if weight_col is None:
        print(f"  ⚠ No weight_kg column found in {athlete_data_csv} — cannot merge")
        return 0
    
    # Find new dates
    new_entries = {d: w for d, w in user_weights.items() if d not in existing_dates}
    
    if not new_entries:
        print(f"  No new weight dates to add (all {len(user_weights)} already present)")
        return 0
    
    print(f"  Adding {len(new_entries)} new weight date(s)")
    
    if dry_run:
        for d, w in sorted(new_entries.items())[:5]:
            print(f"    [dry-run] {d}: {w} kg")
        if len(new_entries) > 5:
            print(f"    ... and {len(new_entries) - 5} more")
        return len(new_entries)
    
    # Append new rows
    for d, w in sorted(new_entries.items()):
        new_row = [''] * len(header)
        new_row[0] = d
        new_row[weight_col] = str(w)
        existing_rows.append(new_row)
    
    # Sort by date (skip header)
    data_rows = existing_rows[1:]
    data_rows.sort(key=lambda r: r[0] if r else '')
    
    # Write back
    with open(athlete_data_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(existing_rows[0])  # Header
        for row in data_rows:
            writer.writerow(row)
    
    print(f"  ✓ Added {len(new_entries)} weight entries to {athlete_data_csv}")
    return len(new_entries)


# ─── Dropbox download ──────────────────────────────────────────────────────

def download_user_data(athlete_dir: str, db_base: str, token: str) -> dict:
    """Download user_data/ files from Dropbox.
    
    Returns dict with counts: {fits: N, activities: bool, weight: bool}
    """
    remote_base = f"{db_base}/user_data"
    local_base = os.path.join(athlete_dir, "user_data")
    
    result = {"fits": 0, "activities": False, "weight": False, "training_plan": False}

    # ── Download individual files ──
    for filename in ["activities.csv", "weight.csv"]:
        remote_path = f"{remote_base}/{filename}"
        local_path = os.path.join(local_base, filename)
        os.makedirs(local_base, exist_ok=True)
        try:
            dropbox_download(remote_path, local_path, token)
            if filename == "activities.csv":
                result["activities"] = True
            elif filename == "weight.csv":
                result["weight"] = True
        except Exception:
            pass  # File doesn't exist — that's fine
    
    # ── Download PDF training plans from user_data/ ──
    try:
        user_files = dropbox_list_folder(remote_base, token)
        if user_files:
            for fname, size in user_files.items():
                if fname.lower().endswith(('.pdf', '.txt')):
                    local_pdf = os.path.join(local_base, fname)
                    os.makedirs(local_base, exist_ok=True)
                    try:
                        dropbox_download(f"{remote_base}/{fname}", local_pdf, token)
                        result["training_plan"] = True
                        print(f"  Downloaded training plan: {fname}")
                    except Exception as e:
                        print(f"  Warning: Failed to download {fname}: {e}")
    except Exception:
        pass  # No files in user_data/ root — that's fine

    # ── Download FIT files from user_data/fits/ ──
    remote_fits = f"{remote_base}/fits"
    local_fits = os.path.join(local_base, "fits")
    
    fit_files = dropbox_list_folder(remote_fits, token)
    if fit_files:
        os.makedirs(local_fits, exist_ok=True)
        for filename, size in fit_files.items():
            if not filename.lower().endswith(('.fit', '.fit.gz')):
                continue
            local_path = os.path.join(local_fits, filename)
            # Skip if already downloaded (same size)
            if os.path.exists(local_path) and os.path.getsize(local_path) == size:
                continue
            try:
                dropbox_download(f"{remote_fits}/{filename}", local_path, token)
                result["fits"] += 1
            except Exception as e:
                print(f"  ⚠ Failed to download {filename}: {e}")
    
    return result


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Merge user-uploaded data into pipeline stores")
    parser.add_argument("--athlete-dir", required=True, help="Athlete directory (e.g. athletes/A002)")
    parser.add_argument("--dropbox-base", default=None, help="Dropbox base path (default: from DB_BASE env)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--skip-download", action="store_true", help="Skip Dropbox download (use local user_data/)")
    args = parser.parse_args()
    
    athlete_dir = args.athlete_dir
    db_base = args.dropbox_base or os.environ.get("DB_BASE", "")
    
    print(f"\n{'─' * 60}")
    print(f"  Merging user data for {athlete_dir}")
    print(f"{'─' * 60}\n")
    
    # Paths
    user_data_dir = os.path.join(athlete_dir, "user_data")
    user_fits_dir = os.path.join(user_data_dir, "fits")
    user_activities = os.path.join(user_data_dir, "activities.csv")
    user_weight = os.path.join(user_data_dir, "weight.csv")
    fits_zip = os.path.join(athlete_dir, "data", "fits.zip")
    pipeline_activities = os.path.join(athlete_dir, "data", "activities.csv")
    athlete_data = os.path.join(athlete_dir, "athlete_data.csv")
    
    # Download from Dropbox
    if not args.skip_download and db_base:
        print("Downloading user_data/ from Dropbox...")
        try:
            token = get_token()
            dl = download_user_data(athlete_dir, db_base, token)
            print(f"  Downloaded: {dl['fits']} FIT(s), "
                  f"activities={'yes' if dl['activities'] else 'no'}, "
                  f"weight={'yes' if dl['weight'] else 'no'}, "
                  f"training_plan={'yes' if dl.get('training_plan') else 'no'}")
        except Exception as e:
            print(f"  ⚠ Dropbox download failed: {e}")
            print(f"  Continuing with local user_data/ if present...")
    
    if not os.path.exists(user_data_dir):
        print(f"  No user_data/ folder found — nothing to merge")
        return
    
    changes = False
    
    # 1. Merge FITs
    print("\n[1/3] FIT files...")
    n_fits = merge_fits(user_fits_dir, fits_zip, dry_run=args.dry_run)
    if n_fits > 0:
        changes = True
    
    # 2. Merge activities.csv
    print("\n[2/3] Activities CSV...")
    if merge_activities_csv(user_activities, pipeline_activities, dry_run=args.dry_run):
        changes = True
    
    # 3. Merge weight
    print("\n[3/3] Weight data...")
    n_weight = merge_weight(user_weight, athlete_data, dry_run=args.dry_run)
    if n_weight > 0:
        changes = True
    
    print(f"\n{'─' * 60}")
    if changes:
        print(f"  ✓ User data merged into {athlete_dir}")
    else:
        print(f"  No new user data to merge")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
