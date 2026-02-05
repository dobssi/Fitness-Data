#!/usr/bin/env python3
"""
rename_fit_files.py - Standardise FIT file names to YYYY-MM-DD_HH-MM-SS.FIT

Uses the Master XLSX to map old filenames -> local start times, then renames
files in TotalHistory (folder + zip) and updates all reference files.

This is a one-time migration utility. After running, all FIT files use the
same naming convention as fetch_fit_files.py (intervals.icu downloads).

What gets renamed:
    - FIT files in TotalHistory folder (and subfolders)
    - FIT files in TotalHistory.zip
    - 'file' column in the Master XLSX
    - 'file' column in activity_overrides.xlsx
    - 'file' column in pending_activities.csv
    - Cache files in persec_cache_dir (*.npz named by FIT stem)

What does NOT change:
    - The FIT file contents (unchanged)
    - Custom-named files (MERGED_*, etc.) are renamed like everything else

Usage:
    python rename_fit_files.py --master Master.xlsx --dry-run    (preview only)
    python rename_fit_files.py --master Master.xlsx              (do it)
    python rename_fit_files.py --master Master.xlsx --zip-only   (only rebuild zip)
"""

import os
import sys
import shutil
import hashlib
import argparse
import zipfile
from pathlib import Path
from datetime import datetime

import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================
TOTAL_HISTORY_FOLDER = r"C:\Users\paul.collyer\Dropbox\Running and Cycling\DataPipeline\TotalHistory"
TOTAL_HISTORY_ZIP = r"C:\Users\paul.collyer\Dropbox\Running and Cycling\DataPipeline\TotalHistory.zip"
OVERRIDE_FILE = r"C:\Users\paul.collyer\Dropbox\Running and Cycling\DataPipeline\activity_overrides.xlsx"
PENDING_ACTIVITIES_CSV = r"C:\Users\paul.collyer\Dropbox\Running and Cycling\DataPipeline\pending_activities.csv"
PERSEC_CACHE_DIR = r"C:\Users\paul.collyer\Dropbox\Running and Cycling\DataPipeline\persec_cache_FULL"
BACKUP_SUFFIX = ".pre_rename_backup"


def build_rename_map(master_path: str) -> dict:
    """
    Build old_filename -> new_filename mapping from Master.
    
    Uses the 'file' and 'date' columns. Date is local time (Europe/Stockholm).
    New name: YYYY-MM-DD_HH-MM-SS.FIT
    
    Returns dict of {old_name_lower: (old_name_original, new_name)}
    Only includes files that actually need renaming.
    """
    df = pd.read_excel(master_path, engine='openpyxl')
    df['date'] = pd.to_datetime(df['date'])
    
    rename_map = {}
    already_correct = 0
    skipped_no_date = 0
    
    for _, row in df.iterrows():
        old_name = str(row.get('file', '')).strip()
        dt = row.get('date')
        
        if not old_name or old_name == 'nan' or pd.isna(dt):
            skipped_no_date += 1
            continue
        
        new_name = dt.strftime('%Y-%m-%d_%H-%M-%S') + '.FIT'
        
        if old_name.lower() == new_name.lower():
            already_correct += 1
            continue
        
        # Collision check: another file already maps to this new_name
        existing = [k for k, v in rename_map.items() if v[1].lower() == new_name.lower()]
        if existing:
            # Append a suffix to disambiguate (shouldn't happen based on analysis)
            base = new_name[:-4]  # Remove .FIT
            suffix = 'b'
            while any(v[1].lower() == f"{base}_{suffix}.fit" for v in rename_map.values()):
                suffix = chr(ord(suffix) + 1)
            new_name = f"{base}_{suffix}.FIT"
            print(f"  Warning: Collision for {old_name}, using {new_name}")
        
        rename_map[old_name.lower()] = (old_name, new_name)
    
    print(f"Rename map: {len(rename_map)} files to rename, {already_correct} already correct, {skipped_no_date} skipped (no date)")
    return rename_map


def rename_folder_files(rename_map: dict, dry_run: bool = True) -> int:
    """Rename FIT files in TotalHistory folder (including subfolders)."""
    folder = Path(TOTAL_HISTORY_FOLDER)
    if not folder.exists():
        print(f"  Folder not found: {folder}")
        return 0
    
    fit_files_raw = list(folder.rglob("*.fit")) + list(folder.rglob("*.FIT"))
    # Deduplicate (Windows is case-insensitive, both globs return same files)
    seen = set()
    fit_files = []
    for f in fit_files_raw:
        key = str(f).lower()
        if key not in seen:
            seen.add(key)
            fit_files.append(f)
    renamed = 0
    
    for fit_path in fit_files:
        old_name = fit_path.name
        entry = rename_map.get(old_name.lower())
        
        if entry is None:
            continue
        
        _, new_name = entry
        new_path = fit_path.parent / new_name
        
        if new_path.exists() and new_path != fit_path:
            print(f"  WARNING: Target exists, skipping: {old_name} -> {new_name}")
            continue
        
        if dry_run:
            print(f"  [DRY] {old_name} -> {new_name}")
        else:
            fit_path.rename(new_path)
        renamed += 1
    
    return renamed


def rebuild_zip(dry_run: bool = True) -> int:
    """Rebuild TotalHistory.zip from the folder with renamed files."""
    folder = Path(TOTAL_HISTORY_FOLDER)
    if not folder.exists():
        print(f"  Folder not found: {folder}")
        return 0
    
    fit_files_raw = list(folder.rglob("*.fit")) + list(folder.rglob("*.FIT"))
    # Deduplicate (Windows is case-insensitive)
    seen_paths = set()
    fit_files = []
    for f in fit_files_raw:
        key = str(f).lower()
        if key not in seen_paths:
            seen_paths.add(key)
            fit_files.append(f)
    
    if dry_run:
        print(f"  [DRY] Would rebuild zip with {len(fit_files)} files")
        return len(fit_files)
    
    # Backup existing zip
    zip_path = Path(TOTAL_HISTORY_ZIP)
    if zip_path.exists():
        backup = zip_path.with_suffix(zip_path.suffix + BACKUP_SUFFIX)
        if not backup.exists():
            shutil.copy2(zip_path, backup)
            print(f"  Backed up zip to {backup.name}")
        zip_path.unlink()
    
    # Track filenames to avoid duplicates
    seen_names = set()
    duplicates = 0
    
    with zipfile.ZipFile(TOTAL_HISTORY_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fit_file in sorted(fit_files):
            name = fit_file.name
            if name.lower() in seen_names:
                duplicates += 1
                continue
            seen_names.add(name.lower())
            zf.write(fit_file, name)
    
    count = len(seen_names)
    print(f"  Rebuilt zip: {count} files" + (f" ({duplicates} duplicates skipped)" if duplicates else ""))
    return count


def update_master(master_path: str, rename_map: dict, dry_run: bool = True) -> int:
    """Update 'file' column in Master XLSX."""
    df = pd.read_excel(master_path, engine='openpyxl')
    updated = 0
    
    for i, row in df.iterrows():
        old_name = str(row.get('file', '')).strip()
        entry = rename_map.get(old_name.lower())
        if entry:
            _, new_name = entry
            df.at[i, 'file'] = new_name
            updated += 1
    
    if not dry_run and updated > 0:
        # Backup original
        backup = Path(master_path).with_suffix('.xlsx' + BACKUP_SUFFIX)
        if not backup.exists():
            shutil.copy2(master_path, backup)
            print(f"  Backed up Master to {backup.name}")
        df.to_excel(master_path, index=False)
    
    print(f"  Master: {updated} file references updated" + (" [DRY]" if dry_run else ""))
    return updated


def update_overrides(rename_map: dict, dry_run: bool = True) -> int:
    """Update 'file' column in activity_overrides.xlsx (filename-based entries only)."""
    if not os.path.exists(OVERRIDE_FILE):
        print("  No override file found - skipping")
        return 0
    
    df = pd.read_excel(OVERRIDE_FILE)
    if 'file' not in df.columns:
        return 0
    
    updated = 0
    for i, row in df.iterrows():
        old_name = str(row.get('file', '')).strip()
        entry = rename_map.get(old_name.lower())
        if entry:
            _, new_name = entry
            df.at[i, 'file'] = new_name
            updated += 1
    
    if not dry_run and updated > 0:
        backup = Path(OVERRIDE_FILE).with_suffix('.xlsx' + BACKUP_SUFFIX)
        if not backup.exists():
            shutil.copy2(OVERRIDE_FILE, backup)
        df.to_excel(OVERRIDE_FILE, index=False)
    
    print(f"  Overrides: {updated} file references updated" + (" [DRY]" if dry_run else ""))
    return updated


def update_pending(rename_map: dict, dry_run: bool = True) -> int:
    """Update 'file' column in pending_activities.csv."""
    if not os.path.exists(PENDING_ACTIVITIES_CSV):
        print("  No pending activities file found - skipping")
        return 0
    
    try:
        with open(PENDING_ACTIVITIES_CSV, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except (FileNotFoundError, UnicodeDecodeError):
        return 0
    
    if len(lines) < 2:
        return 0
    
    updated = 0
    new_lines = [lines[0]]  # Header
    
    for line in lines[1:]:
        parts = line.split(',', 1)
        if len(parts) >= 1:
            old_name = parts[0].strip()
            entry = rename_map.get(old_name.lower())
            if entry:
                _, new_name = entry
                new_lines.append(new_name + ',' + parts[1] if len(parts) > 1 else new_name + '\n')
                updated += 1
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    if not dry_run and updated > 0:
        with open(PENDING_ACTIVITIES_CSV, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
    
    print(f"  Pending activities: {updated} file references updated" + (" [DRY]" if dry_run else ""))
    return updated


def rename_cache_files(rename_map: dict, dry_run: bool = True) -> int:
    """Rename .npz cache files to match new FIT filenames."""
    cache_dir = Path(PERSEC_CACHE_DIR)
    if not cache_dir.exists():
        print(f"  Cache dir not found: {cache_dir} - skipping")
        return 0
    
    npz_files = list(cache_dir.glob("*.npz"))
    renamed = 0
    
    for npz_path in npz_files:
        # Cache files are named by FIT stem: "5812301439434752.npz"
        old_stem = npz_path.stem
        # Try matching with .fit and .FIT extensions
        entry = rename_map.get(f"{old_stem}.fit") or rename_map.get(f"{old_stem}.FIT")
        # Also try the stem with .fit lowercase
        if entry is None:
            entry = rename_map.get(old_stem.lower() + ".fit")
        
        if entry is None:
            continue
        
        _, new_fit_name = entry
        new_stem = Path(new_fit_name).stem
        new_path = npz_path.parent / f"{new_stem}.npz"
        
        if new_path.exists() and new_path != npz_path:
            print(f"  WARNING: Cache target exists, skipping: {npz_path.name} -> {new_path.name}")
            continue
        
        if dry_run:
            print(f"  [DRY] Cache: {npz_path.name} -> {new_path.name}")
        else:
            npz_path.rename(new_path)
        renamed += 1
    
    if renamed > 0 or not dry_run:
        print(f"  Cache files: {renamed} renamed" + (" [DRY]" if dry_run else ""))
    return renamed


def main():
    parser = argparse.ArgumentParser(description="Rename FIT files to YYYY-MM-DD_HH-MM-SS.FIT")
    parser.add_argument("--master", required=True, help="Master XLSX file path")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't rename")
    parser.add_argument("--execute", action="store_true", help="Actually rename (without this, defaults to dry-run)")
    parser.add_argument("--zip-only", action="store_true", help="Only rebuild the zip (after manual rename)")
    args = parser.parse_args()
    
    # Default to dry-run unless --execute is given
    if not args.dry_run and not args.execute and not args.zip_only:
        args.dry_run = True
        print("(No --execute flag, running as dry-run)")
    
    print("=" * 60)
    print("FIT File Rename Utility")
    print("=" * 60)
    
    if args.zip_only:
        print("\nRebuilding zip from folder...")
        rebuild_zip(dry_run=args.dry_run)
        return 0
    
    # Build the rename map from Master
    print(f"\nReading Master: {args.master}")
    rename_map = build_rename_map(args.master)
    
    if not rename_map:
        print("\nNothing to rename!")
        return 0
    
    # Show sample
    print(f"\nSample renames:")
    for i, (old_lower, (old_orig, new_name)) in enumerate(rename_map.items()):
        if i >= 10:
            print(f"  ... and {len(rename_map) - 10} more")
            break
        print(f"  {old_orig} -> {new_name}")
    
    if args.dry_run:
        print("\n[DRY RUN - no files will be changed]\n")
    else:
        print(f"\nExecuting rename of {len(rename_map)} files...")
        print()
    
    # Execute
    print(f"\n--- Step 1: Rename files in TotalHistory folder ---")
    n_folder = rename_folder_files(rename_map, dry_run=args.dry_run)
    
    print(f"\n--- Step 2: Rebuild TotalHistory.zip ---")
    n_zip = rebuild_zip(dry_run=args.dry_run)
    
    print(f"\n--- Step 3: Update Master XLSX ---")
    n_master = update_master(args.master, rename_map, dry_run=args.dry_run)
    
    print(f"\n--- Step 4: Update override file ---")
    n_overrides = update_overrides(rename_map, dry_run=args.dry_run)
    
    print(f"\n--- Step 5: Update pending activities ---")
    n_pending = update_pending(rename_map, dry_run=args.dry_run)
    
    print(f"\n--- Step 6: Rename cache files ---")
    n_cache = rename_cache_files(rename_map, dry_run=args.dry_run)
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"Summary {'(DRY RUN)' if args.dry_run else ''}:")
    print(f"  Folder files renamed: {n_folder}")
    print(f"  Zip rebuilt: {n_zip} files")
    print(f"  Master references updated: {n_master}")
    print(f"  Override references updated: {n_overrides}")
    print(f"  Pending references updated: {n_pending}")
    print(f"  Cache files renamed: {n_cache}")
    print(f"{'=' * 60}")
    
    if not args.dry_run:
        print("\nDone! Run a FULL pipeline rebuild to verify everything works.")
        print("Backups were created with '.pre_rename_backup' suffix.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
