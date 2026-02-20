"""
fetch_fit_files.py — Download new FIT files from intervals.icu
==============================================================
Phase 2: Replaces manual FIT file export from Garmin Connect.

Tracks the last-synced activity date in sync_state.json and only
downloads FIT files for new running activities.

Usage:
  python fetch_fit_files.py                               # Download new runs
  python fetch_fit_files.py --full                        # Re-download all runs
  python fetch_fit_files.py --since 2026-01-01            # Download from specific date
  python fetch_fit_files.py --list-only                   # Preview without downloading
  python fetch_fit_files.py --rezip                       # Recreate TotalHistory.zip

Output:
  - FIT files saved to --fit-dir (default: TotalHistory/)
  - Optional: recreates TotalHistory.zip for pipeline compatibility
  - Updates sync_state.json with last successful sync

Environment variables:
  INTERVALS_API_KEY=your_key
  INTERVALS_ATHLETE_ID=i12345
"""

import os
import sys
import json
import gzip
import zipfile
import argparse
from datetime import datetime, timedelta
from pathlib import Path

try:
    from intervals_fetch import IntervalsClient
except ImportError:
    sys.exit("Cannot import intervals_fetch.py — ensure it's in the same directory")


# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_FIT_DIR = "FIT_downloads"
DEFAULT_STATE_FILE = "fit_sync_state.json"
DEFAULT_ZIP_FILE = "TotalHistory.zip"
FULL_HISTORY_START = "2013-01-01"
RUNNING_TYPES = {"Run", "VirtualRun"}     # Activity types to download FIT files for


def load_sync_state(path: str) -> dict:
    """Load sync state from JSON file."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_sync_state(path: str, state: dict):
    """Save sync state to JSON file."""
    state["updated_at"] = datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def get_existing_fit_files(fit_dir: str, zip_path: str = "") -> set:
    """Get set of existing FIT filenames (without extension) in the directory and/or zip."""
    existing = set()
    
    if os.path.isdir(fit_dir):
        for f in Path(fit_dir).rglob("*.fit"):
            existing.add(f.stem)
        for f in Path(fit_dir).rglob("*.FIT"):
            existing.add(f.stem)
    
    # Also check inside fits.zip (CI: FIT_downloads is empty but zip has all history)
    if zip_path and os.path.exists(zip_path):
        try:
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for name in zf.namelist():
                    stem = Path(name).stem
                    if name.lower().endswith('.fit'):
                        existing.add(stem)
        except Exception as e:
            print(f"  Warning: could not read zip {zip_path}: {e}")
    
    return existing


def activity_to_filename(activity: dict) -> str:
    """
    Generate a FIT filename from an intervals.icu activity.
    
    Matches the naming convention used by Garmin Connect exports:
    e.g., "2026-02-04_10-30-00.FIT" or activity ID based.
    
    We use the start_date_local to create a Garmin-compatible name.
    """
    date_str = activity.get("start_date_local", "")
    
    if date_str:
        # Parse ISO datetime and format as Garmin-style filename
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d_%H-%M-%S")
        except (ValueError, TypeError):
            pass
    
    # Fallback: use activity ID
    act_id = activity.get("id", "unknown")
    return act_id.replace(":", "_")


def fetch_new_fit_files(client: IntervalsClient,
                        fit_dir: str,
                        since: str,
                        existing_files: set,
                        list_only: bool = False) -> list:
    """
    Fetch FIT files for new running activities.
    
    Args:
        client: IntervalsClient instance
        fit_dir: Directory to save FIT files
        since: Start date for activity search
        existing_files: Set of existing filenames (stems) to skip
        list_only: If True, list activities without downloading
    
    Returns:
        List of (filename, activity) tuples for successfully downloaded files
    """
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Fetching running activities {since} -> {today}...")
    
    all_activities = client.get_activities(since, today)
    run_activities = [
        act for act in all_activities
        if act.get("type") in RUNNING_TYPES
    ]
    
    print(f"  -> {len(all_activities)} total activities, {len(run_activities)} runs")
    
    # Determine which need downloading
    to_download = []
    already_have = 0
    
    # Load known activity IDs from sync state (handles filename mismatch between
    # datetime-named downloads and ID-named files in the zip)
    known_activity_ids = set(state.get("downloaded_activity_ids", []))
    
    for act in run_activities:
        filename = activity_to_filename(act)
        act_id = str(act.get("id", ""))
        
        if filename in existing_files or act_id in known_activity_ids:
            already_have += 1
            continue
        
        to_download.append((filename, act))
    
    print(f"  -> {already_have} already in {fit_dir}, {len(to_download)} new to download")
    
    if not to_download:
        print("  Nothing new to download.")
        return []
    
    if list_only:
        print(f"\n  New activities to download:")
        for filename, act in to_download:
            dist_km = (act.get("distance", 0) or 0) / 1000
            name = act.get("name", "?")
            print(f"    {filename}.FIT  {dist_km:6.2f}km  {name}")
        return to_download
    
    # Download FIT files
    os.makedirs(fit_dir, exist_ok=True)
    downloaded = []
    errors = 0
    
    for i, (filename, act) in enumerate(to_download, 1):
        act_id = act.get("id", "?")
        name = act.get("name", "?")
        dist_km = (act.get("distance", 0) or 0) / 1000
        
        print(f"  [{i}/{len(to_download)}] {filename}.FIT ({dist_km:.1f}km, {name})...", end=" ", flush=True)
        
        try:
            fit_bytes = client.download_fit(act_id)
            
            if fit_bytes is None:
                print("[WARN] no FIT file available")
                continue
            
            fit_path = os.path.join(fit_dir, f"{filename}.FIT")
            with open(fit_path, "wb") as f:
                f.write(fit_bytes)
            
            print(f"[OK] {len(fit_bytes):,} bytes")
            downloaded.append((filename, act))
            
        except Exception as e:
            print(f"[FAIL] {e}")
            errors += 1
    
    print(f"\n  Downloaded: {len(downloaded)}, errors: {errors}")
    return downloaded


def recreate_zip(fit_dir: str, zip_path: str):
    """Recreate TotalHistory.zip from all FIT files in directory (including subfolders)."""
    print(f"\nRecreating {zip_path}...")
    
    folder = Path(fit_dir)
    fit_files = list(folder.rglob("*.fit")) + list(folder.rglob("*.FIT"))
    
    if not fit_files:
        print(f"  [WARN] No FIT files found in {fit_dir}")
        return
    
    # Track filenames to avoid duplicates
    seen_names = set()
    duplicates = 0
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fit_file in sorted(fit_files):
            name = fit_file.name
            if name in seen_names:
                duplicates += 1
                continue
            seen_names.add(name)
            zf.write(fit_file, name)
    
    print(f"  Created: {zip_path} ({len(seen_names)} files, {os.path.getsize(zip_path) / 1024 / 1024:.1f} MB)")
    if duplicates > 0:
        print(f"  [WARN] Skipped {duplicates} duplicate filenames")


def generate_pending_activities(downloaded: list, pending_csv_path: str):
    """
    Append downloaded activities to pending_activities.csv.
    
    Uses intervals.icu activity names so they appear in the Master file.
    Skips entries where a date-based key already exists (from dispatch metadata).
    """
    if not downloaded:
        return
    
    print(f"\nUpdating {pending_csv_path}...")
    
    # Read existing entries — both filename keys and date-based keys
    existing_files = set()
    existing_dates = set()  # date-based keys from dispatch metadata
    if os.path.exists(pending_csv_path):
        import csv
        import re
        date_pat = re.compile(r'^(\d{4}-\d{2}-\d{2})\s*(?:#\d+)?$')
        with open(pending_csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    key = row[0].strip()
                    existing_files.add(key)
                    m = date_pat.match(key)
                    if m:
                        existing_dates.add(m.group(1))
    else:
        with open(pending_csv_path, "w", encoding="utf-8") as f:
            f.write("file,activity_name,shoe\n")
    
    # Append new entries (skip if filename exists OR date already has dispatch metadata)
    added = 0
    skipped_dispatch = 0
    with open(pending_csv_path, "a", encoding="utf-8") as f:
        for filename, act in downloaded:
            fit_name = f"{filename}.FIT"
            if fit_name in existing_files:
                continue
            
            # Extract date from filename (YYYY-MM-DD_HH-MM-SS.FIT)
            fit_date = filename[:10] if len(filename) >= 10 else ""
            if fit_date in existing_dates:
                skipped_dispatch += 1
                continue
            
            name = act.get("name", "").replace(",", " ").replace('"', "")
            if not name:
                name = f"{act.get('type', 'Run')} {act.get('start_date_local', '')[:10]}"
            
            f.write(f'{fit_name},"{name}",\n')
            added += 1
    
    if skipped_dispatch > 0:
        print(f"  Skipped {skipped_dispatch} (dispatch metadata already set)")
    print(f"  Added {added} entries to pending_activities.csv")


def refresh_activity_names(client, pending_csv: str, days: int = 14):
    """
    Re-query intervals.icu for recent activity names and update pending_activities.csv.
    
    This catches name edits made after the initial download — e.g. user renames
    "Morning Run" to "Tempo 5K" on intervals.icu after uploading.
    
    Only updates names for files already in pending_activities.csv.
    Also adds entries for recent runs not yet in pending (e.g. if the CSV was cleared).
    """
    import csv
    
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    print(f"\n--- Refreshing activity names (last {days} days) ---")
    
    # Fetch recent activities from intervals.icu
    activities = client.get_activities(since)
    runs = [a for a in activities if a.get("type") in ("Run", "VirtualRun")]
    print(f"  Found {len(runs)} runs on intervals.icu since {since}")
    
    if not runs:
        return
    
    # Build lookup: filename stem -> intervals name
    # fetch_fit_files saves as YYYY-MM-DD_HH-MM-SS.FIT
    intervals_names = {}
    for act in runs:
        start = act.get("start_date_local", "")
        name = act.get("name", "")
        if start and name:
            # Convert ISO datetime to our filename format
            try:
                dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                stem = dt.strftime("%Y-%m-%d_%H-%M-%S")
                intervals_names[f"{stem}.FIT"] = name.replace(",", " ").replace('"', "")
            except (ValueError, TypeError):
                pass
    
    if not intervals_names:
        print("  No names to refresh")
        return
    
    # Read existing pending_activities.csv
    existing_rows = []
    existing_files = set()
    if os.path.exists(pending_csv):
        with open(pending_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row:
                    existing_rows.append(row)
                    existing_files.add(row[0].strip())
    
    # Update existing entries and add new ones
    updated = 0
    added = 0
    
    # Update existing rows
    for i, row in enumerate(existing_rows):
        filename = row[0].strip()
        if filename in intervals_names:
            old_name = row[1].strip().strip('"') if len(row) > 1 else ""
            new_name = intervals_names[filename]
            if old_name != new_name:
                row[1] = f'"{new_name}"'
                updated += 1
    
    # Add new entries for files not yet in pending
    for filename, name in intervals_names.items():
        if filename not in existing_files:
            existing_rows.append([filename, f'"{name}"', ""])
            added += 1
    
    # Write back
    if updated > 0 or added > 0:
        with open(pending_csv, "w", encoding="utf-8", newline="") as f:
            f.write("file,activity_name,shoe\n")
            for row in existing_rows:
                f.write(",".join(row) + "\n")
        print(f"  Updated: {updated} names, Added: {added} new entries")
    else:
        print("  No changes needed")


def main():
    parser = argparse.ArgumentParser(
        description="Download new FIT files from intervals.icu"
    )
    parser.add_argument("--fit-dir", default=DEFAULT_FIT_DIR,
                        help=f"Directory for FIT files (default: {DEFAULT_FIT_DIR})")
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE,
                        help=f"Sync state file (default: {DEFAULT_STATE_FILE})")
    parser.add_argument("--zip", default=DEFAULT_ZIP_FILE,
                        help=f"Output zip file (default: {DEFAULT_ZIP_FILE})")
    parser.add_argument("--since", default=None,
                        help="Download from this date (overrides sync state)")
    parser.add_argument("--full", action="store_true",
                        help="Download all history (ignores sync state)")
    parser.add_argument("--list-only", action="store_true",
                        help="List new activities without downloading")
    parser.add_argument("--rezip", action="store_true",
                        help="Just recreate zip from existing FIT files")
    parser.add_argument("--no-zip", action="store_true", default=True,
                        help="Skip zip creation after download (default: True, use --zip to rebuild)")
    parser.add_argument("--zip-rebuild", action="store_true",
                        help="Rebuild zip from FIT_downloads after download (REPLACES existing zip)")
    parser.add_argument("--pending-csv", default="pending_activities.csv",
                        help="Path to pending_activities.csv")
    parser.add_argument("--no-pending", action="store_true",
                        help="Skip updating pending_activities.csv")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--athlete-id", default=None)
    parser.add_argument("--refresh-names", type=int, default=0, metavar="DAYS",
                        help="Re-query intervals.icu for activity names from the last N days "
                             "and update pending_activities.csv (e.g. --refresh-names 14)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Fetch FIT Files from intervals.icu")
    print("=" * 60)
    
    # Clean up any previous signal file
    try:
        os.remove('_new_runs_added.tmp')
    except FileNotFoundError:
        pass

    # Rezip mode: just recreate the zip and exit
    if args.rezip:
        recreate_zip(args.fit_dir, args.zip)
        return 0
    
    # Initialize client
    try:
        client = IntervalsClient(api_key=args.api_key, athlete_id=args.athlete_id)
    except ValueError as e:
        print(f"\nERROR: {e}")
        return 1
    
    # Refresh activity names if requested (before download, so new names are up to date)
    if args.refresh_names > 0 and not args.no_pending:
        refresh_activity_names(client, args.pending_csv, days=args.refresh_names)

    # Determine start date
    state = load_sync_state(args.state_file)
    
    if args.full:
        since = FULL_HISTORY_START
        print(f"\nFull download mode: {since} -> today")
    elif args.since:
        since = args.since
        print(f"\nManual start date: {since}")
    elif "last_sync_date" in state:
        # Go back 2 days from last sync to catch any late-arriving activities
        last = datetime.strptime(state["last_sync_date"], "%Y-%m-%d")
        since = (last - timedelta(days=2)).strftime("%Y-%m-%d")
        print(f"\nIncremental from last sync: {state['last_sync_date']} (checking from {since})")
    else:
        # First run: start from 30 days ago (not full history)
        # Use --full explicitly for a complete download
        since = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        print(f"\nNo sync state found — starting from {since} (use --full for complete history)")
    
    # Get existing files
    existing = get_existing_fit_files(args.fit_dir, zip_path=args.zip)
    print(f"Existing FIT files in {args.fit_dir}: {len(existing)}")
    
    # Fetch and download
    downloaded = fetch_new_fit_files(
        client, args.fit_dir, since, existing, list_only=args.list_only
    )
    
    if args.list_only:
        return 0
    
    # Append new FIT files to the zip so rebuild_from_fit_zip can find them
    if downloaded and os.path.exists(args.zip):
        print(f"\nAppending {len(downloaded)} new FIT file(s) to {args.zip}...")
        existing_in_zip = set()
        with zipfile.ZipFile(args.zip, 'r') as zf:
            existing_in_zip = {os.path.basename(n).lower() for n in zf.namelist()}
        added = 0
        with zipfile.ZipFile(args.zip, 'a', zipfile.ZIP_DEFLATED) as zf:
            for fname, _act in downloaded:
                # downloaded stores stem without .FIT; actual file has .FIT
                fname_fit = fname if fname.upper().endswith('.FIT') else fname + '.FIT'
                if fname_fit.lower() not in existing_in_zip:
                    fit_path = os.path.join(args.fit_dir, fname_fit)
                    if os.path.exists(fit_path):
                        zf.write(fit_path, fname_fit)
                        added += 1
                    else:
                        print(f"  [WARN] {fit_path} not found on disk")
        print(f"  Added {added} to zip ({args.zip})")
    
    # Update pending_activities.csv
    if not args.no_pending and downloaded:
        generate_pending_activities(downloaded, args.pending_csv)
    
    # Recreate zip only if explicitly requested
    if args.zip_rebuild and downloaded:
        recreate_zip(args.fit_dir, args.zip)
    
    # Update sync state
    if downloaded:
        today = datetime.now().strftime("%Y-%m-%d")
        state["last_sync_date"] = today
        state["last_download_count"] = len(downloaded)
        state["total_fit_files"] = len(get_existing_fit_files(args.fit_dir, zip_path=args.zip))
        # Track downloaded activity IDs for dedup (filenames differ between zip and fetch)
        existing_ids = set(state.get("downloaded_activity_ids", []))
        for fname, act in downloaded:
            act_id = str(act.get("id", ""))
            if act_id:
                existing_ids.add(act_id)
        state["downloaded_activity_ids"] = sorted(existing_ids)
        save_sync_state(args.state_file, state)
        print(f"\nSync state updated: {args.state_file}")
    
    # Signal to Daily_Update.bat that new runs were downloaded
    if downloaded:
        try:
            with open('_new_runs_added.tmp', 'w') as f:
                f.write(str(len(downloaded)))
        except Exception:
            pass
    
    print(f"\n[OK] Done. {len(downloaded)} new FIT files downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
