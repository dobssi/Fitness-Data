#!/usr/bin/env python3
"""ci/initial_data_ingest.py — Download and ingest initial athlete data from Dropbox.

Called during INITIAL pipeline mode. Finds any zip in the athlete's data/ folder
on Dropbox, downloads it, detects whether it's a Strava or Garmin export, and
produces fits.zip + activities.csv ready for rebuild_from_fit_zip.py.

FIT files are renamed to YYYY-MM-DD_HH-MM-SS.FIT (matching intervals.icu naming)
to prevent duplicates when intervals.icu later fetches the same runs.

Usage:
    python ci/initial_data_ingest.py \
        --athlete-dir athletes/A005 \
        --dropbox-base "/Running and Cycling/DataPipeline/athletes/A005" \
        --tz Europe/Stockholm
"""

import argparse
import io
import os
import sys
import zipfile
from datetime import timezone

# Add ci/ to path for dropbox_sync imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from dropbox_sync import get_token, dropbox_list_folder, dropbox_download


def fit_start_timestamp(fit_bytes: bytes, tz_str: str = "UTC") -> str:
    """Read start_time from FIT file bytes, return 'YYYY-MM-DD_HH-MM-SS' in local tz.

    Returns None if timestamp cannot be read.
    """
    from fitparse import FitFile
    from datetime import datetime
    try:
        import pytz
        local_tz = pytz.timezone(tz_str)
    except ImportError:
        from zoneinfo import ZoneInfo
        local_tz = ZoneInfo(tz_str)

    try:
        fit = FitFile(io.BytesIO(fit_bytes))
        for msg in fit.get_messages("session"):
            start = msg.get_value("start_time")
            if start is not None:
                if isinstance(start, datetime):
                    if start.tzinfo is None:
                        start = start.replace(tzinfo=timezone.utc)
                    local_dt = start.astimezone(local_tz)
                    return local_dt.strftime("%Y-%m-%d_%H-%M-%S")
        # Fallback: try first record timestamp
        fit = FitFile(io.BytesIO(fit_bytes))
        for msg in fit.get_messages("record"):
            ts = msg.get_value("timestamp")
            if ts is not None:
                if isinstance(ts, datetime):
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    local_dt = ts.astimezone(local_tz)
                    return local_dt.strftime("%Y-%m-%d_%H-%M-%S")
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Download and ingest initial athlete data")
    parser.add_argument("--athlete-dir", required=True, help="Local athlete directory (e.g. athletes/A005)")
    parser.add_argument("--dropbox-base", required=True, help="Dropbox base path for athlete")
    parser.add_argument("--tz", default="Europe/London", help="Timezone")
    args = parser.parse_args()

    data_dir = os.path.join(args.athlete_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # ── Step 1: Find and download zip(s) from Dropbox data/ folder ────────
    remote_data = args.dropbox_base.rstrip("/") + "/data"
    token = get_token()

    print(f"\nLooking for data files in {remote_data}/ ...")
    try:
        files = dropbox_list_folder(remote_data, token)
    except Exception as e:
        print(f"  Could not list {remote_data}/: {e}")
        files = {}

    zips = [f for f in files if f.lower().endswith(".zip") and f.lower() != "fits.zip"]
    
    if not zips:
        # Check if fits.zip already exists (maybe from a previous run)
        fits_zip = os.path.join(data_dir, "fits.zip")
        if os.path.isfile(fits_zip):
            print(f"  No export zip found, but fits.zip exists — ready to use")
            return 0
        print(f"  No zip files found in data/ on Dropbox")
        print(f"  Will rely on intervals.icu sync if available")
        return 0

    # Download the largest zip (the export)
    export_name = zips[0]
    if len(zips) > 1:
        print(f"  Found {len(zips)} zips: {zips}")
        print(f"  Using: {export_name}")

    local_zip = os.path.join(data_dir, export_name)
    remote_zip = remote_data + "/" + export_name
    
    print(f"  Downloading {export_name}...")
    dropbox_download(remote_zip, local_zip, token)
    
    size_mb = os.path.getsize(local_zip) / (1024 * 1024)
    print(f"  Downloaded: {size_mb:.0f} MB")

    # ── Step 2: Detect export type ────────────────────────────────────────
    with zipfile.ZipFile(local_zip, "r") as zf:
        names = zf.namelist()
        has_activities_folder = any("activities/" in n for n in names)
        fit_files = [n for n in names if n.lower().endswith(".fit")]

    if has_activities_folder:
        export_type = "strava"
    elif fit_files:
        export_type = "garmin"
    else:
        print(f"  WARNING: Could not detect export type. Trying as Garmin/FIT.")
        export_type = "garmin"

    print(f"  Detected: {export_type} export ({len(fit_files)} FIT files in zip)")

    # ── Step 3: Ingest ────────────────────────────────────────────────────
    fits_zip_path = os.path.join(data_dir, "fits.zip")
    activities_csv = os.path.join(data_dir, "activities.csv")

    if export_type == "strava":
        # Use strava_ingest.py for full Strava processing
        import subprocess
        strava_out = os.path.join(data_dir, "strava_data")
        cmd = [
            sys.executable, "strava_ingest.py",
            "--strava-zip", local_zip,
            "--out-dir", strava_out,
            "--persec-cache-dir", os.path.join(args.athlete_dir, "persec_cache"),
            "--tz", args.tz,
        ]
        print(f"\n  Running strava_ingest.py...")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  ERROR: strava_ingest.py failed with code {result.returncode}")
            return 1
        
        # Copy outputs to expected locations
        strava_fits = os.path.join(strava_out, "fits.zip")
        strava_csv = os.path.join(strava_out, "activities.csv")
        if os.path.isfile(strava_fits):
            import shutil
            shutil.copy2(strava_fits, fits_zip_path)
            print(f"  ✓ fits.zip ready")
        if os.path.isfile(strava_csv):
            import shutil
            shutil.copy2(strava_csv, activities_csv)
            print(f"  ✓ activities.csv ready")

    else:
        # Garmin or raw FIT export — extract FIT files into fits.zip
        # Rename to YYYY-MM-DD_HH-MM-SS.FIT to match intervals.icu naming
        # Filter to running activities only by reading sport type from FIT session
        print(f"  Extracting and renaming {len(fit_files)} FIT files...")
        renamed = 0
        skipped_nonrun = 0
        seen_names = set()
        with zipfile.ZipFile(local_zip, "r") as zin:
            with zipfile.ZipFile(fits_zip_path, "w", zipfile.ZIP_DEFLATED) as zout:
                for fit in fit_files:
                    data = zin.read(fit)
                    # Filter: read sport type from FIT session — skip non-running
                    try:
                        from fitparse import FitFile
                        _fit = FitFile(io.BytesIO(data))
                        _sess = next(iter(_fit.get_messages("session")), None)
                        _sport = _sess.get_value("sport") if _sess else None
                        if _sport is not None:
                            _s = str(_sport).lower()
                            if ("run" not in _s) and ("running" not in _s):
                                skipped_nonrun += 1
                                continue
                    except Exception:
                        pass  # If we can't read sport, include (rebuild will filter later)
                    
                    ts_name = fit_start_timestamp(data, args.tz)
                    if ts_name:
                        new_name = f"{ts_name}.FIT"
                        # Handle rare same-second starts (e.g. multisport)
                        if new_name in seen_names:
                            for suffix in range(2, 10):
                                candidate = f"{ts_name}_{suffix}.FIT"
                                if candidate not in seen_names:
                                    new_name = candidate
                                    break
                        seen_names.add(new_name)
                        renamed += 1
                    else:
                        new_name = os.path.basename(fit)
                        seen_names.add(new_name)
                    zout.writestr(new_name, data)
        n_included = len(fit_files) - skipped_nonrun
        print(f"  ✓ fits.zip ready ({n_included} running files, {renamed} renamed)")
        if skipped_nonrun:
            print(f"    Skipped {skipped_nonrun} non-running FIT files")

        # Create empty activities.csv if needed
        if not os.path.isfile(activities_csv):
            with open(activities_csv, "w") as f:
                f.write("activity_id,timestamp,name,type\n")
            print(f"  ✓ activities.csv created (empty)")

    print(f"\n  Initial data ingest complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
