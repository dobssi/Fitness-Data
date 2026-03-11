#!/usr/bin/env python3
"""ci/initial_data_ingest.py — Download and ingest initial athlete data from Dropbox.

Called during INITIAL pipeline mode. Finds zip(s) in the athlete's data/ folder
on Dropbox, downloads them, detects export types (Strava, Polar, Garmin), and
produces fits.zip + activities.csv ready for rebuild_from_fit_zip.py.

Supports multiple zips: e.g. Polar primary + Strava supplementary (for treadmill
runs not in Polar export). The Strava supplementary produces:
  - Extra FIT files merged into fits.zip
  - gpx_tcx_summaries.csv for TCX-only runs (rebuild --extra-summaries)
  - activities.csv for name matching

FIT files are renamed to YYYY-MM-DD_HH-MM-SS.FIT (matching intervals.icu naming)
to prevent duplicates when intervals.icu later fetches the same runs.

Usage:
    python ci/initial_data_ingest.py \\
        --athlete-dir athletes/A007 \\
        --dropbox-base "/Running and Cycling/DataPipeline/athletes/A007" \\
        --tz Europe/Stockholm
"""

import argparse
import io
import os
import shutil
import subprocess
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


def detect_export_type(zip_path: str) -> str:
    """Detect whether a zip is a Strava, Polar, or Garmin export."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        has_activities_folder = any("activities/" in n for n in names)
        polar_sessions = [n for n in names if "training-session-" in n and n.endswith(".json")]
        fit_files = [n for n in names if n.lower().endswith(".fit")]

    if has_activities_folder:
        return "strava"
    elif polar_sessions:
        return "polar"
    elif fit_files:
        return "garmin"
    else:
        return "unknown"


def ingest_polar(local_zip: str, data_dir: str, tz: str) -> int:
    """Run polar_ingest.py on a Polar export zip."""
    cmd = [
        sys.executable, "ci/polar_ingest.py",
        "--polar-zip", local_zip,
        "--out-dir", data_dir,
        "--tz", tz,
    ]
    print(f"\n  Running polar_ingest.py...")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def ingest_strava(local_zip: str, data_dir: str, athlete_dir: str, tz: str,
                  supplementary: bool = False) -> int:
    """Run strava_ingest.py on a Strava export zip.

    If supplementary=True, merges FIT files into existing fits.zip and preserves
    gpx_tcx_summaries.csv for rebuild --extra-summaries.
    """
    strava_out = os.path.join(data_dir, "strava_data")
    cmd = [
        sys.executable, "strava_ingest.py",
        "--strava-zip", local_zip,
        "--out-dir", strava_out,
        "--persec-cache-dir", os.path.join(athlete_dir, "persec_cache"),
        "--tz", tz,
    ]
    print(f"\n  Running strava_ingest.py{'  (supplementary)' if supplementary else ''}...")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: strava_ingest.py failed with code {result.returncode}")
        return result.returncode

    fits_zip_path = os.path.join(data_dir, "fits.zip")
    activities_csv = os.path.join(data_dir, "activities.csv")
    strava_fits = os.path.join(strava_out, "fits.zip")
    strava_csv = os.path.join(strava_out, "activities.csv")
    strava_summaries = os.path.join(strava_out, "gpx_tcx_summaries.csv")

    if supplementary:
        # Merge Strava FIT files into existing fits.zip
        if os.path.isfile(strava_fits) and os.path.isfile(fits_zip_path):
            n_added = 0
            existing_names = set()
            with zipfile.ZipFile(fits_zip_path, "r") as zf:
                existing_names = set(zf.namelist())
            with zipfile.ZipFile(fits_zip_path, "a", zipfile.ZIP_DEFLATED) as zout:
                with zipfile.ZipFile(strava_fits, "r") as zin:
                    for name in zin.namelist():
                        if name not in existing_names:
                            zout.writestr(name, zin.read(name))
                            n_added += 1
            print(f"  ✓ Merged {n_added} new FIT files into fits.zip")
        elif os.path.isfile(strava_fits) and not os.path.isfile(fits_zip_path):
            shutil.copy2(strava_fits, fits_zip_path)
            print(f"  ✓ fits.zip ready (from Strava)")

        # Copy GPX/TCX summaries for rebuild --extra-summaries
        extra_summaries_path = os.path.join(data_dir, "gpx_tcx_summaries.csv")
        if os.path.isfile(strava_summaries):
            shutil.copy2(strava_summaries, extra_summaries_path)
            # Count rows
            try:
                with open(extra_summaries_path) as f:
                    n_rows = sum(1 for _ in f) - 1  # minus header
                print(f"  ✓ gpx_tcx_summaries.csv ready ({n_rows} extra runs)")
            except Exception:
                print(f"  ✓ gpx_tcx_summaries.csv ready")

        # Merge activities.csv (append Strava names to existing)
        if os.path.isfile(strava_csv):
            if os.path.isfile(activities_csv):
                # Append Strava activities to existing CSV
                import pandas as pd
                try:
                    df_existing = pd.read_csv(activities_csv)
                    df_strava = pd.read_csv(strava_csv)
                    df_merged = pd.concat([df_existing, df_strava], ignore_index=True)
                    df_merged.to_csv(activities_csv, index=False)
                    print(f"  ✓ activities.csv merged ({len(df_strava)} Strava entries added)")
                except Exception as e:
                    print(f"  WARNING: Could not merge activities.csv: {e}")
            else:
                shutil.copy2(strava_csv, activities_csv)
                print(f"  ✓ activities.csv ready (from Strava)")
    else:
        # Primary Strava ingest — just copy outputs
        if os.path.isfile(strava_fits):
            shutil.copy2(strava_fits, fits_zip_path)
            print(f"  ✓ fits.zip ready")
        if os.path.isfile(strava_csv):
            shutil.copy2(strava_csv, activities_csv)
            print(f"  ✓ activities.csv ready")
        # Copy GPX/TCX summaries for rebuild --extra-summaries
        extra_summaries_path = os.path.join(data_dir, "gpx_tcx_summaries.csv")
        if os.path.isfile(strava_summaries):
            shutil.copy2(strava_summaries, extra_summaries_path)
            print(f"  ✓ gpx_tcx_summaries.csv ready")

    return 0


def ingest_garmin(local_zip: str, data_dir: str, tz: str) -> int:
    """Extract FIT files from a Garmin export zip into fits.zip."""
    fits_zip_path = os.path.join(data_dir, "fits.zip")
    activities_csv = os.path.join(data_dir, "activities.csv")

    with zipfile.ZipFile(local_zip, "r") as zf:
        fit_files = [n for n in zf.namelist() if n.lower().endswith(".fit")]

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

                ts_name = fit_start_timestamp(data, tz)
                if ts_name:
                    new_name = f"{ts_name}.FIT"
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

    return 0


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
        fits_zip = os.path.join(data_dir, "fits.zip")
        if os.path.isfile(fits_zip):
            print(f"  No export zip found, but fits.zip exists — ready to use")
            return 0
        print(f"  No zip files found in data/ on Dropbox")
        print(f"  Will rely on intervals.icu sync if available")
        return 0

    # ── Step 2: Download ALL zips and classify ────────────────────────────
    exports = []  # list of (local_path, export_type)
    for zip_name in zips:
        local_zip = os.path.join(data_dir, zip_name)
        remote_zip = remote_data + "/" + zip_name

        print(f"\n  Downloading {zip_name}...")
        dropbox_download(remote_zip, local_zip, token)

        size_mb = os.path.getsize(local_zip) / (1024 * 1024)
        export_type = detect_export_type(local_zip)
        print(f"  Downloaded: {size_mb:.0f} MB — detected as {export_type}")
        exports.append((local_zip, export_type))

    # ── Step 3: Process — primary first, then supplementary ──────────────
    # Priority: Polar → Garmin → Strava as primary (produces FIT files)
    # Any remaining Strava exports are processed as supplementary (adds TCX runs)
    primary_done = False
    strava_exports = []

    # Sort: non-strava first
    for local_zip, export_type in exports:
        if export_type == "strava":
            strava_exports.append(local_zip)
        elif not primary_done:
            print(f"\n{'='*60}")
            print(f"Processing PRIMARY export: {os.path.basename(local_zip)} ({export_type})")
            print(f"{'='*60}")
            if export_type == "polar":
                rc = ingest_polar(local_zip, data_dir, args.tz)
            elif export_type == "garmin":
                rc = ingest_garmin(local_zip, data_dir, args.tz)
            else:
                print(f"  WARNING: Unknown export type '{export_type}'. Trying as Garmin.")
                rc = ingest_garmin(local_zip, data_dir, args.tz)
            if rc != 0:
                return rc
            primary_done = True

    # Process Strava exports
    for local_zip in strava_exports:
        is_supplementary = primary_done
        print(f"\n{'='*60}")
        print(f"Processing {'SUPPLEMENTARY' if is_supplementary else 'PRIMARY'} Strava export: {os.path.basename(local_zip)}")
        print(f"{'='*60}")
        rc = ingest_strava(local_zip, data_dir, args.athlete_dir, args.tz,
                          supplementary=is_supplementary)
        if rc != 0:
            return rc
        primary_done = True

    # ── Summary ──────────────────────────────────────────────────────────
    fits_zip_path = os.path.join(data_dir, "fits.zip")
    activities_csv = os.path.join(data_dir, "activities.csv")
    extra_summaries = os.path.join(data_dir, "gpx_tcx_summaries.csv")

    print(f"\n{'='*60}")
    print(f"INITIAL DATA INGEST COMPLETE")
    print(f"{'='*60}")
    if os.path.isfile(fits_zip_path):
        with zipfile.ZipFile(fits_zip_path, "r") as zf:
            n_fits = len([n for n in zf.namelist() if n.lower().endswith(".fit")])
        print(f"  fits.zip:              {n_fits} FIT files")
    if os.path.isfile(activities_csv):
        print(f"  activities.csv:        ✓")
    if os.path.isfile(extra_summaries):
        print(f"  gpx_tcx_summaries.csv: ✓ (TCX-only runs for --extra-summaries)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
