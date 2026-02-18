"""
unpack_strava_export.py — Extract running FIT files from a Strava bulk export.

Strava bulk exports contain:
  - activities/ folder with .fit.gz, .gpx.gz, .tcx.gz files
  - activities.csv with summary data for all activities

This script:
  1. Extracts all .fit.gz files → .fit in the output folder
  2. Copies activities.csv (filtered to runs only)
  3. Reports how many activities are FIT vs GPX/TCX (for pipeline compatibility)
  4. Optionally zips the extracted FIT files for rebuild_from_fit_zip.py

Usage:
    python unpack_strava_export.py export_12345.zip --out-dir ./data/
    
    This creates:
      ./data/fits/          — FIT files (uncompressed)
      ./data/fits.zip       — FIT files zipped (for rebuild_from_fit_zip.py)
      ./data/activities.csv — Strava activities summary (runs only)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path


def unpack_strava_export(
    export_zip_path: str,
    out_dir: str,
    activity_types: set[str] | None = None,
    verbose: bool = True
) -> dict:
    """
    Extract FIT files from Strava bulk export.
    
    Args:
        export_zip_path: Path to Strava export ZIP
        out_dir: Output directory
        activity_types: Activity types to include (default: running types)
        verbose: Print progress
        
    Returns:
        dict with counts and paths
    """
    if activity_types is None:
        activity_types = {"Run", "Race", "Trail Run", "Track Run"}
    
    out_path = Path(out_dir)
    fits_dir = out_path / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total_activities": 0,
        "run_activities": 0,
        "fit_extracted": 0,
        "gpx_skipped": 0,
        "tcx_skipped": 0,
        "other_skipped": 0,
        "errors": 0,
        "activities_csv": None,
        "fits_dir": str(fits_dir),
        "fits_zip": None,
    }
    
    # Parse activities.csv to know which files are runs
    run_filenames = set()  # filenames (without path) of running activities
    activities_csv_data = None
    
    with zipfile.ZipFile(export_zip_path, "r") as zf:
        # Find activities.csv
        csv_candidates = [n for n in zf.namelist() if n.endswith("activities.csv")]
        
        if csv_candidates:
            csv_name = csv_candidates[0]
            with zf.open(csv_name) as f:
                text = io.TextIOWrapper(f, encoding="utf-8")
                reader = csv.DictReader(text)
                rows = list(reader)
                stats["total_activities"] = len(rows)
                
                # Filter to runs
                run_rows = []
                for row in rows:
                    act_type = row.get("Activity Type", "").strip()
                    if act_type in activity_types:
                        run_rows.append(row)
                        # Strava uses "Filename" column for the activity file path
                        fn = row.get("Filename", "").strip()
                        if fn:
                            run_filenames.add(fn)
                
                stats["run_activities"] = len(run_rows)
                
                if verbose:
                    print(f"Activities CSV: {stats['total_activities']} total, "
                          f"{stats['run_activities']} runs")
                
                # Save filtered CSV
                if run_rows:
                    csv_out = out_path / "activities.csv"
                    with open(csv_out, "w", newline="", encoding="utf-8") as cf:
                        writer = csv.DictWriter(cf, fieldnames=reader.fieldnames)
                        writer.writeheader()
                        writer.writerows(run_rows)
                    stats["activities_csv"] = str(csv_out)
                    if verbose:
                        print(f"Saved filtered activities.csv: {csv_out}")
        else:
            if verbose:
                print("Warning: No activities.csv found in export")
            # Without CSV, extract ALL activity files
            run_filenames = None  # means "accept all"
        
        # Extract FIT files
        activity_files = [n for n in zf.namelist() 
                         if n.startswith("activities/") and not n.endswith("/")]
        
        for name in activity_files:
            basename = os.path.basename(name)
            
            # If we have activities.csv, filter to runs only
            if run_filenames is not None and name not in run_filenames:
                # Also check without leading path variations
                if not any(name.endswith(rf.lstrip("./")) for rf in run_filenames):
                    continue
            
            # Determine file type
            lower = basename.lower()
            
            if lower.endswith(".fit.gz"):
                # Decompress .fit.gz → .fit
                out_name = basename[:-3]  # remove .gz
                out_file = fits_dir / out_name
                try:
                    with zf.open(name) as gz_in:
                        with gzip.open(gz_in, "rb") as decompressed:
                            with open(out_file, "wb") as f_out:
                                shutil.copyfileobj(decompressed, f_out)
                    stats["fit_extracted"] += 1
                except Exception as e:
                    if verbose:
                        print(f"  Error extracting {basename}: {e}")
                    stats["errors"] += 1
                    
            elif lower.endswith(".fit"):
                # Already uncompressed FIT
                out_file = fits_dir / basename
                try:
                    with zf.open(name) as f_in:
                        with open(out_file, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    stats["fit_extracted"] += 1
                except Exception as e:
                    if verbose:
                        print(f"  Error extracting {basename}: {e}")
                    stats["errors"] += 1
                    
            elif lower.endswith(".gpx.gz") or lower.endswith(".gpx"):
                stats["gpx_skipped"] += 1
                
            elif lower.endswith(".tcx.gz") or lower.endswith(".tcx"):
                stats["tcx_skipped"] += 1
                
            else:
                stats["other_skipped"] += 1
        
        if verbose:
            print(f"\nExtraction complete:")
            print(f"  FIT files extracted: {stats['fit_extracted']}")
            if stats["gpx_skipped"]:
                print(f"  GPX files skipped:   {stats['gpx_skipped']} (not yet supported)")
            if stats["tcx_skipped"]:
                print(f"  TCX files skipped:   {stats['tcx_skipped']} (not yet supported)")
            if stats["errors"]:
                print(f"  Errors:              {stats['errors']}")
    
    # Create fits.zip for rebuild_from_fit_zip.py
    if stats["fit_extracted"] > 0:
        fits_zip = out_path / "fits.zip"
        if verbose:
            print(f"\nCreating {fits_zip}...")
        with zipfile.ZipFile(fits_zip, "w", zipfile.ZIP_DEFLATED) as zout:
            for fit_file in sorted(fits_dir.glob("*.fit")):
                zout.write(fit_file, fit_file.name)
        stats["fits_zip"] = str(fits_zip)
        if verbose:
            print(f"  Created {fits_zip} ({os.path.getsize(fits_zip) / 1024 / 1024:.1f} MB)")
    
    # Summary
    if verbose:
        total_run_files = stats["fit_extracted"] + stats["gpx_skipped"] + stats["tcx_skipped"]
        if total_run_files > 0:
            fit_pct = stats["fit_extracted"] / total_run_files * 100
            print(f"\nFIT coverage: {fit_pct:.0f}% of running activities have FIT files")
            if fit_pct < 90:
                print(f"  {stats['gpx_skipped'] + stats['tcx_skipped']} activities are GPX/TCX only")
                print(f"  GPX/TCX support is planned but not yet implemented")
                print(f"  These activities will be missing from the dashboard")
    
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract running FIT files from a Strava bulk export"
    )
    parser.add_argument(
        "export_zip",
        help="Path to Strava export ZIP file (e.g., export_12345.zip)"
    )
    parser.add_argument(
        "--out-dir",
        default="./data",
        help="Output directory (default: ./data)"
    )
    parser.add_argument(
        "--include-types",
        nargs="+",
        default=["Run", "Race", "Trail Run", "Track Run"],
        help="Activity types to include (default: Run, Race, Trail Run, Track Run)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.export_zip):
        print(f"Error: {args.export_zip} not found")
        return 1
    
    print(f"Unpacking Strava export: {args.export_zip}")
    print(f"Output directory: {args.out_dir}")
    print(f"Activity types: {', '.join(args.include_types)}")
    print()
    
    stats = unpack_strava_export(
        args.export_zip,
        args.out_dir,
        activity_types=set(args.include_types),
        verbose=not args.quiet
    )
    
    if stats["fit_extracted"] == 0:
        print("\n⚠️  No FIT files found! The athlete may need a Garmin Connect export instead.")
        return 1
    
    print(f"\n✓ Ready for pipeline. Next step:")
    print(f"  python rebuild_from_fit_zip.py --fit-zip {stats['fits_zip']} \\")
    print(f"    --template template.xlsx --out {args.out_dir}/Master_FULL.xlsx \\")
    print(f"    --persec-cache-dir {args.out_dir}/persec_cache/")
    if stats["activities_csv"]:
        print(f"    --strava {stats['activities_csv']}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
