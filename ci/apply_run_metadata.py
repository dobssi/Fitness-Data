#!/usr/bin/env python3
"""
apply_run_metadata.py — Apply run metadata from GitHub Actions dispatch inputs.

Called before the pipeline to inject run name and override data for the most
recent activity.  Designed to be triggered from the iOS GitHub app after a run.

Usage:
    python ci/apply_run_metadata.py \
        --run-name "Haga parkrun" \
        --race \
        --parkrun \
        --distance 5.0 \
        --surface-adj 0.97 \
        --surface TRAIL \
        --shoe "Nike Vaporfly" \
        --date 2026-02-10 \
        --notes "Muddy conditions" \
        --override-file activity_overrides.xlsx \
        --pending-file pending_activities.csv

All flags are optional. If none are provided, the script does nothing.

--date defaults to today (Stockholm time) if any override flags are set but
no date is given.  This covers the common case: you finish a run, open the
GitHub app, and trigger UPDATE with metadata — all on the same day.

The script:
  1. Appends a row to activity_overrides.xlsx (if any of: race, parkrun,
     distance, surface-adj, surface, notes are provided)
  2. Appends a row to pending_activities.csv (if run-name or shoe provided)
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Apply run metadata from dispatch inputs")
    ap.add_argument("--run-name", default="", help="Activity name (e.g. 'Haga parkrun')")
    ap.add_argument("--race", action="store_true", help="Flag as race")
    ap.add_argument("--parkrun", action="store_true", help="Flag as parkrun")
    ap.add_argument("--distance", type=float, default=None, help="Official distance in km")
    ap.add_argument("--surface-adj", type=float, default=None, help="Surface adjustment factor")
    ap.add_argument("--surface", default="", help="Surface type: TRAIL, TRACK, INDOOR_TRACK, SNOW, HEAVY_SNOW")
    ap.add_argument("--shoe", default="", help="Shoe name")
    ap.add_argument("--temp", type=float, default=None, help="Temperature override (°C)")
    ap.add_argument("--notes", default="", help="Notes")
    ap.add_argument("--date", default="", help="Run date YYYY-MM-DD (default: today Stockholm)")
    ap.add_argument("--run-number", type=int, default=None,
                    help="Which run that day if multiple (1=first, 2=second)")
    ap.add_argument("--override-file", default="activity_overrides.xlsx")
    ap.add_argument("--pending-file", default="pending_activities.csv")
    args = ap.parse_args()

    # --- Determine if there's anything to do ---
    has_override = any([
        args.race, args.parkrun, args.distance is not None,
        args.surface_adj is not None, args.surface, args.temp is not None,
        args.notes,
    ])
    has_pending = bool(args.run_name.strip() or args.shoe.strip())

    if not has_override and not has_pending:
        print("[run-metadata] No metadata provided — skipping.")
        return

    # --- Resolve date ---
    if args.date:
        date_str = args.date.strip()
    else:
        # Default to today in Stockholm time
        try:
            from zoneinfo import ZoneInfo
            date_str = datetime.now(ZoneInfo("Europe/Stockholm")).strftime("%Y-%m-%d")
        except ImportError:
            # Fallback: UTC+1
            from datetime import timezone, timedelta
            date_str = datetime.now(timezone(timedelta(hours=1))).strftime("%Y-%m-%d")

    # File identifier for override: "2026-02-10" or "2026-02-10 #2"
    file_id = date_str
    if args.run_number is not None:
        file_id = f"{date_str} #{args.run_number}"

    print(f"[run-metadata] Date: {file_id}")

    # --- Apply overrides ---
    if has_override:
        _apply_override(args, file_id)

    # --- Apply pending activity name ---
    if has_pending:
        _apply_pending(args, file_id)

    print("[run-metadata] Done.")


def _apply_override(args, file_id: str):
    """Append a row to activity_overrides.xlsx."""
    xlsx_path = args.override_file

    # Build the new row
    new_row = {
        "file": file_id,
        "race_flag": 1 if args.race else 0,
        "parkrun": 1 if args.parkrun else 0,
        "official_distance_km": args.distance if args.distance is not None else np.nan,
        "surface_adj": args.surface_adj if args.surface_adj is not None else np.nan,
        "surface": args.surface.strip().upper() if args.surface else "",
        "temp_override": args.temp if args.temp is not None else np.nan,
        "notes": args.notes.strip(),
    }

    if Path(xlsx_path).exists():
        try:
            df = pd.read_excel(xlsx_path, engine="openpyxl", dtype={"file": str})
        except Exception as e:
            print(f"[run-metadata] Warning: Could not read {xlsx_path}: {e}")
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    # Ensure all columns exist
    for col, val in new_row.items():
        if col not in df.columns:
            df[col] = np.nan if isinstance(val, float) else ""

    # Check for duplicate — same file_id already present
    if len(df) > 0 and "file" in df.columns:
        existing = df["file"].astype(str).str.strip()
        if (existing == file_id).any():
            print(f"[run-metadata] Override for '{file_id}' already exists — updating.")
            mask = existing == file_id
            for col, val in new_row.items():
                if col == "file":
                    continue
                # Only overwrite if we have a meaningful value
                if pd.notna(val) and val != "" and val != 0:
                    df.loc[mask, col] = val
        else:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    # Set 'file' column width
    from openpyxl import load_workbook
    wb = load_workbook(xlsx_path)
    wb.active.column_dimensions['A'].width = 25
    wb.save(xlsx_path)
    summary_parts = []
    if args.race:
        summary_parts.append("race")
    if args.parkrun:
        summary_parts.append("parkrun")
    if args.distance is not None:
        summary_parts.append(f"{args.distance}km")
    if args.surface_adj is not None:
        summary_parts.append(f"sadj={args.surface_adj}")
    if args.surface:
        summary_parts.append(f"surface={args.surface}")
    if args.temp is not None:
        summary_parts.append(f"temp={args.temp}°C")
    if args.notes:
        summary_parts.append(f'"{args.notes}"')
    print(f"[run-metadata] Override → {xlsx_path}: {file_id} ({', '.join(summary_parts)})")


def _apply_pending(args, file_id: str):
    """Append a row to pending_activities.csv."""
    csv_path = args.pending_file

    new_row = {
        "file": file_id,
        "activity_name": args.run_name.strip(),
        "shoe": args.shoe.strip(),
    }

    if Path(csv_path).exists():
        try:
            df = pd.read_csv(csv_path, dtype=str)
        except Exception:
            df = pd.DataFrame(columns=["file", "activity_name", "shoe"])
    else:
        df = pd.DataFrame(columns=["file", "activity_name", "shoe"])

    # Check for duplicate
    if len(df) > 0 and "file" in df.columns:
        existing = df["file"].astype(str).str.strip()
        if (existing == file_id).any():
            print(f"[run-metadata] Pending name for '{file_id}' already exists — updating.")
            mask = existing == file_id
            if args.run_name.strip():
                df.loc[mask, "activity_name"] = args.run_name.strip()
            if args.shoe.strip():
                df.loc[mask, "shoe"] = args.shoe.strip()
        else:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(csv_path, index=False)
    parts = []
    if args.run_name.strip():
        parts.append(f'name="{args.run_name.strip()}"')
    if args.shoe.strip():
        parts.append(f'shoe="{args.shoe.strip()}"')
    print(f"[run-metadata] Pending → {csv_path}: {file_id} ({', '.join(parts)})")


if __name__ == "__main__":
    main()
