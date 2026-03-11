#!/usr/bin/env python3
"""
scan_polar_sports.py — Scan a Polar JSON export for sport ID distribution.

Usage:
  python scan_polar_sports.py <polar_export_dir>
  python scan_polar_sports.py <polar_export_zip>

Reads all training-session JSON files and reports:
  - Sport ID frequency table (which IDs and how many sessions each)
  - Running vs skipped breakdown using current RUNNING_SPORT_IDS
  - Sample session details for unrecognised sport IDs

This helps identify missing sport IDs (e.g. treadmill) without
needing a full pipeline rebuild.
"""

import json
import os
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime

# Current running sport IDs (must match polar_ingest.py)
RUNNING_SPORT_IDS = {"1", "27", "69", "95"}

KNOWN_SPORT_NAMES = {
    "1": "Running",
    "2": "Cycling",
    "3": "Swimming",
    "5": "Cross-country skiing",
    "11": "Hiking",
    "13": "Strength training",
    "17": "Road cycling",
    "22": "Core training",
    "27": "Trail running / Jogging",
    "29": "Walking",
    "32": "Stretching",
    "39": "Yoga",
    "51": "Nordic walking",
    "62": "Group fitness",
    "68": "Indoor running / Treadmill (V800 era)",
    "69": "Trail running (alternate)",
    "73": "Mountain biking",
    "85": "Indoor running",
    "87": "Indoor cycling",
    "88": "Treadmill",
    "95": "Treadmill running",
    "97": "Functional training",
    "102": "Swim (pool)",
    "103": "Indoor rowing",
    "126": "Mobility / Recovery",
}


def scan_directory(path):
    """Scan a directory of Polar JSON files."""
    sessions = []
    for root, dirs, files in os.walk(path):
        for fname in files:
            if fname.endswith(".json") and "training-session" in root.lower():
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    sessions.append((fname, data))
                except Exception:
                    pass
    return sessions


def scan_zip(zip_path):
    """Scan a zip containing Polar JSON exports."""
    sessions = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".json") and ("training-session" in name.lower() or "/exercises/" in name.lower()):
                try:
                    with zf.open(name) as f:
                        data = json.load(f)
                    sessions.append((name, data))
                except Exception:
                    pass
    return sessions


def analyse(sessions):
    sport_counter = Counter()
    sport_samples = defaultdict(list)  # sport_id -> [(date, name, distance, duration, has_hr, has_samples)]

    for fname, data in sessions:
        sport = data.get("sport", {})
        sport_id = str(sport.get("id", "unknown"))
        sport_counter[sport_id] += 1

        # Gather sample info
        start = data.get("startTime", "")
        name = data.get("name", "") or data.get("note", "") or ""
        distance = data.get("distance", 0) or 0
        duration = data.get("duration", "") or ""
        
        # Check for per-second data
        exercises = data.get("exercises", [])
        has_samples = False
        has_hr = False
        for ex in exercises:
            samples = ex.get("samples", {})
            if samples.get("heartRate") or samples.get("speed"):
                has_samples = True
            if samples.get("heartRate"):
                has_hr = True

        if len(sport_samples[sport_id]) < 5:  # keep up to 5 samples per sport
            sport_samples[sport_id].append({
                "date": start[:10] if start else "?",
                "name": name[:50],
                "distance_km": round(distance / 1000, 1) if distance else 0,
                "duration": duration,
                "has_hr": has_hr,
                "has_samples": has_samples,
            })

    # Report
    print(f"\n{'='*70}")
    print(f"POLAR SPORT ID SCAN — {len(sessions)} total sessions")
    print(f"{'='*70}\n")

    print(f"{'ID':>5s}  {'Count':>6s}  {'In pipeline?':>12s}  {'Known name'}")
    print(f"{'-'*5}  {'-'*6}  {'-'*12}  {'-'*30}")

    for sport_id, count in sorted(sport_counter.items(), key=lambda x: -x[1]):
        in_pipeline = "✓ YES" if sport_id in RUNNING_SPORT_IDS else "✗ NO"
        known = KNOWN_SPORT_NAMES.get(sport_id, "???")
        flag = "  ← CANDIDATE?" if sport_id not in RUNNING_SPORT_IDS and (
            "run" in known.lower() or "treadmill" in known.lower() or "indoor run" in known.lower()
        ) else ""
        print(f"{sport_id:>5s}  {count:>6d}  {in_pipeline:>12s}  {known}{flag}")

    # Show samples for unrecognised running-like sports
    print(f"\n{'='*70}")
    print("CANDIDATE RUNNING SPORTS (not currently in RUNNING_SPORT_IDS)")
    print(f"{'='*70}\n")

    candidates_found = False
    for sport_id in sorted(sport_counter.keys()):
        if sport_id in RUNNING_SPORT_IDS:
            continue
        known = KNOWN_SPORT_NAMES.get(sport_id, "???")
        if "run" in known.lower() or "treadmill" in known.lower() or "indoor" in known.lower() or known == "???":
            candidates_found = True
            print(f"  Sport ID {sport_id} ({known}) — {sport_counter[sport_id]} sessions:")
            for s in sport_samples[sport_id]:
                hr_flag = "HR✓" if s["has_hr"] else "no HR"
                samp_flag = "samples✓" if s["has_samples"] else "no samples"
                print(f"    {s['date']} | {s['distance_km']:5.1f}km | {hr_flag} | {samp_flag} | {s['name']}")
            print()

    if not candidates_found:
        print("  None found — all running-like sports already in RUNNING_SPORT_IDS\n")

    # Summary
    running_count = sum(c for sid, c in sport_counter.items() if sid in RUNNING_SPORT_IDS)
    skipped_count = sum(c for sid, c in sport_counter.items() if sid not in RUNNING_SPORT_IDS)
    print(f"{'='*70}")
    print(f"SUMMARY: {running_count} running sessions, {skipped_count} skipped")
    print(f"{'='*70}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    if os.path.isdir(path):
        sessions = scan_directory(path)
    elif zipfile.is_zipfile(path):
        sessions = scan_zip(path)
    else:
        print(f"Error: {path} is not a directory or zip file")
        sys.exit(1)

    if not sessions:
        print(f"No training-session JSON files found in {path}")
        sys.exit(1)

    analyse(sessions)
