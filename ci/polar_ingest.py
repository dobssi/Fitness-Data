#!/usr/bin/env python3
"""polar_ingest.py — Convert Polar Flow JSON export to pipeline-compatible FIT files.

Reads the Polar data export zip (training-session-*.json files), extracts
per-second data (HR, speed, altitude, distance, GPS, cadence, power),
filters to running activities, and produces:
  - fits.zip: FIT files named YYYY-MM-DD_HH-MM-SS.FIT
  - activities.csv: activity names/metadata for Strava-style matching

Polar JSON structure:
  - training-session-*.json: per-exercise data with 1Hz samples
  - activity-*.json: daily activity summaries (ignored)
  - training-target-*.json: planned workouts (ignored)

Power handling:
  Polar records Stryd power as LEFT_CRANK_CURRENT_POWER (per-foot via BLE
  cycling power meter profile). Polar's summary doubles this to get total.
  Polar also has native wrist-based running power on Vantage V/V2/V3 which
  uses the same channel. We store the raw per-second value and let the
  pipeline decide how to use it (GAP mode ignores power entirely).

  The --power-scale flag (default 2) controls the multiplier applied to
  LEFT_CRANK_CURRENT_POWER to get total watts.

Usage:
    python polar_ingest.py \\
        --polar-zip /path/to/polar_export.zip \\
        --out-dir /path/to/output \\
        --tz Europe/Stockholm

    Produces:
        output/fits.zip          — FIT files for rebuild_from_fit_zip.py
        output/activities.csv    — activity names/metadata

Dependencies:
    pip install fitparse numpy
    (fitparse is only needed for FIT writing via the fit_tool approach,
     but we use a minimal FIT writer instead)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import struct
import sys
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# ─── Polar sport ID mapping ─────────────────────────────────────────────────
# Polar uses numeric sport IDs. These are the running-related ones.
# We accept all of these and let rebuild_from_fit_zip filter further if needed.
RUNNING_SPORT_IDS = {
    "1",    # Running
    "27",   # Trail running / Jogging (seen in data)
    "69",   # Trail running (alternate)
    "95",   # Treadmill running
}

# Sport IDs we explicitly skip (common non-running)
# Everything not in RUNNING_SPORT_IDS is skipped by default.


# ─── Minimal FIT file writer ────────────────────────────────────────────────
# We write valid FIT files with just enough structure for rebuild_from_fit_zip.py
# to parse: file_id, session, and record messages.

# FIT epoch: 1989-12-31 00:00:00 UTC
FIT_EPOCH = datetime(1989, 12, 31, 0, 0, 0, tzinfo=timezone.utc)

# FIT message types
MESG_FILE_ID = 0
MESG_SESSION = 18
MESG_RECORD = 20
MESG_LAP = 19
MESG_EVENT = 21
MESG_DEVICE_INFO = 23

# FIT base types (per FIT SDK specification)
FIT_ENUM = 0
FIT_SINT8 = 1
FIT_UINT8 = 2
FIT_SINT16 = 3
FIT_UINT16 = 4
FIT_SINT32 = 5
FIT_UINT32 = 6
FIT_STRING = 7
FIT_UINT32Z = 12


def _fit_timestamp(dt: datetime) -> int:
    """Convert datetime to FIT timestamp (seconds since FIT epoch)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int((dt - FIT_EPOCH).total_seconds())


def _crc16(data: bytes) -> int:
    """Calculate FIT CRC-16."""
    crc_table = [
        0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
        0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400,
    ]
    crc = 0
    for byte in data:
        tmp = crc_table[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc = crc ^ tmp ^ crc_table[byte & 0xF]
        tmp = crc_table[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc = crc ^ tmp ^ crc_table[(byte >> 4) & 0xF]
    return crc


class FITWriter:
    """Minimal FIT file writer for running activities."""

    def __init__(self):
        self._definitions = {}
        self._data = bytearray()

    def _write_definition(self, local_mesg: int, global_mesg: int,
                          fields: list[tuple[int, int, int]]):
        """Write a definition message.
        fields: list of (field_def_num, size_bytes, base_type)
        """
        # Definition message header: bit 6 set
        header = 0x40 | (local_mesg & 0x0F)
        self._data.append(header)
        self._data.append(0)  # reserved
        self._data.append(0)  # architecture: little-endian
        self._data.extend(struct.pack('<H', global_mesg))
        self._data.append(len(fields))
        for fdn, size, btype in fields:
            self._data.append(fdn)
            self._data.append(size)
            self._data.append(btype)
        self._definitions[local_mesg] = fields

    def _write_data(self, local_mesg: int, values: list[bytes]):
        """Write a data message."""
        header = local_mesg & 0x0F
        self._data.append(header)
        for v in values:
            self._data.extend(v)

    def add_file_id(self, timestamp: datetime):
        """Write file_id message (required first message)."""
        fields = [
            (0, 1, FIT_ENUM),     # type: activity
            (1, 2, FIT_UINT16),   # manufacturer
            (2, 2, FIT_UINT16),   # product
            (3, 4, FIT_UINT32Z),  # serial_number
            (4, 4, FIT_UINT32),   # time_created
        ]
        self._write_definition(0, MESG_FILE_ID, fields)
        self._write_data(0, [
            struct.pack('<B', 4),          # type = activity
            struct.pack('<H', 123),        # manufacturer = Polar (arbitrary)
            struct.pack('<H', 0),          # product
            struct.pack('<I', 0),          # serial_number
            struct.pack('<I', _fit_timestamp(timestamp)),
        ])

    def add_session(self, start_time: datetime, total_elapsed_s: float,
                    total_distance_m: float, avg_hr: int, max_hr: int,
                    avg_speed_ms: float, sport: int = 1):
        """Write session message."""
        fields = [
            (253, 4, FIT_UINT32),  # timestamp
            (2, 4, FIT_UINT32),    # start_time
            (7, 4, FIT_UINT32),    # total_elapsed_time (ms, stored as 1/1000)
            (8, 4, FIT_UINT32),    # total_timer_time (ms)
            (9, 4, FIT_UINT32),    # total_distance (cm, stored as 1/100)
            (5, 1, FIT_ENUM),      # sport
            (16, 1, FIT_UINT8),    # avg_heart_rate
            (17, 1, FIT_UINT8),    # max_heart_rate
            (14, 2, FIT_UINT16),   # avg_speed (mm/s, stored as 1/1000)
        ]
        self._write_definition(1, MESG_SESSION, fields)

        elapsed_ms = int(total_elapsed_s * 1000)
        dist_cm = int(total_distance_m * 100)
        end_time = start_time + timedelta(seconds=total_elapsed_s)

        self._write_data(1, [
            struct.pack('<I', _fit_timestamp(end_time)),
            struct.pack('<I', _fit_timestamp(start_time)),
            struct.pack('<I', elapsed_ms),
            struct.pack('<I', elapsed_ms),
            struct.pack('<I', dist_cm),
            struct.pack('<B', sport),  # 1 = running
            struct.pack('<B', min(255, max(0, avg_hr))),
            struct.pack('<B', min(255, max(0, max_hr))),
            struct.pack('<H', int(avg_speed_ms * 1000)),
        ])

    def add_record(self, timestamp: datetime, lat: float | None = None,
                   lon: float | None = None, altitude_m: float | None = None,
                   hr_bpm: int | None = None, speed_ms: float | None = None,
                   distance_m: float | None = None, cadence: int | None = None,
                   power_w: int | None = None):
        """Write a record (per-second data point)."""
        # Build field list dynamically based on what data is available
        fields = [(253, 4, FIT_UINT32)]  # timestamp always present
        values = [struct.pack('<I', _fit_timestamp(timestamp))]

        if lat is not None and lon is not None:
            # FIT stores lat/lon as semicircles: degrees * (2^31 / 180)
            fields.append((0, 4, FIT_SINT32))  # position_lat
            fields.append((1, 4, FIT_SINT32))  # position_long
            lat_semi = int(lat * (2**31 / 180.0))
            lon_semi = int(lon * (2**31 / 180.0))
            values.append(struct.pack('<i', lat_semi))
            values.append(struct.pack('<i', lon_semi))

        if altitude_m is not None and np.isfinite(altitude_m):
            # FIT altitude: (value + 500) * 5, stored as uint16
            fields.append((2, 2, FIT_UINT16))
            alt_fit = int((altitude_m + 500.0) * 5.0)
            alt_fit = max(0, min(65534, alt_fit))
            values.append(struct.pack('<H', alt_fit))

        if hr_bpm is not None and hr_bpm > 0:
            fields.append((3, 1, FIT_UINT8))
            values.append(struct.pack('<B', min(255, max(0, int(hr_bpm)))))

        if speed_ms is not None and np.isfinite(speed_ms) and speed_ms >= 0:
            # FIT speed: m/s * 1000, stored as uint16
            fields.append((6, 2, FIT_UINT16))
            values.append(struct.pack('<H', min(65534, int(speed_ms * 1000))))

        if distance_m is not None and np.isfinite(distance_m):
            # FIT distance: metres * 100, stored as uint32
            fields.append((5, 4, FIT_UINT32))
            values.append(struct.pack('<I', int(distance_m * 100)))

        if cadence is not None and cadence > 0:
            # FIT cadence for running: strides/min (Polar reports steps/min)
            fields.append((4, 1, FIT_UINT8))
            values.append(struct.pack('<B', min(255, max(0, int(cadence)))))

        if power_w is not None and power_w > 0:
            fields.append((7, 2, FIT_UINT16))
            values.append(struct.pack('<H', min(65534, max(0, int(power_w)))))

        # Use local message 2 for records, redefine each time
        # (simpler than tracking field changes)
        self._write_definition(2, MESG_RECORD, fields)
        self._write_data(2, values)

    def build(self) -> bytes:
        """Build the complete FIT file with header and CRC."""
        data_bytes = bytes(self._data)
        data_size = len(data_bytes)

        # 14-byte header
        header = struct.pack('<BBHI4s',
                             14,           # header size
                             0x20,         # protocol version 2.0
                             0x0000,       # profile version
                             data_size,    # data size
                             b'.FIT')      # data type
        header_crc = _crc16(header[:12])
        header = header + struct.pack('<H', header_crc)

        # File CRC
        file_crc = _crc16(header + data_bytes)

        return header + data_bytes + struct.pack('<H', file_crc)


# ─── Polar JSON parsing ─────────────────────────────────────────────────────

def parse_polar_session(data: dict, tz_str: str = "UTC",
                        power_scale: int = 2) -> dict | None:
    """Parse a Polar training-session JSON into a structured dict.

    Returns None if the session has no usable per-second data.
    """
    try:
        import pytz
        local_tz = pytz.timezone(tz_str)
    except ImportError:
        from zoneinfo import ZoneInfo
        local_tz = ZoneInfo(tz_str)

    sport_id = str(data.get("sport", {}).get("id", ""))
    if sport_id not in RUNNING_SPORT_IDS:
        return None

    start_str = data.get("startTime", "")
    if not start_str:
        return None

    # Parse start time (local time, no timezone in Polar JSON)
    try:
        start_local = datetime.fromisoformat(start_str)
    except (ValueError, TypeError):
        return None

    # Assume startTime is in the athlete's local timezone
    if start_local.tzinfo is None:
        try:
            start_local = start_local.replace(tzinfo=local_tz)
        except Exception:
            start_local = start_local.replace(tzinfo=timezone.utc)

    start_utc = start_local.astimezone(timezone.utc)

    exercises = data.get("exercises", [])
    if not exercises:
        return None

    ex = exercises[0]
    samples_container = ex.get("samples", {})
    if isinstance(samples_container, dict):
        sample_list = samples_container.get("samples", [])
    else:
        sample_list = []

    if not sample_list:
        return None

    # Extract per-second arrays
    arrays = {}
    for s in sample_list:
        stype = s.get("type", "")
        interval_ms = s.get("intervalMillis", 1000)
        values = s.get("values", [])
        if values:
            # Convert NaN strings to actual NaN
            arr = []
            for v in values:
                if isinstance(v, str) and v.lower() == "nan":
                    arr.append(np.nan)
                else:
                    try:
                        arr.append(float(v))
                    except (ValueError, TypeError):
                        arr.append(np.nan)
            arrays[stype] = {
                "values": np.array(arr, dtype=np.float64),
                "interval_ms": interval_ms
            }

    # Must have at least HR or speed
    has_hr = "HEART_RATE" in arrays
    has_speed = "SPEED" in arrays
    if not (has_hr or has_speed):
        return None

    # Determine number of seconds from the longest array
    n_samples = max(len(a["values"]) for a in arrays.values())
    if n_samples < 30:
        return None

    # Build per-second arrays (all at 1Hz assumed)
    hr = arrays.get("HEART_RATE", {}).get("values", np.full(n_samples, np.nan))
    speed_kmh = arrays.get("SPEED", {}).get("values", np.full(n_samples, np.nan))
    altitude = arrays.get("ALTITUDE", {}).get("values", np.full(n_samples, np.nan))
    distance = arrays.get("DISTANCE", {}).get("values", np.full(n_samples, np.nan))
    cadence = arrays.get("CADENCE", {}).get("values", np.full(n_samples, np.nan))
    temperature = arrays.get("TEMPERATURE", {}).get("values", np.full(n_samples, np.nan))

    # Power: scale by power_scale (default 2x for Polar's half-power recording)
    raw_power = arrays.get("LEFT_CRANK_CURRENT_POWER", {}).get("values",
                           np.full(n_samples, np.nan))
    power = raw_power * power_scale

    # Speed: convert km/h to m/s
    speed_ms = speed_kmh / 3.6

    # GPS waypoints (separate from samples — different structure)
    routes_data = ex.get("routes", {})
    waypoints = []
    if isinstance(routes_data, dict):
        route = routes_data.get("route", {})
        if isinstance(route, dict):
            waypoints = route.get("wayPoints", [])
        elif isinstance(route, list):
            waypoints = route

    # Build GPS arrays aligned to per-second timestamps
    lat_arr = np.full(n_samples, np.nan)
    lon_arr = np.full(n_samples, np.nan)
    if waypoints:
        for wp in waypoints:
            elapsed_ms = wp.get("elapsedMillis", None)
            lat = wp.get("latitude", None)
            lon = wp.get("longitude", None)
            if elapsed_ms is not None and lat is not None and lon is not None:
                idx = int(elapsed_ms / 1000)
                if 0 <= idx < n_samples:
                    lat_arr[idx] = lat
                    lon_arr[idx] = lon

        # Interpolate GPS gaps (waypoints may be sparse)
        for arr in [lat_arr, lon_arr]:
            valid = np.isfinite(arr)
            if valid.sum() >= 2:
                indices = np.arange(n_samples)
                arr[~valid] = np.interp(indices[~valid], indices[valid], arr[valid])

    # Summary stats
    name = data.get("name", "") or ""
    note = data.get("note", "") or ""
    distance_m = data.get("distanceMeters", 0) or 0
    duration_ms = data.get("durationMillis", 0) or 0
    hr_avg = data.get("hrAvg", 0) or 0
    hr_max = data.get("hrMax", 0) or 0
    product = data.get("product", {}).get("modelName", "")

    return {
        "start_utc": start_utc,
        "start_local": start_local,
        "name": name,
        "note": note,
        "sport_id": sport_id,
        "distance_m": float(distance_m),
        "duration_s": float(duration_ms) / 1000.0,
        "hr_avg": int(hr_avg),
        "hr_max": int(hr_max),
        "product": product,
        "n_samples": n_samples,
        # Per-second arrays
        "hr": hr[:n_samples],
        "speed_ms": speed_ms[:n_samples],
        "altitude": altitude[:n_samples],
        "distance": distance[:n_samples],
        "cadence": cadence[:n_samples],
        "power": power[:n_samples],
        "temperature": temperature[:n_samples],
        "lat": lat_arr[:n_samples],
        "lon": lon_arr[:n_samples],
    }


def session_to_fit(session: dict) -> bytes:
    """Convert a parsed Polar session to FIT file bytes."""
    writer = FITWriter()

    start_utc = session["start_utc"]
    writer.add_file_id(start_utc)

    # Session summary
    avg_speed = np.nanmean(session["speed_ms"][session["speed_ms"] > 0.3]) \
        if np.any(session["speed_ms"] > 0.3) else 0.0
    writer.add_session(
        start_time=start_utc,
        total_elapsed_s=session["duration_s"],
        total_distance_m=session["distance_m"],
        avg_hr=session["hr_avg"],
        max_hr=session["hr_max"],
        avg_speed_ms=float(avg_speed),
        sport=1,  # running
    )

    # Per-second records
    n = session["n_samples"]
    for i in range(n):
        ts = start_utc + timedelta(seconds=i)

        lat = session["lat"][i] if np.isfinite(session["lat"][i]) else None
        lon = session["lon"][i] if np.isfinite(session["lon"][i]) else None
        alt = session["altitude"][i] if np.isfinite(session["altitude"][i]) else None
        hr = int(session["hr"][i]) if np.isfinite(session["hr"][i]) and session["hr"][i] > 0 else None
        spd = session["speed_ms"][i] if np.isfinite(session["speed_ms"][i]) else None
        dist = session["distance"][i] if np.isfinite(session["distance"][i]) else None
        cad = int(session["cadence"][i]) if np.isfinite(session["cadence"][i]) and session["cadence"][i] > 0 else None
        pwr = int(session["power"][i]) if np.isfinite(session["power"][i]) and session["power"][i] > 0 else None

        writer.add_record(
            timestamp=ts,
            lat=lat, lon=lon,
            altitude_m=alt,
            hr_bpm=hr,
            speed_ms=spd,
            distance_m=dist,
            cadence=cad,
            power_w=pwr,
        )

    return writer.build()


# ─── Main ingestion ─────────────────────────────────────────────────────────

def ingest_polar_zip(polar_zip_path: str, out_dir: str, tz_str: str = "UTC",
                     power_scale: int = 2, verbose: bool = True) -> dict:
    """Process a Polar export zip and produce fits.zip + activities.csv.

    Returns summary dict with counts.
    """
    os.makedirs(out_dir, exist_ok=True)
    fits_zip_path = os.path.join(out_dir, "fits.zip")
    activities_csv_path = os.path.join(out_dir, "activities.csv")

    sessions = []
    skipped_sport = 0
    skipped_no_data = 0
    errors = 0

    with zipfile.ZipFile(polar_zip_path, "r") as zin:
        names = [n for n in zin.namelist() if "training-session-" in n and n.endswith(".json")]
        if verbose:
            print(f"\nFound {len(names)} training-session files in {polar_zip_path}")

        for i, name in enumerate(sorted(names)):
            try:
                raw = zin.read(name)
                data = json.loads(raw)
            except Exception as e:
                if verbose and errors < 5:
                    print(f"  [ERROR] {name}: {e}")
                errors += 1
                continue

            session = parse_polar_session(data, tz_str=tz_str, power_scale=power_scale)
            if session is None:
                sport_id = str(data.get("sport", {}).get("id", ""))
                if sport_id not in RUNNING_SPORT_IDS:
                    skipped_sport += 1
                else:
                    skipped_no_data += 1
                continue

            sessions.append(session)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Parsed {i + 1}/{len(names)}...")

    if verbose:
        print(f"\n  Running sessions: {len(sessions)}")
        print(f"  Skipped (non-running): {skipped_sport}")
        print(f"  Skipped (no per-sec data): {skipped_no_data}")
        print(f"  Errors: {errors}")

    # Sort by start time
    sessions.sort(key=lambda s: s["start_utc"])

    # Write FIT files to zip
    if verbose:
        print(f"\n  Converting {len(sessions)} sessions to FIT...")

    seen_names = set()
    fit_count = 0
    activities_rows = []

    with zipfile.ZipFile(fits_zip_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for session in sessions:
            local_dt = session["start_local"]
            fit_name = local_dt.strftime("%Y-%m-%d_%H-%M-%S") + ".FIT"

            # Handle duplicate timestamps
            if fit_name in seen_names:
                for suffix in range(2, 10):
                    candidate = local_dt.strftime("%Y-%m-%d_%H-%M-%S") + f"_{suffix}.FIT"
                    if candidate not in seen_names:
                        fit_name = candidate
                        break
            seen_names.add(fit_name)

            try:
                fit_bytes = session_to_fit(session)
                zout.writestr(fit_name, fit_bytes)
                fit_count += 1

                # Activity metadata
                display_name = session["name"] or session["note"] or ""
                activities_rows.append({
                    "timestamp": local_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "name": display_name,
                    "type": "Run",
                    "distance_km": round(session["distance_m"] / 1000, 2),
                    "duration_s": int(session["duration_s"]),
                    "avg_hr": session["hr_avg"],
                    "product": session["product"],
                    "fit_filename": fit_name,
                    "has_power": bool(np.any(np.isfinite(session["power"]) & (session["power"] > 0))),
                    "has_gps": bool(np.any(np.isfinite(session["lat"]))),
                    "n_samples": session["n_samples"],
                })
            except Exception as e:
                if verbose:
                    print(f"  [ERROR] FIT conversion failed for {local_dt}: {e}")
                errors += 1

    if verbose:
        print(f"  Written: {fits_zip_path} ({fit_count} FIT files)")

    # Write activities.csv (Strava-compatible columns for name matching)
    with open(activities_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Activity Date", "Activity Name", "Activity Type",
            "Distance", "Elapsed Time", "Average Heart Rate",
            # Extra columns for our use
            "product", "fit_filename", "has_power", "has_gps", "n_samples",
        ])
        writer.writeheader()
        for row in activities_rows:
            writer.writerow({
                "Activity Date": row["timestamp"],
                "Activity Name": row["name"],
                "Activity Type": "Run",
                "Distance": row["distance_km"],
                "Elapsed Time": row["duration_s"],
                "Average Heart Rate": row["avg_hr"],
                "product": row["product"],
                "fit_filename": row["fit_filename"],
                "has_power": row["has_power"],
                "has_gps": row["has_gps"],
                "n_samples": row["n_samples"],
            })

    if verbose:
        print(f"  Written: {activities_csv_path} ({len(activities_rows)} entries)")

        # Summary stats
        has_power = sum(1 for r in activities_rows if r["has_power"])
        has_gps = sum(1 for r in activities_rows if r["has_gps"])
        products = {}
        for r in activities_rows:
            p = r["product"] or "unknown"
            products[p] = products.get(p, 0) + 1
        print(f"\n  With power: {has_power}/{len(activities_rows)}")
        print(f"  With GPS: {has_gps}/{len(activities_rows)}")
        print(f"  Products: {products}")

        if activities_rows:
            first = activities_rows[0]["timestamp"]
            last = activities_rows[-1]["timestamp"]
            print(f"  Date range: {first} → {last}")

    return {
        "total_sessions": len(sessions),
        "fit_files": fit_count,
        "skipped_sport": skipped_sport,
        "skipped_no_data": skipped_no_data,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert Polar Flow JSON export to pipeline-compatible FIT files"
    )
    parser.add_argument("--polar-zip", required=True,
                        help="Path to Polar export zip file")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for fits.zip and activities.csv")
    parser.add_argument("--tz", default="Europe/Stockholm",
                        help="Athlete timezone (default: Europe/Stockholm)")
    parser.add_argument("--power-scale", type=int, default=2,
                        help="Multiplier for LEFT_CRANK_CURRENT_POWER (default: 2)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    result = ingest_polar_zip(
        polar_zip_path=args.polar_zip,
        out_dir=args.out_dir,
        tz_str=args.tz,
        power_scale=args.power_scale,
        verbose=not args.quiet,
    )

    print(f"\nDone. {result['fit_files']} FIT files produced.")
    return 0 if result["fit_files"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
