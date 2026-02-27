"""
strava_ingest.py - Ingest a Strava bulk export for the running pipeline.

Strava bulk exports contain:
  - activities.csv  (metadata: name, type, date, distance, time, workout type)
  - activities/     (activity files: mix of .fit.gz, .fit, .gpx.gz, .gpx, .tcx.gz, .tcx)

This script:
  1. Extracts the Strava export zip
  2. Parses activities.csv for race flags, names, types
  3. Identifies running activities (filters out cycling, swimming, etc.)
  4. Extracts/decompresses activity files (.fit.gz → .fit, .gpx.gz → .gpx)
  5. Converts GPX files to a FIT-equivalent per-second DataFrame
  6. Writes a consolidated FIT zip (all .fit files) + activities.csv ready for rebuild

Usage:
    python strava_ingest.py --strava-zip "export_12345.zip" --out-dir "./athlete_data"

Output:
    <out-dir>/fits.zip           — All FIT files (original + converted from GPX)
    <out-dir>/activities.csv     — Cleaned activities.csv (runs only)
    <out-dir>/strava_race_overrides.csv — Auto-detected races for activity_overrides

The pipeline then runs:
    python rebuild_from_fit_zip.py --fit-zip <out-dir>/fits.zip --strava <out-dir>/activities.csv ...
"""

import argparse
import csv
import gzip
import io
import os
import shutil
import struct
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


# GPX namespace
GPX_NS = {
    'gpx': 'http://www.topografix.com/GPX/1/1',
    'gpx10': 'http://www.topografix.com/GPX/1/0',
    'gpxtpx': 'http://www.garmin.com/xmlschemas/TrackPointExtension/v1',
    'gpxtpx2': 'http://www.garmin.com/xmlschemas/TrackPointExtension/v2',
    'ns3': 'http://www.garmin.com/xmlschemas/TrackPointExtension/v1',
}

# TCX namespace
TCX_NS = {
    'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2',
    'ns2': 'http://www.garmin.com/xmlschemas/ActivityExtension/v2',
    'ns3': 'http://www.garmin.com/xmlschemas/TrackPointExtension/v1',
}


def parse_gpx_to_df(gpx_path: str) -> pd.DataFrame:
    """
    Parse a GPX file into a per-second DataFrame matching rebuild's format.
    
    Returns DataFrame with columns:
        ts, hr, cadence, power_w, speed, dist_m, elev_m, lat, lon
    
    GPX files from Strava typically contain:
        - <trkpt lat="" lon=""> with <ele>, <time>
        - Extensions with <gpxtpx:TrackPointExtension> containing <gpxtpx:hr>, <gpxtpx:cad>
    """
    tree = ET.parse(gpx_path)
    root = tree.getroot()
    
    # Detect namespace (GPX 1.0 vs 1.1)
    ns_tag = root.tag
    if 'GPX/1/1' in ns_tag:
        ns = 'gpx'
    elif 'GPX/1/0' in ns_tag:
        ns = 'gpx10'
    else:
        # Try to extract namespace from root tag
        ns = 'gpx'
    
    # Build namespace map dynamically from the document
    nsmap = {}
    for prefix, uri in GPX_NS.items():
        nsmap[prefix] = uri
    
    # Also register any namespaces found in the document
    # ElementTree doesn't expose namespace declarations easily,
    # so we use a brute-force search for common extension patterns
    
    rows = []
    
    # Find all trackpoints across all tracks and segments
    # Try GPX 1.1 first, then 1.0, then no-namespace fallback
    trkpts = root.findall(f'.//{{{GPX_NS["gpx"]}}}trkpt')
    if not trkpts:
        trkpts = root.findall(f'.//{{{GPX_NS["gpx10"]}}}trkpt')
    if not trkpts:
        # No-namespace fallback
        trkpts = root.findall('.//trkpt')
    
    prev_lat = prev_lon = prev_ts = None
    cumulative_dist_m = 0.0
    
    for pt in trkpts:
        lat = _float_or_nan(pt.get('lat'))
        lon = _float_or_nan(pt.get('lon'))
        
        # Elevation
        elev = _find_text_float(pt, 'ele', nsmap, ns)
        
        # Timestamp
        time_str = _find_text(pt, 'time', nsmap, ns)
        if time_str is None:
            continue
        try:
            ts_dt = _parse_iso_timestamp(time_str)
            ts = ts_dt.timestamp()
        except (ValueError, TypeError):
            continue
        
        # Heart rate and cadence from extensions
        hr = np.nan
        cad = np.nan
        
        # Search for HR in multiple extension formats
        hr = _find_extension_value(pt, 'hr', nsmap, ns)
        cad = _find_extension_value(pt, 'cad', nsmap, ns)
        
        # Calculate speed from position delta (if we have previous point)
        speed = np.nan
        if prev_lat is not None and prev_ts is not None and ts > prev_ts:
            dist = _haversine_m(prev_lat, prev_lon, lat, lon)
            dt = ts - prev_ts
            if dt > 0 and dt < 30:  # skip gaps > 30s
                speed = dist / dt
                cumulative_dist_m += dist
            elif dt >= 30:
                # Gap — still accumulate approximate distance but don't compute speed
                cumulative_dist_m += dist
        
        rows.append((ts, hr, cad, np.nan, speed, cumulative_dist_m, elev, lat, lon))
        
        prev_lat, prev_lon, prev_ts = lat, lon, ts
    
    if not rows:
        return pd.DataFrame(columns=["ts", "hr", "cadence", "power_w", "speed", "dist_m", "elev_m", "lat", "lon"])
    
    df = pd.DataFrame(rows, columns=["ts", "hr", "cadence", "power_w", "speed", "dist_m", "elev_m", "lat", "lon"])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    
    # Fill speed for first point
    if len(df) > 1 and np.isnan(df.at[0, 'speed']):
        df.at[0, 'speed'] = df.at[1, 'speed'] if not np.isnan(df.at[1, 'speed']) else 0.0
    
    return df


def parse_tcx_to_df(tcx_path: str) -> pd.DataFrame:
    """
    Parse a TCX file into a per-second DataFrame matching rebuild's format.
    
    TCX files contain:
        - <Trackpoint> with <Time>, <Position>/<LatitudeDegrees>/<LongitudeDegrees>,
          <AltitudeMeters>, <DistanceMeters>, <HeartRateBpm>/<Value>, <Cadence>
    """
    tree = ET.parse(tcx_path)
    root = tree.getroot()
    
    rows = []
    prev_lat = prev_lon = prev_ts = None
    
    # Find all trackpoints
    trkpts = root.findall(f'.//{{{TCX_NS["tcx"]}}}Trackpoint')
    if not trkpts:
        # No-namespace fallback
        trkpts = root.findall('.//Trackpoint')
    
    for pt in trkpts:
        # Timestamp
        time_el = pt.find(f'{{{TCX_NS["tcx"]}}}Time')
        if time_el is None:
            time_el = pt.find('Time')
        if time_el is None or time_el.text is None:
            continue
        try:
            ts_dt = _parse_iso_timestamp(time_el.text)
            ts = ts_dt.timestamp()
        except (ValueError, TypeError):
            continue
        
        # Position
        lat = lon = np.nan
        pos = pt.find(f'{{{TCX_NS["tcx"]}}}Position')
        if pos is None:
            pos = pt.find('Position')
        if pos is not None:
            lat_el = pos.find(f'{{{TCX_NS["tcx"]}}}LatitudeDegrees')
            lon_el = pos.find(f'{{{TCX_NS["tcx"]}}}LongitudeDegrees')
            if lat_el is None:
                lat_el = pos.find('LatitudeDegrees')
            if lon_el is None:
                lon_el = pos.find('LongitudeDegrees')
            if lat_el is not None and lat_el.text:
                lat = _float_or_nan(lat_el.text)
            if lon_el is not None and lon_el.text:
                lon = _float_or_nan(lon_el.text)
        
        # Altitude
        elev = _tcx_float(pt, 'AltitudeMeters')
        
        # Distance
        dist_m = _tcx_float(pt, 'DistanceMeters')
        
        # Heart rate
        hr = np.nan
        hr_el = pt.find(f'{{{TCX_NS["tcx"]}}}HeartRateBpm')
        if hr_el is None:
            hr_el = pt.find('HeartRateBpm')
        if hr_el is not None:
            val_el = hr_el.find(f'{{{TCX_NS["tcx"]}}}Value')
            if val_el is None:
                val_el = hr_el.find('Value')
            if val_el is not None and val_el.text:
                hr = _float_or_nan(val_el.text)
        
        # Cadence
        cad = _tcx_float(pt, 'Cadence')
        # Also check RunCadence in extensions
        if np.isnan(cad):
            cad = _tcx_extension_float(pt, 'RunCadence')
        
        # Speed from position delta
        speed = np.nan
        if prev_lat is not None and prev_ts is not None and ts > prev_ts:
            d = _haversine_m(prev_lat, prev_lon, lat, lon)
            dt = ts - prev_ts
            if 0 < dt < 30:
                speed = d / dt
        
        rows.append((ts, hr, cad, np.nan, speed, dist_m, elev, lat, lon))
        prev_lat, prev_lon, prev_ts = lat, lon, ts
    
    if not rows:
        return pd.DataFrame(columns=["ts", "hr", "cadence", "power_w", "speed", "dist_m", "elev_m", "lat", "lon"])
    
    df = pd.DataFrame(rows, columns=["ts", "hr", "cadence", "power_w", "speed", "dist_m", "elev_m", "lat", "lon"])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    
    return df


# ---------------------------------------------------------------------------
# XML helper functions
# ---------------------------------------------------------------------------

def _float_or_nan(val):
    """Convert string to float, returning NaN on failure."""
    if val is None:
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def _find_text(el, tag, nsmap, ns):
    """Find text of a child element, trying namespaced then bare."""
    child = el.find(f'{{{nsmap[ns]}}}{tag}')
    if child is not None and child.text:
        return child.text.strip()
    # Bare fallback
    child = el.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return None


def _find_text_float(el, tag, nsmap, ns):
    """Find float value of a child element."""
    txt = _find_text(el, tag, nsmap, ns)
    return _float_or_nan(txt)


def _find_extension_value(pt, field_name, nsmap, ns):
    """
    Search for a value in GPX track point extensions.
    
    Tries multiple namespace patterns used by Garmin/Strava exports:
        <extensions><gpxtpx:TrackPointExtension><gpxtpx:hr>
        <extensions><ns3:TrackPointExtension><ns3:hr>
    """
    val = np.nan
    
    # Find <extensions> element
    ext = pt.find(f'{{{nsmap[ns]}}}extensions')
    if ext is None:
        ext = pt.find('extensions')
    if ext is None:
        return val
    
    # Search in all known extension namespaces
    for ext_ns_key in ('gpxtpx', 'gpxtpx2', 'ns3'):
        ext_ns = nsmap.get(ext_ns_key)
        if ext_ns is None:
            continue
        tpe = ext.find(f'{{{ext_ns}}}TrackPointExtension')
        if tpe is not None:
            el = tpe.find(f'{{{ext_ns}}}{field_name}')
            if el is not None and el.text:
                val = _float_or_nan(el.text)
                if not np.isnan(val):
                    return val
    
    # Brute force: search all children of extensions for the field name
    for child in ext.iter():
        tag = child.tag
        if tag and child.text:
            # Strip namespace
            local_name = tag.split('}')[-1] if '}' in tag else tag
            if local_name.lower() == field_name.lower():
                val = _float_or_nan(child.text)
                if not np.isnan(val):
                    return val
    
    return val


def _tcx_float(pt, tag):
    """Find float value of a TCX child element."""
    el = pt.find(f'{{{TCX_NS["tcx"]}}}{tag}')
    if el is None:
        el = pt.find(tag)
    if el is not None and el.text:
        return _float_or_nan(el.text)
    return np.nan


def _tcx_extension_float(pt, field_name):
    """Find a value in TCX extensions."""
    ext = pt.find(f'{{{TCX_NS["tcx"]}}}Extensions')
    if ext is None:
        ext = pt.find('Extensions')
    if ext is None:
        return np.nan
    for child in ext.iter():
        tag = child.tag
        if tag and child.text:
            local_name = tag.split('}')[-1] if '}' in tag else tag
            if local_name.lower() == field_name.lower():
                return _float_or_nan(child.text)
    return np.nan


def _parse_iso_timestamp(s: str) -> datetime:
    """Parse ISO 8601 timestamp, handling various formats from Strava/Garmin."""
    s = s.strip()
    # Handle Z suffix
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    # Handle fractional seconds
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    # Try common formats
    for fmt in ('%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S.%f%z', 
                '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f'):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {s}")


def _haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in metres between two lat/lon points."""
    if any(np.isnan(x) for x in (lat1, lon1, lat2, lon2)):
        return 0.0
    R = 6371000  # Earth radius in metres
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ---------------------------------------------------------------------------
# GPX/TCX → NPZ cache (same format as rebuild's per-second cache)
# ---------------------------------------------------------------------------

def save_persec_npz(df: pd.DataFrame, out_path: str):
    """
    Save a per-second DataFrame as NPZ in the same format rebuild uses.
    
    This allows StepB to read GPX-derived data identically to FIT-derived data.
    """
    arrays = {}
    for col in ['ts', 'hr', 'cadence', 'power_w', 'speed', 'dist_m', 'elev_m', 'lat', 'lon']:
        if col in df.columns:
            arrays[col] = df[col].to_numpy(dtype=np.float64)
    
    # Derived columns that rebuild adds
    if 'ts' in arrays and len(arrays['ts']) > 0:
        arrays['t'] = arrays['ts'] - arrays['ts'][0]  # elapsed seconds
    
    # Grade from elevation
    if 'dist_m' in arrays and 'elev_m' in arrays:
        dist = arrays['dist_m']
        elev = arrays['elev_m']
        if np.isfinite(dist).any() and np.isfinite(elev).any():
            dd = np.diff(dist, prepend=dist[0])
            de = np.diff(elev, prepend=elev[0])
            dd = np.where(dd > 0.5, dd, np.nan)  # need meaningful distance
            grade = de / dd
            grade = np.clip(np.where(np.isfinite(grade), grade, 0.0), -0.5, 0.5)
            arrays['grade'] = grade
    
    # Moving flag
    if 'speed' in arrays:
        arrays['moving'] = (arrays['speed'] > 0.5).astype(np.float64)
    
    # GPS valid
    if 'lat' in arrays and 'lon' in arrays:
        arrays['gps_valid'] = (np.isfinite(arrays['lat']) & np.isfinite(arrays['lon'])).astype(np.float64)
    
    np.savez_compressed(out_path, **arrays)


# ---------------------------------------------------------------------------
# Strava activities.csv parsing
# ---------------------------------------------------------------------------

def load_strava_activities(csv_path: str) -> pd.DataFrame:
    """
    Load and parse Strava activities.csv.
    
    Returns DataFrame with columns:
        activity_id, filename, activity_name, activity_type, date,
        distance_km, moving_time_s, elapsed_time_s, elev_gain_m,
        is_race, workout_type
    """
    df = pd.read_csv(csv_path)
    
    # Normalise column names (Strava format varies slightly across exports)
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower().replace(' ', '_')
        col_map[c] = cl
    df = df.rename(columns=col_map)
    
    # Activity ID  
    if 'activity_id' not in df.columns:
        # Some exports use just 'id'
        if 'id' in df.columns:
            df['activity_id'] = df['id']
        else:
            df['activity_id'] = range(len(df))
    
    # Filename — Strava names activity files by activity_id
    if 'filename' not in df.columns:
        # Strava export typically has a 'Filename' column pointing to activities/XXXX.fit.gz
        for c in df.columns:
            if 'filename' in c.lower() or 'file' in c.lower():
                df['filename'] = df[c]
                break
    
    # Activity type — filter to runs
    type_col = None
    for c in df.columns:
        if 'activity_type' in c or 'type' == c:
            type_col = c
            break
    
    if type_col:
        df['activity_type_lower'] = df[type_col].astype(str).str.lower()
        # Keep runs only (including trail runs, virtual runs, etc.)
        run_mask = df['activity_type_lower'].str.contains('run', na=False)
        n_total = len(df)
        df = df[run_mask].copy()
        n_runs = len(df)
        print(f"  Strava activities.csv: {n_runs} runs out of {n_total} total activities")
    
    # Distance
    dist_col = None
    for c in df.columns:
        if 'distance' in c.lower() and 'km' not in c.lower():
            dist_col = c
            break
    if dist_col:
        df['distance_km'] = pd.to_numeric(df[dist_col], errors='coerce')
        # Strava distance is in metres or km depending on export version
        if df['distance_km'].median(skipna=True) > 1000:
            df['distance_km'] = df['distance_km'] / 1000.0
    
    # Workout type — "Race" detection
    # Strava uses workout_type: 1 = Race, 0 = default, 2 = long run, 3 = workout
    wt_col = None
    for c in df.columns:
        if 'workout_type' in c.lower():
            wt_col = c
            break
    if wt_col:
        df['workout_type'] = pd.to_numeric(df[wt_col], errors='coerce').fillna(0).astype(int)
        df['is_race'] = (df['workout_type'] == 1)
    else:
        df['is_race'] = False
    
    # Activity name
    name_col = None
    for c in df.columns:
        if 'activity_name' in c or ('name' in c.lower() and 'file' not in c.lower()):
            name_col = c
            break
    if name_col:
        df['activity_name'] = df[name_col].astype(str).fillna('')
    else:
        df['activity_name'] = ''
    
    # Activity date
    date_col = None
    for c in df.columns:
        if 'activity_date' in c or ('date' in c.lower() and 'activity' in c.lower()):
            date_col = c
            break
    if date_col is None:
        for c in df.columns:
            if 'date' in c.lower():
                date_col = c
                break
    if date_col:
        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    
    return df


def extract_strava_export(zip_path: str, work_dir: str) -> dict:
    """
    Extract a Strava bulk export zip.
    
    Returns dict with:
        activities_csv: path to activities.csv (or None)
        fit_files: list of extracted .fit file paths
        gpx_files: list of extracted .gpx file paths  
        tcx_files: list of extracted .tcx file paths
        activity_map: dict mapping activity filename → extracted path
    """
    result = {
        'activities_csv': None,
        'fit_files': [],
        'gpx_files': [],
        'tcx_files': [],
        'activity_map': {},
    }
    
    extract_dir = os.path.join(work_dir, '_strava_extract')
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"Extracting Strava export: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        for info in z.infolist():
            name = info.filename
            name_lower = name.lower()
            
            # Skip directories and non-relevant files
            if info.is_dir():
                continue
            
            # activities.csv
            if name_lower.endswith('activities.csv') and '/' in name:
                # Could be at top level or in a subfolder
                out_path = os.path.join(extract_dir, 'activities.csv')
                with z.open(info) as src, open(out_path, 'wb') as dst:
                    dst.write(src.read())
                result['activities_csv'] = out_path
                continue
            elif name_lower == 'activities.csv':
                out_path = os.path.join(extract_dir, 'activities.csv')
                with z.open(info) as src, open(out_path, 'wb') as dst:
                    dst.write(src.read())
                result['activities_csv'] = out_path
                continue
            
            # Activity files (in activities/ folder or root)
            basename = os.path.basename(name)
            basename_lower = basename.lower()
            
            is_activity = ('activities/' in name_lower or 'activity' in name_lower 
                          or basename_lower.endswith(('.fit', '.fit.gz', '.gpx', '.gpx.gz', '.tcx', '.tcx.gz')))
            
            if not is_activity:
                continue
            
            # Extract and decompress
            fits_dir = os.path.join(extract_dir, 'fits')
            os.makedirs(fits_dir, exist_ok=True)
            
            raw_data = z.read(info.filename)
            
            # Handle .gz compression
            if basename_lower.endswith('.gz'):
                try:
                    raw_data = gzip.decompress(raw_data)
                    basename = basename[:-3]  # strip .gz
                    basename_lower = basename.lower()
                except Exception as e:
                    print(f"  Warning: failed to decompress {name}: {e}")
                    continue
            
            out_path = os.path.join(fits_dir, basename)
            with open(out_path, 'wb') as f:
                f.write(raw_data)
            
            result['activity_map'][basename] = out_path
            
            if basename_lower.endswith('.fit'):
                result['fit_files'].append(out_path)
            elif basename_lower.endswith('.gpx'):
                result['gpx_files'].append(out_path)
            elif basename_lower.endswith('.tcx'):
                result['tcx_files'].append(out_path)
    
    print(f"  Found: {len(result['fit_files'])} FIT, {len(result['gpx_files'])} GPX, "
          f"{len(result['tcx_files'])} TCX files")
    if result['activities_csv']:
        print(f"  activities.csv: {result['activities_csv']}")
    else:
        print("  Warning: No activities.csv found in export")
    
    return result


def gpx_to_fit_equivalent_npz(gpx_path: str, out_dir: str) -> str | None:
    """
    Convert a GPX file to an NPZ cache file that rebuild/StepB can read.
    
    Returns the NPZ path, or None if the file couldn't be parsed.
    """
    try:
        df = parse_gpx_to_df(gpx_path)
    except ET.ParseError as e:
        print(f"  Warning: XML parse error in {gpx_path}: {e}")
        return None
    except Exception as e:
        print(f"  Warning: failed to parse GPX {gpx_path}: {e}")
        return None
    
    if len(df) < 10:
        print(f"  Warning: too few points in {gpx_path}: {len(df)}")
        return None
    
    # Check for HR data (essential for RF calculation)
    hr_valid = df['hr'].notna().sum()
    if hr_valid < 10:
        print(f"  Warning: no HR data in {gpx_path} — skipping (need HR for fitness tracking)")
        return None
    
    basename = os.path.splitext(os.path.basename(gpx_path))[0]
    npz_path = os.path.join(out_dir, f"{basename}.npz")
    save_persec_npz(df, npz_path)
    return npz_path


def tcx_to_fit_equivalent_npz(tcx_path: str, out_dir: str) -> str | None:
    """
    Convert a TCX file to an NPZ cache file that rebuild/StepB can read.
    
    Returns the NPZ path, or None if the file couldn't be parsed.
    """
    try:
        df = parse_tcx_to_df(tcx_path)
    except ET.ParseError as e:
        print(f"  Warning: XML parse error in {tcx_path}: {e}")
        return None
    except Exception as e:
        print(f"  Warning: failed to parse TCX {tcx_path}: {e}")
        return None
    
    if len(df) < 10:
        print(f"  Warning: too few points in {tcx_path}: {len(df)}")
        return None
    
    hr_valid = df['hr'].notna().sum()
    if hr_valid < 10:
        print(f"  Warning: no HR data in {tcx_path} — skipping")
        return None
    
    basename = os.path.splitext(os.path.basename(tcx_path))[0]
    npz_path = os.path.join(out_dir, f"{basename}.npz")
    save_persec_npz(df, npz_path)
    return npz_path


def build_summary_from_df(df: pd.DataFrame, source_file: str, tz_local: str = "Europe/London") -> dict:
    """
    Build a summary row (matching rebuild's output format) from a per-second DataFrame.
    
    This is used for GPX/TCX files that bypass the FIT parsing path.
    Produces the same column structure as rebuild_from_fit_zip.summarize_fit().
    """
    from zoneinfo import ZoneInfo
    
    if len(df) < 10:
        return None
    
    ts = df['ts'].values
    hr = df['hr'].values
    speed = df['speed'].values
    elev = df['elev_m'].values
    lat = df['lat'].values
    lon = df['lon'].values
    
    # Start time
    start_epoch = ts[0]
    start_utc = pd.Timestamp(start_epoch, unit='s', tz='UTC')
    try:
        tz = ZoneInfo(tz_local)
        start_local = start_utc.to_pydatetime().astimezone(tz).replace(tzinfo=None)
    except Exception:
        start_local = start_utc.tz_localize(None)
    
    # Duration
    elapsed_s = int(round(ts[-1] - ts[0]))
    
    # Moving time (speed > 0.5 m/s)
    dt = np.diff(ts, prepend=ts[0])
    dt = np.clip(dt, 0, 10)
    moving = speed > 0.5
    moving_s = int(round(np.sum(dt[moving])))
    if moving_s == 0:
        moving_s = elapsed_s
    
    # Distance
    dist_m = df['dist_m'].values
    if np.isfinite(dist_m).any():
        dist_km = float(np.nanmax(dist_m) - np.nanmin(dist_m)) / 1000.0
    else:
        # Calculate from GPS
        dist_km = 0.0
        for i in range(1, len(lat)):
            if np.isfinite(lat[i]) and np.isfinite(lon[i]) and np.isfinite(lat[i-1]) and np.isfinite(lon[i-1]):
                dist_km += _haversine_m(lat[i-1], lon[i-1], lat[i], lon[i]) / 1000.0
    
    # Pace
    pace = (moving_s / 60.0) / dist_km if dist_km > 0 else np.nan
    
    # HR stats
    hr_valid = hr[np.isfinite(hr) & (hr > 0)]
    avg_hr = float(np.mean(hr_valid)) if len(hr_valid) > 0 else np.nan
    max_hr_val = float(np.max(hr_valid)) if len(hr_valid) > 0 else np.nan
    
    # Speed stats
    avg_speed_mps = (dist_km * 1000.0 / moving_s) if moving_s > 0 else np.nan
    
    # Elevation
    elev_valid = elev[np.isfinite(elev)]
    elev_gain = np.nan
    elev_loss = np.nan
    if len(elev_valid) > 10:
        de = np.diff(elev_valid)
        elev_gain = float(np.sum(de[de > 0]))
        elev_loss = float(abs(np.sum(de[de < 0])))
    
    # Grade
    grade_mean_pct = np.nan
    if 'grade' in df.columns:
        g = df['grade'].values
        g_mov = g[moving & np.isfinite(g)]
        if len(g_mov) > 0:
            grade_mean_pct = float(np.mean(g_mov)) * 100.0
    
    # GPS metrics
    gps_valid = np.isfinite(lat) & np.isfinite(lon)
    gps_coverage = float(np.mean(gps_valid))
    
    lat_med = float(np.nanmedian(lat[gps_valid])) if gps_valid.any() else np.nan
    lon_med = float(np.nanmedian(lon[gps_valid])) if gps_valid.any() else np.nan
    
    return {
        'date': start_local,
        'file': source_file,
        'notes': 'gpx_import' if source_file.lower().endswith('.gpx') else 'tcx_import',
        'distance_km': round(dist_km, 3),
        'gps_distance_km': round(dist_km, 3),
        'gps_coverage': round(gps_coverage, 3),
        'gps_distance_ratio': 1.0,
        'elapsed_time_s': elapsed_s,
        'moving_time_s': moving_s,
        'avg_pace_min_per_km': round(pace, 2) if np.isfinite(pace) else np.nan,
        'avg_power_w': np.nan,  # No power in GPX
        'npower_w': np.nan,
        'avg_hr': round(avg_hr, 1) if np.isfinite(avg_hr) else np.nan,
        'max_hr': int(round(max_hr_val)) if np.isfinite(max_hr_val) else np.nan,
        'hr_corrected': False,
        'RE_avg': np.nan,
        'RE_normalised': np.nan,
        'elev_gain_m': round(elev_gain, 1) if np.isfinite(elev_gain) else np.nan,
        'elev_loss_m': round(elev_loss, 1) if np.isfinite(elev_loss) else np.nan,
        'grade_mean_pct': round(grade_mean_pct, 2) if np.isfinite(grade_mean_pct) else np.nan,
        'avg_speed_mps': round(avg_speed_mps, 3) if np.isfinite(avg_speed_mps) else np.nan,
        'nPower_HR': np.nan,
        'start_time_utc': start_utc,
        'gps_lat_med': lat_med,
        'gps_lon_med': lon_med,
        # Stryd fields (empty for GPX imports)
        'stryd_manufacturer': '',
        'stryd_product': '',
        'stryd_serial_number': '',
        'stryd_ant_device_number': '',
        'speed_source': 'gps',
        'alt_quality': '',
    }


def generate_race_overrides(activities_df: pd.DataFrame, out_path: str) -> int:
    """
    Generate activity_overrides entries from Strava race-flagged activities.
    
    Returns count of races found.
    """
    if 'is_race' not in activities_df.columns:
        return 0
    
    races = activities_df[activities_df['is_race'] == True].copy()
    if len(races) == 0:
        return 0
    
    rows = []
    for _, r in races.iterrows():
        filename = r.get('filename', '')
        if pd.notna(filename) and filename:
            # Extract just the activity file basename (strip path and .gz)
            basename = os.path.basename(str(filename))
            if basename.endswith('.gz'):
                basename = basename[:-3]
        else:
            continue
        
        dist = r.get('distance_km', np.nan)
        name = r.get('activity_name', '')
        date = r.get('date', '')
        
        # Map distance to official distance
        official_dist = np.nan
        if np.isfinite(dist):
            if 4.5 <= dist <= 5.5:
                official_dist = 5.0
            elif 9.5 <= dist <= 10.5:
                official_dist = 10.0
            elif 20.5 <= dist <= 21.5:
                official_dist = 21.097
            elif 41.5 <= dist <= 43.0:
                official_dist = 42.195
        
        rows.append({
            'file': basename,
            'race_flag': 1,
            'official_distance_km': official_dist,
            'surface_adj': '',
            'temp_override': '',
            'notes': f'Strava race: {name}' if name else 'Strava race',
        })
    
    if rows:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_path, index=False)
        print(f"  Generated {len(rows)} race override entries → {out_path}")
    
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Ingest Strava bulk export for the running pipeline")
    ap.add_argument("--strava-zip", required=True, help="Path to Strava bulk export ZIP")
    ap.add_argument("--out-dir", required=True, help="Output directory for processed files")
    ap.add_argument("--tz", default="Europe/London", help="Local timezone (default: Europe/London)")
    ap.add_argument("--persec-cache-dir", default="", help="Write per-second NPZ cache for GPX/TCX files")
    ap.add_argument("--skip-gpx-summary", action="store_true", help="Skip building summary rows for GPX/TCX (just extract FITs)")
    args = ap.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Create temp working directory
    work_dir = os.path.join(args.out_dir, '_work')
    os.makedirs(work_dir, exist_ok=True)
    
    # Step 1: Extract
    print("\n=== Step 1: Extract Strava export ===")
    export = extract_strava_export(args.strava_zip, work_dir)
    
    # Step 2: Parse activities.csv
    print("\n=== Step 2: Parse activities.csv ===")
    activities_df = None
    if export['activities_csv']:
        activities_df = load_strava_activities(export['activities_csv'])
        # Copy cleaned activities.csv to output
        out_csv = os.path.join(args.out_dir, 'activities.csv')
        shutil.copy2(export['activities_csv'], out_csv)
        print(f"  Copied activities.csv → {out_csv}")
    
    # Step 3: Generate race overrides from Strava workout_type
    if activities_df is not None:
        race_csv = os.path.join(args.out_dir, 'strava_race_overrides.csv')
        n_races = generate_race_overrides(activities_df, race_csv)
        if n_races > 0:
            print(f"  Found {n_races} Strava-flagged races")
    
    # Step 4: Handle GPX/TCX files
    n_gpx_ok = 0
    n_tcx_ok = 0
    gpx_summaries = []
    
    if export['gpx_files'] or export['tcx_files']:
        print(f"\n=== Step 3: Process GPX/TCX files ===")
        
        cache_dir = args.persec_cache_dir or os.path.join(args.out_dir, 'persec_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        for gpx_path in export['gpx_files']:
            npz = gpx_to_fit_equivalent_npz(gpx_path, cache_dir)
            if npz:
                n_gpx_ok += 1
                if not args.skip_gpx_summary:
                    df = parse_gpx_to_df(gpx_path)
                    summary = build_summary_from_df(df, os.path.basename(gpx_path), args.tz)
                    if summary:
                        gpx_summaries.append(summary)
        
        for tcx_path in export['tcx_files']:
            npz = tcx_to_fit_equivalent_npz(tcx_path, cache_dir)
            if npz:
                n_tcx_ok += 1
                if not args.skip_gpx_summary:
                    df = parse_tcx_to_df(tcx_path)
                    summary = build_summary_from_df(df, os.path.basename(tcx_path), args.tz)
                    if summary:
                        gpx_summaries.append(summary)
        
        print(f"  GPX: {n_gpx_ok}/{len(export['gpx_files'])} processed successfully")
        print(f"  TCX: {n_tcx_ok}/{len(export['tcx_files'])} processed successfully")
    
    # Step 5: Create consolidated FIT zip
    print(f"\n=== Step 4: Build output FIT zip ===")
    fits_zip_path = os.path.join(args.out_dir, 'fits.zip')
    n_fit = 0
    with zipfile.ZipFile(fits_zip_path, 'w', zipfile.ZIP_DEFLATED) as zout:
        for fit_path in export['fit_files']:
            zout.write(fit_path, os.path.basename(fit_path))
            n_fit += 1
    print(f"  {n_fit} FIT files → {fits_zip_path}")
    
    # Step 6: Save GPX/TCX summaries if any
    if gpx_summaries:
        summary_path = os.path.join(args.out_dir, 'gpx_tcx_summaries.csv')
        pd.DataFrame(gpx_summaries).to_csv(summary_path, index=False)
        print(f"  {len(gpx_summaries)} GPX/TCX summaries → {summary_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"STRAVA INGEST COMPLETE")
    print(f"{'='*60}")
    print(f"  FIT files:        {n_fit}")
    print(f"  GPX converted:    {n_gpx_ok}")
    print(f"  TCX converted:    {n_tcx_ok}")
    if activities_df is not None:
        print(f"  Running activities in CSV: {len(activities_df)}")
    print(f"\nOutput directory: {args.out_dir}")
    print(f"\nNext steps:")
    print(f"  python rebuild_from_fit_zip.py \\")
    print(f"    --fit-zip {fits_zip_path} \\")
    if export['activities_csv']:
        print(f"    --strava {os.path.join(args.out_dir, 'activities.csv')} \\")
    print(f"    --template master_template.xlsx \\")
    print(f"    --out {os.path.join(args.out_dir, 'Master_GPSQ_ID.xlsx')} \\")
    print(f"    --tz {args.tz}")
    if gpx_summaries:
        print(f"\n  Note: {len(gpx_summaries)} GPX/TCX activities were pre-processed.")
        print(f"  Their summaries are in {os.path.join(args.out_dir, 'gpx_tcx_summaries.csv')}")
        print(f"  These need to be appended to the master after rebuild processes the FIT files.")
    
    # Cleanup work directory
    try:
        shutil.rmtree(work_dir)
    except Exception:
        pass


if __name__ == "__main__":
    main()
