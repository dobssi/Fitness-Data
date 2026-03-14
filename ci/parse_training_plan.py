#!/usr/bin/env python3
from __future__ import annotations
"""
parse_training_plan.py — Parse a training plan PDF into planned_sessions.yml.

Athletes drop a PDF training plan into user_data/ on Dropbox.
This script extracts the schedule and converts it to a date-anchored
planned_sessions.yml that StepB reads for Banister CTL/ATL projection.

Handles:
  - Swedish and English day names and training terminology
  - Week headers with dates (e.g. "Vecka 10, 2 - 8.3" or "Week 10: March 2-8")
  - Session classification: easy, tempo, threshold, intervals, fartlek, long run, etc.
  - Duration extraction from time markers (e.g. 42', 90 min, 1h30)
  - TSS estimation from duration x intensity category
  - Optional calibration from athlete's historical training data

Usage:
  python ci/parse_training_plan.py --pdf training_plan.pdf --output planned_sessions.yml
  python ci/parse_training_plan.py --pdf plan.pdf --output ps.yml --master Master.xlsx

Output format (planned_sessions.yml):
  source: "FK Studenterna - Hannover HM"
  race_date: "2026-04-12"
  race_distance_km: 21.1
  weeks:
    - start: "2026-03-16"
      volume_pct: 90
      sessions:
        - day: mon
          description: "Easy 30' + strides"
          tss: 35
"""

import argparse
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import yaml
except ImportError:
    import json  # Fallback for YAML output


# ─── Day name mapping (Swedish + English) ────────────────────────────────────

_DAY_PATTERNS = {
    # Swedish
    'må': 'mon', 'mån': 'mon',
    'ti': 'tue', 'tis': 'tue',
    'on': 'wed', 'ons': 'wed',
    'to': 'thu', 'tor': 'thu',
    'fr': 'fri', 'fre': 'fri',
    'lö': 'sat', 'lör': 'sat',
    'sö': 'sun', 'sön': 'sun',
    # English
    'mo': 'mon', 'mon': 'mon',
    'tu': 'tue', 'tue': 'tue',
    'we': 'wed', 'wed': 'wed',
    'th': 'thu', 'thu': 'thu',
    'fr': 'fri', 'fri': 'fri',
    'sa': 'sat', 'sat': 'sat',
    'su': 'sun', 'sun': 'sun',
}

_DAY_OFFSET = {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6}

# Month name mapping (Swedish + English)
_MONTH_NAMES = {
    'jan': 1, 'januari': 1, 'january': 1,
    'feb': 2, 'februari': 2, 'february': 2,
    'mar': 3, 'mars': 3, 'march': 3,
    'apr': 4, 'april': 4,
    'maj': 5, 'may': 5,
    'jun': 6, 'juni': 6, 'june': 6,
    'jul': 7, 'juli': 7, 'july': 7,
    'aug': 8, 'augusti': 8, 'august': 8,
    'sep': 9, 'sept': 9, 'september': 9,
    'okt': 10, 'oct': 10, 'oktober': 10, 'october': 10,
    'nov': 11, 'november': 11,
    'dec': 12, 'december': 12,
}


# ─── Text extraction ─────────────────────────────────────────────────────────

def extract_text(file_path: str) -> str:
    """Extract text from PDF or plain text file."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ('.txt', '.md', '.text'):
        with open(file_path, encoding='utf-8', errors='replace') as f:
            return f.read()
    else:
        # PDF
        import pdfplumber
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return '\n'.join(pages)


# ─── Week header detection ───────────────────────────────────────────────────

def _infer_year(month: int, day: int) -> int:
    """Infer the year for a month/day based on current date."""
    today = date.today()
    # If month is more than 3 months in the past, assume next year
    candidate = date(today.year, month, day)
    if (today - candidate).days > 90:
        return today.year + 1
    return today.year


def _parse_date_component(text: str, default_month: int = None) -> date | None:
    """Parse a date component like '2.3', '8.3', 'March 2', '2 March', '2026-03-02'."""
    text = text.strip().rstrip('.')

    # ISO format: 2026-03-02
    m = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})', text)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # Swedish: D.M or D/M
    m = re.match(r'(\d{1,2})[./](\d{1,2})', text)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        return date(_infer_year(month, day), month, day)

    # English: Month D or D Month
    for mname, mnum in _MONTH_NAMES.items():
        # "March 2" or "Mar 2"
        m = re.match(rf'{re.escape(mname)}\.?\s+(\d{{1,2}})', text, re.IGNORECASE)
        if m:
            day = int(m.group(1))
            return date(_infer_year(mnum, day), mnum, day)
        # "2 March" or "2 Mar"
        m = re.match(rf'(\d{{1,2}})\s+{re.escape(mname)}', text, re.IGNORECASE)
        if m:
            day = int(m.group(1))
            return date(_infer_year(mnum, day), mnum, day)

    # Just a day number with default month
    m = re.match(r'(\d{1,2})$', text)
    if m and default_month:
        day = int(m.group(1))
        return date(_infer_year(default_month, day), default_month, day)

    return None


def detect_week_headers(lines: list[str]) -> list[tuple[int, date, date, int | None]]:
    """Find week header lines and extract date ranges + volume percentage.

    Returns: [(line_index, week_start, week_end, volume_pct), ...]
    """
    results = []

    # Pattern: "Vecka N, D - D.M" or "Vecka N, D.M - D.M" or "Vecka N, D.M – D.M"
    # Also: "Week N, ..." or "Week N: ..."
    # Exclude "Exempelvecka" (example week) which is illustrative, not a real week
    week_re = re.compile(
        r'(?<!exempel)(?:vecka|week|v|w)\s*(\d{1,2})\s*[,:]\s*(.+)',
        re.IGNORECASE
    )
    # Volume percentage: (<90% av maxvolym) or (90%) or (<80% of max)
    vol_re = re.compile(r'[<>~]?\s*(\d{2,3})\s*%', re.IGNORECASE)

    for i, line in enumerate(lines):
        m = week_re.match(line.strip())
        if not m:
            continue

        date_part = m.group(2).strip()

        # Extract volume percentage if present
        vol_pct = None
        vm = vol_re.search(date_part)
        if vm:
            vol_pct = int(vm.group(1))
            # Remove the volume part from date parsing
            date_part = date_part[:vm.start()].strip().rstrip('(').strip()

        # Split on dash/en-dash/em-dash
        parts = re.split(r'\s*[–—-]\s*', date_part, maxsplit=1)
        if len(parts) == 2:
            start_date = _parse_date_component(parts[0].strip())
            # If start has no month, try to get month from end
            end_date = _parse_date_component(parts[1].strip())
            if start_date is None and end_date is not None:
                start_date = _parse_date_component(
                    parts[0].strip(), default_month=end_date.month
                )
            if start_date and end_date:
                # Ensure start is Monday of that week
                start_monday = start_date - timedelta(days=start_date.weekday())
                results.append((i, start_monday, end_date, vol_pct))

    return results


# ─── Session line parsing ────────────────────────────────────────────────────

def parse_day_line(line: str) -> tuple[str, str] | None:
    """Parse a line into (day_abbrev, session_description).

    Returns None if line doesn't start with a day name.
    """
    line = line.strip()
    if not line:
        return None

    # Match day prefix: "Ti:", "Tue:", "Må:", "Mon:", "Sö:", etc.
    # Also handle "Ti " or "Sun " (space without colon/dot)
    m = re.match(r'([A-Za-zÅÄÖåäö]{2,3})\s*[:.]?\s+(.*)', line)
    if not m:
        return None

    day_str = m.group(1).lower()
    desc = m.group(2).strip()

    if day_str in _DAY_PATTERNS:
        return _DAY_PATTERNS[day_str], desc

    return None


# ─── Language detection and translation ───────────────────────────────────────

# Swedish → English translation map for common training terms
_SV_EN_TERMS = {
    # Session types
    'långpass': 'Long run',
    'långlopp': 'Long run',
    'lugnt': 'Easy',
    'lugn': 'Easy',
    'lätt': 'Easy',
    'vila': 'Rest',
    'ledigt': 'Rest',
    'ledig': 'Rest',
    'fridag': 'Rest',
    'tempol': 'Tempo run',
    'tempo': 'Tempo',
    'tröskel': 'Threshold',
    'intervall': 'Intervals',
    'intervaller': 'Intervals',
    'fartlek': 'Fartlek',
    'tävling': 'Race',
    'lopp': 'Race',
    'stigningar': 'strides',
    'acceleration': 'strides',
    'uppvärmning': 'Warmup',
    'nedvarvning': 'Cooldown',
    'koordination': 'drills',
    'sprintbackar': 'hill sprints',
    # Modifiers
    'med': 'with',
    'och': 'and',
    'min': 'min',
    'timmar': 'hours',
    'timme': 'hour',
    # Cross-training
    'styrka': 'Strength',
    'gym': 'Gym',
    'cykel': 'Bike',
    'simning': 'Swim',
    'skidor': 'Ski',
    'alt träning': 'Cross-train',
}

# Markers that indicate Swedish text
_SV_MARKERS = {'vecka', 'må', 'ti', 'on', 'to', 'lö', 'sö', 'långpass',
               'lugn', 'vila', 'ledigt', 'tröskel', 'tävling', 'och',
               'med', 'stigningar', 'fridag'}


def detect_language(text: str) -> str:
    """Detect whether plan text is Swedish or English.

    Returns 'sv' or 'en'.
    """
    text_lower = text.lower()
    sv_count = sum(1 for marker in _SV_MARKERS
                   if re.search(rf'\b{re.escape(marker)}\b', text_lower))
    return 'sv' if sv_count >= 2 else 'en'


def translate_description(desc: str, lang: str) -> str:
    """Translate a session description to English if Swedish.

    Preserves technical notation (F10:, E4:, 21KP, rep schemes, durations).
    Only translates known Swedish words.
    """
    if lang != 'sv':
        return desc

    # Don't translate session codes (F10:, E4:, D3:, L5:, S2:, N3:)
    # or race names (proper nouns starting with uppercase)
    # or pace notation (21KP, 42KP, @, x, ')

    result = desc
    for sv_term, en_term in _SV_EN_TERMS.items():
        # Case-insensitive whole-word replacement, preserving original case position
        pattern = re.compile(rf'\b{re.escape(sv_term)}\b', re.IGNORECASE)
        result = pattern.sub(en_term, result)

    return result


# ─── Session classification ──────────────────────────────────────────────────

def classify_session(description: str) -> str:
    """Classify a session description into a type category.

    Returns one of: rest, easy, easy_strides, tempo, threshold,
    intervals, fartlek, long_run, race, cross_train
    """
    desc_lower = description.lower()

    # Rest / off
    if re.search(r'\b(vila|rest|off|ledigt?|fri\s*dag)\b', desc_lower):
        return 'rest'

    # Race / test
    if re.search(r'\b(tävling|race|lopp|parkrun|sgp|test|match)\b', desc_lower):
        return 'race'

    # Cross-training
    if re.search(r'\b(alt\s*trän|cross.?train|cykel|bike|swim|sim\b|styrka|strength|gym|skidor|ski)', desc_lower):
        return 'cross_train'

    # Fartlek (check before intervals — fartlek contains pace intervals)
    if re.search(r'\b(fartlek)\b', desc_lower) or re.match(r'^F\d+', description):
        return 'fartlek'

    # Intervals / repetitions
    if re.search(r'\b(intervall|interval|repet|repeat)\b', desc_lower):
        return 'intervals'
    # Session codes: E1-E4 = threshold intervals, D1-D6 = Z3 reps
    if re.match(r'^E\d+', description):
        return 'threshold'
    if re.match(r'^D\d+', description):
        return 'intervals'

    # Long run
    if re.search(r'\b(långpass|long\s*run|long\b)\b', desc_lower) or re.match(r'^L\d+', description):
        return 'long_run'
    # N-type sessions (marathon-pace long runs)
    if re.match(r'^N\d+', description):
        return 'long_run'
    # S-type sessions (steady state Z3)
    if re.match(r'^S\d+', description):
        return 'tempo'

    # Threshold / tempo
    if re.search(r'\b(tröskel|threshold|tempo|LT\b|laktat|Z4)\b', desc_lower):
        return 'threshold'

    # Strides / drills / 30/30s
    if re.search(r'\b(strides?|stigningar|acceleration|30/30|koo\b|koordination|sprintback)', desc_lower):
        return 'easy_strides'

    # Duration-based heuristics
    dur = extract_duration_minutes(description)
    if dur:
        # Long if >= 80 minutes and no intensity markers
        if dur >= 80 and not re.search(r'(21KP|42KP|10KP|5KP|Z[3-5]|@\s*\d)', desc_lower):
            return 'long_run'

    # Check for interval notation: NxN or Nx(...)
    if re.search(r'\d+\s*x\s*[\d(]', desc_lower):
        return 'intervals'

    # Default: easy run
    return 'easy'


# ─── Duration extraction ─────────────────────────────────────────────────────

def extract_duration_minutes(description: str) -> float | None:
    """Extract total session duration from description text.

    Priority order:
      1. Explicit total: "= 125'" or "= <100'"
      2. Rep scheme: "6-7x (3'/3')" → reps * per-rep + warmup
      3. Leading duration: "90' + 10x30/30" → 90 + strides overhead
      4. Single standalone duration: "45'" or "60 min"
      5. Distance-based: "4km" at estimated pace
    """
    # 1. Explicit total duration: "= 125'" or "= <100'" or "= <80'"
    m = re.search(r'=\s*[<>~]?\s*(\d+)\s*[\'′]?', description)
    if m:
        val = int(m.group(1))
        if 20 <= val <= 200:  # Sanity: must be a plausible duration
            return float(val)

    # 2. Rep scheme: "6-7x (3'/3')" or "6x (1000 @ 21KP, ...)"
    m = re.search(r'(\d+)(?:-(\d+))?\s*x\s*\(?', description)
    if m:
        reps = int(m.group(2) or m.group(1))  # Use upper bound if range
        # Look for the per-rep content after the "x"
        after_x = description[m.end():]
        rep_scope = after_x.split(')')[0] if ')' in after_x else after_x[:40]
        # Find minute-marked segments: single prime (') = minutes, double ('') = seconds
        # Only count single-prime markers, skip double-prime (seconds)
        rep_mins = re.findall(r'(\d+)\s*[\'′](?![\'′])', rep_scope)
        if rep_mins:
            per_rep_total = sum(int(s) for s in rep_mins)
            return float(reps * per_rep_total + 15)  # +15 warmup/cooldown
        # Distance-based reps: "4x 4km" or "6x (1000"
        dm = re.search(r'(\d+)\s*(?:km\b|m\b)', after_x[:20])
        if dm:
            dist_val = int(dm.group(1))
            if dist_val < 100:  # km
                per_rep_min = dist_val * 4.5
            else:  # metres (1000m, 400m, 200m etc.)
                per_rep_min = dist_val / 1000 * 4.5
            return float(reps * (per_rep_min + 1.5) + 15)  # +rest +warmup
        # Bare numbers after x( likely metres: "6x (1000 @"
        dm = re.search(r'(\d{3,5})\s*[@\s]', after_x[:20])
        if dm:
            dist_val = int(dm.group(1))
            per_rep_min = dist_val / 1000 * 4.5
            return float(reps * (per_rep_min + 1.5) + 15)

    # 3. Leading duration: "90' + 10x30/30" or "75' + 10x 30/30"
    m = re.match(r'(\d+)\s*[\'′]\s*\+', description)
    if m:
        base = int(m.group(1))
        return float(base + 10)  # Base run + strides/drills overhead

    # 4. Progressive segments at start: "20'/15'/10'/15' @ ..." → sum the slash-separated parts
    m = re.match(r"(\d+)[\'′]/(\d+)[\'′](?:/(\d+)[\'′])?(?:/(\d+)[\'′])?", description)
    if m:
        total = sum(int(g) for g in m.groups() if g)
        return float(total + 12)  # + warmup/cooldown/pauses

    # 5. Hours + minutes: "1h30" or "1:30"
    m = re.search(r'(\d+)\s*[h:]\s*(\d{1,2})', description)
    if m:
        return float(int(m.group(1)) * 60 + int(m.group(2)))

    # 6. Single standalone duration at start of description: "45'" or "60 min"
    m = re.match(r'(\d+)\s*(?:[\'′]|min\b)', description)
    if m:
        return float(m.group(1))

    # 7. Duration embedded but not at start (be conservative — only large values)
    m = re.search(r'\b(\d{2,3})\s*(?:[\'′]|min\b)', description)
    if m:
        val = int(m.group(1))
        if 30 <= val <= 200:
            return float(val)

    # 8. Distance-based with no duration: "4km" → estimate
    m = re.search(r'(\d+)\s*km\b', description, re.IGNORECASE)
    if m:
        km = int(m.group(1))
        if km <= 50:
            return float(km * 5.5)  # ~5.5 min/km average

    return None


# ─── TSS estimation ──────────────────────────────────────────────────────────

# Default TSS-per-minute rates by session type
# Calibrated from A001 actual data: F sessions ~120-140 TSS in 55min (~2.3/min),
# E sessions 150-266 in 65-112min (~2.3/min), easy ~1.1/min, long ~1.5/min
_DEFAULT_TSS_RATE = {
    'rest': 0,
    'easy': 1.1,
    'easy_strides': 1.3,
    'tempo': 1.8,
    'threshold': 2.3,
    'intervals': 2.2,
    'fartlek': 2.3,
    'long_run': 1.5,
    'race': 2.5,
    'cross_train': 0,  # Non-running, excluded from running TSS
}

# Default durations when none can be extracted (minutes)
_DEFAULT_DURATION = {
    'rest': 0,
    'easy': 45,
    'easy_strides': 40,
    'tempo': 50,
    'threshold': 60,
    'intervals': 55,
    'fartlek': 50,
    'long_run': 90,
    'race': 60,
    'cross_train': 60,
}


def estimate_tss(session_type: str, duration_min: float | None,
                 calibration: dict | None = None) -> float:
    """Estimate TSS from session type and duration.

    Uses TSS/minute rates, optionally calibrated from athlete history.
    """
    if session_type == 'rest':
        return 0.0
    if session_type == 'cross_train':
        return 0.0  # Non-running

    rates = calibration if calibration else _DEFAULT_TSS_RATE
    rate = rates.get(session_type, 1.0)

    dur = duration_min if duration_min else _DEFAULT_DURATION.get(session_type, 45)
    return round(dur * rate)


def build_calibration(master_path: str) -> dict | None:
    """Build TSS calibration from athlete's historical training data.

    Reads master XLSX, classifies historical sessions, computes median
    TSS-per-minute for each category. Returns rate dict or None.
    """
    if not master_path or not os.path.exists(master_path):
        return None

    try:
        import pandas as pd
        df = pd.read_excel(master_path, sheet_name=0)

        # Need at least TSS, moving_time_s or duration, and enough rows
        if 'TSS' not in df.columns or len(df) < 20:
            return None

        # Get duration in minutes
        if 'moving_time_s' in df.columns:
            df['_dur_min'] = df['moving_time_s'] / 60.0
        elif 'duration_s' in df.columns:
            df['_dur_min'] = df['duration_s'] / 60.0
        else:
            return None

        # Filter to valid rows
        valid = df[(df['TSS'] > 0) & (df['_dur_min'] > 5)].copy()
        if len(valid) < 20:
            return None

        valid['_tss_per_min'] = valid['TSS'] / valid['_dur_min']

        # Classify by simple heuristics based on available columns
        rates = {}

        # Easy runs: short-ish, low TSS/min
        easy = valid[valid['_dur_min'].between(20, 65)]
        if len(easy) > 5:
            easy_low = easy[easy['_tss_per_min'] <= easy['_tss_per_min'].quantile(0.5)]
            if len(easy_low) > 3:
                rates['easy'] = round(easy_low['_tss_per_min'].median(), 2)

        # Long runs: >= 75 min
        long_runs = valid[valid['_dur_min'] >= 75]
        if len(long_runs) > 3:
            rates['long_run'] = round(long_runs['_tss_per_min'].median(), 2)

        # High intensity: top quartile TSS/min, 30-70 min
        quality = valid[valid['_dur_min'].between(30, 70)]
        if len(quality) > 5:
            high = quality[quality['_tss_per_min'] >= quality['_tss_per_min'].quantile(0.75)]
            if len(high) > 3:
                median_rate = round(high['_tss_per_min'].median(), 2)
                rates['intervals'] = median_rate
                rates['threshold'] = round(median_rate * 1.1, 2)
                rates['fartlek'] = round(median_rate * 0.95, 2)
                rates['tempo'] = median_rate

        # Strides: slightly above easy
        if 'easy' in rates:
            rates['easy_strides'] = round(rates['easy'] * 1.15, 2)

        if rates:
            # Fill in any missing from defaults
            for k, v in _DEFAULT_TSS_RATE.items():
                if k not in rates:
                    rates[k] = v
            print(f"  Calibrated TSS rates from {len(valid)} historical runs:")
            for k, v in sorted(rates.items()):
                if v > 0:
                    print(f"    {k}: {v:.2f} TSS/min")
            return rates

    except Exception as e:
        print(f"  Warning: TSS calibration failed: {e}")

    return None


# ─── Race detection ──────────────────────────────────────────────────────────

_RACE_DISTANCES = {
    'halvmarathon': 21.1, 'halvmara': 21.1, 'half marathon': 21.1, 'half': 21.1,
    'hm': 21.1, '21k': 21.1, '21km': 21.1, '21.1': 21.1,
    'marathon': 42.195, 'mara': 42.195, '42k': 42.195, '42km': 42.195,
    '10k': 10.0, '10km': 10.0, 'mil': 10.0, 'milen': 10.0,
    '5k': 5.0, '5km': 5.0,
    '3k': 3.0, '3km': 3.0, '3000': 3.0,
    '1500': 1.5, '1500m': 1.5, 'mile': 1.609,
}


def detect_race_info(text: str) -> tuple[str | None, float | None]:
    """Detect target race date and distance from plan text.

    Returns (race_date_str, distance_km) or (None, None).
    """
    text_lower = text.lower()

    # Try to find race distance
    distance = None
    for keyword, dist in _RACE_DISTANCES.items():
        if keyword in text_lower:
            distance = dist
            break

    # Try to find race date — look for patterns like "den 12 april", "April 12"
    race_date = None
    # "den DD month" or "DD month" or "month DD"
    for mname, mnum in _MONTH_NAMES.items():
        # "den 12 april" or "12 april" or "12. april"
        m = re.search(rf'(?:den\s+)?(\d{{1,2}})\.?\s+{re.escape(mname)}\b', text_lower)
        if m:
            day = int(m.group(1))
            race_date = date(_infer_year(mnum, day), mnum, day)
            break
        # "april 12"
        m = re.search(rf'{re.escape(mname)}\s+(\d{{1,2}})', text_lower)
        if m:
            day = int(m.group(1))
            race_date = date(_infer_year(mnum, day), mnum, day)
            break

    # Also look for "vecka 15" or "week 15" as race week indicator
    if not race_date:
        m = re.search(r'(?:vecka|week)\s+(\d{1,2})', text_lower)
        if m:
            # Can't resolve to exact date without year context — skip
            pass

    return (race_date.isoformat() if race_date else None, distance)


# ─── Plan title detection ────────────────────────────────────────────────────

def detect_plan_source(text: str, pdf_filename: str) -> str:
    """Try to detect a meaningful plan title from the text or filename."""
    lines = text.strip().split('\n')

    # First non-empty line is often the title
    for line in lines[:5]:
        line = line.strip()
        if len(line) > 3 and len(line) < 100:
            return line

    return Path(pdf_filename).stem


# ─── Easy day filling ────────────────────────────────────────────────────────

def fill_easy_days(sessions: list[dict], volume_pct: int | None = None) -> list[dict]:
    """Fill in missing days with easy runs or rest.

    Rules:
    - If >= 5 sessions defined, missing days are rest
    - If 3-4 sessions, add easy runs on Mon/Wed/Fri gaps
    - Always have at least 1 rest day (prefer Friday)
    - Scale easy durations by volume_pct if provided
    """
    existing_days = {s['day'] for s in sessions}
    all_days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    missing = [d for d in all_days if d not in existing_days]

    if not missing:
        return sessions

    n_existing = len(sessions)
    result = list(sessions)

    # Ensure at least one rest day — prefer Friday
    rest_day = 'fri'
    has_rest = any(classify_session(s['description']) == 'rest' for s in sessions)

    if n_existing >= 5:
        # Just add rest for missing days
        for d in missing:
            result.append({'day': d, 'description': 'Rest', 'tss': 0})
    else:
        # Fill with easy runs, ensuring one rest day
        for d in missing:
            if d == rest_day and not has_rest:
                desc, tss = 'Rest', 0
                has_rest = True
            elif n_existing + len([x for x in result if x not in sessions]) >= 6:
                desc, tss = 'Rest', 0
            else:
                # Easy run — vary by day
                if d in ('mon',):
                    desc, tss = "Easy 30' + strides", 35
                elif d in ('wed',):
                    desc, tss = "Easy 45'", 40
                else:
                    desc, tss = "Easy 40'", 38
            result.append({'day': d, 'description': desc, 'tss': tss})

    # Scale easy day TSS by volume percentage
    if volume_pct and volume_pct < 100:
        for s in result:
            if s not in sessions and s['tss'] > 0:
                s['tss'] = round(s['tss'] * volume_pct / 100)

    # Sort by day order
    result.sort(key=lambda s: _DAY_OFFSET.get(s['day'], 0))
    return result


# ─── Main parser ─────────────────────────────────────────────────────────────

def parse_plan(text: str, pdf_filename: str, today: date,
               calibration: dict | None = None) -> dict:
    """Parse training plan text into structured output.

    Returns dict ready for YAML output.
    """
    lines = text.split('\n')

    # Detect plan metadata
    source = detect_plan_source(text, pdf_filename)
    race_date, race_distance = detect_race_info(text)

    # Detect language for translation
    lang = detect_language(text)
    if lang == 'sv':
        print("  Detected Swedish plan — translating to English")

    # Find week headers
    week_headers = detect_week_headers(lines)
    if not week_headers:
        print("  Warning: No week headers found in plan text")
        return {'source': source, 'weeks': []}

    print(f"  Found {len(week_headers)} week(s) in plan")

    # Parse sessions for each week
    weeks = []
    for wi, (header_idx, week_start, week_end, vol_pct) in enumerate(week_headers):
        # Find the end of this week's section (next week header, example block, or end of text)
        if wi + 1 < len(week_headers):
            section_end = week_headers[wi + 1][0]
        else:
            section_end = len(lines)

        # Also stop at example/abbreviation/glossary sections
        for li in range(header_idx + 1, section_end):
            if re.match(r'(?:exempel|example|förkortning|abbreviat|glossar)',
                        lines[li].strip(), re.IGNORECASE):
                section_end = li
                break

        # Extract session lines from this week's section
        sessions = []
        for li in range(header_idx + 1, section_end):
            parsed = parse_day_line(lines[li])
            if parsed:
                day, desc = parsed
                stype = classify_session(desc)  # Classify on original (handles Swedish)
                dur = extract_duration_minutes(desc)
                tss = estimate_tss(stype, dur, calibration)
                desc_out = translate_description(desc.strip(), lang)

                sessions.append({
                    'day': day,
                    'description': desc_out,
                    'tss': tss,
                })

        if sessions:
            # Fill easy days
            sessions = fill_easy_days(sessions, vol_pct)

            weeks.append({
                'start': week_start.isoformat(),
                'volume_pct': vol_pct,
                'sessions': sessions,
            })
            print(f"  Week {week_start}: {len(sessions)} sessions"
                  f"{f' ({vol_pct}% volume)' if vol_pct else ''}")

    # Filter to future weeks only (keep current week if partially future)
    current_monday = today - timedelta(days=today.weekday())
    future_weeks = [w for w in weeks if date.fromisoformat(w['start']) >= current_monday]

    if len(future_weeks) < len(weeks):
        n_past = len(weeks) - len(future_weeks)
        print(f"  Filtered out {n_past} past week(s), keeping {len(future_weeks)} future week(s)")

    result = {
        'source': source,
        'weeks': future_weeks,
    }
    if race_date:
        result['race_date'] = race_date
    if race_distance:
        result['race_distance_km'] = race_distance

    return result


# ─── YAML output ─────────────────────────────────────────────────────────────

def write_output(data: dict, output_path: str):
    """Write planned sessions to YAML file."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    try:
        import yaml
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Auto-generated from training plan PDF\n")
            f.write(f"# Parsed: {date.today().isoformat()}\n\n")
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True,
                      sort_keys=False)
    except ImportError:
        # Fallback: write as JSON (still valid YAML subset)
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Auto-generated from training plan PDF\n")
            f.write(f"# Parsed: {date.today().isoformat()}\n\n")
            json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Written to {output_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parse training plan PDF into planned_sessions.yml"
    )
    parser.add_argument("--pdf", required=True,
                        help="Path to training plan PDF")
    parser.add_argument("--output", required=True,
                        help="Output planned_sessions.yml path")
    parser.add_argument("--master", default=None,
                        help="Path to master XLSX for TSS calibration")
    parser.add_argument("--today", default=None,
                        help="Override today's date (YYYY-MM-DD) for testing")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"Error: PDF not found: {args.pdf}")
        sys.exit(1)

    today_date = date.fromisoformat(args.today) if args.today else date.today()

    print(f"\n{'─' * 60}")
    print(f"  Parsing training plan: {args.pdf}")
    print(f"{'─' * 60}\n")

    # Extract text
    text = extract_text(args.pdf)
    if len(text) < 50:
        print("Error: Could not extract meaningful text from PDF")
        print("  (Is it a scanned image? Only text-based PDFs are supported)")
        sys.exit(1)

    print(f"  Extracted {len(text)} characters from PDF")

    # Build calibration from master if available
    calibration = build_calibration(args.master)

    # Parse
    plan = parse_plan(text, args.pdf, today_date, calibration)

    if not plan.get('weeks'):
        print("\n  No future weeks found in plan — nothing to output")
        sys.exit(0)

    # Summary
    total_sessions = sum(len(w['sessions']) for w in plan['weeks'])
    print(f"\n  Summary: {len(plan['weeks'])} weeks, {total_sessions} sessions")
    if plan.get('race_date'):
        print(f"  Race: {plan.get('source', '?')} on {plan['race_date']}"
              f" ({plan.get('race_distance_km', '?')}km)")

    # Write output
    write_output(plan, args.output)

    print(f"\n{'─' * 60}")
    print(f"  Done")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
