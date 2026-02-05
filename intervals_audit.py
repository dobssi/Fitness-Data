"""
intervals_audit.py  —  Audit intervals.icu account completeness
================================================================
Pulls all activities and wellness data via the intervals.icu API,
then prints summary tables:
  1. Activity count by year and sport type
  2. Weight data coverage by year (days with weight recorded)
  3. Gaps in activity history (periods > N days with no activity)

Setup:
  1. Go to https://intervals.icu/settings -> Developer Settings
  2. Copy your API key
  3. Find your athlete ID (shown in the URL when logged in, e.g. intervals.icu/athlete/i12345)
  4. Set them below or as environment variables:
       INTERVALS_API_KEY=your_key
       INTERVALS_ATHLETE_ID=i12345

Usage:
  pip install requests tabulate --break-system-packages
  python intervals_audit.py
"""

import os
import sys
import json
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import requests
except ImportError:
    sys.exit("Missing 'requests' — run: pip install requests --break-system-packages")

try:
    from tabulate import tabulate
except ImportError:
    sys.exit("Missing 'tabulate' — run: pip install tabulate --break-system-packages")

# -- Configuration ----------------------------------------------
API_KEY     = os.environ.get("INTERVALS_API_KEY", "YOUR_API_KEY_HERE")
ATHLETE_ID  = os.environ.get("INTERVALS_ATHLETE_ID", "YOUR_ATHLETE_ID_HERE")
BASE_URL    = "https://intervals.icu/api/v1"

# Date range to audit (adjust as needed)
OLDEST      = "2013-01-01"
NEWEST      = datetime.now().strftime("%Y-%m-%d")

# Gap detection threshold (days)
GAP_THRESHOLD_DAYS = 14

# -- API helpers ------------------------------------------------
session = requests.Session()
session.auth = ("API_KEY", API_KEY)


def api_get(endpoint, params=None):
    """GET from intervals.icu API with error handling."""
    url = f"{BASE_URL}{endpoint}"
    resp = session.get(url, params=params, timeout=30)
    if resp.status_code == 401:
        sys.exit("Authentication failed — check your API key and athlete ID.")
    if resp.status_code == 403:
        sys.exit("Access forbidden — check permissions for your API key.")
    resp.raise_for_status()
    return resp.json()


# -- Fetch activities -------------------------------------------
def fetch_activities():
    """Fetch all activities in date range."""
    print(f"Fetching activities {OLDEST} -> {NEWEST} ...")
    data = api_get(
        f"/athlete/{ATHLETE_ID}/activities",
        params={"oldest": OLDEST, "newest": NEWEST}
    )
    print(f"  -> {len(data)} activities found\n")
    return data


# -- Fetch wellness (weight) -----------------------------------
def fetch_wellness():
    """Fetch all wellness records in date range."""
    print(f"Fetching wellness data {OLDEST} -> {NEWEST} ...")
    data = api_get(
        f"/athlete/{ATHLETE_ID}/wellness",
        params={"oldest": OLDEST, "newest": NEWEST}
    )
    print(f"  -> {len(data)} wellness records found\n")
    return data


# -- Analysis ---------------------------------------------------
def analyse_activities(activities):
    """Count activities by year and sport type."""
    by_year_sport = defaultdict(lambda: defaultdict(int))
    by_year_total = defaultdict(int)
    all_sports = set()
    dates = []

    for act in activities:
        date_str = act.get("start_date_local", "")[:10]
        sport = act.get("type", "Unknown")
        if not date_str:
            continue
        year = date_str[:4]
        by_year_sport[year][sport] += 1
        by_year_total[year] += 1
        all_sports.add(sport)
        dates.append(date_str)

    return by_year_sport, by_year_total, sorted(all_sports), sorted(dates)


def analyse_wellness(wellness):
    """Count days with weight data by year."""
    weight_by_year = defaultdict(int)
    days_by_year = defaultdict(int)
    weight_range_by_year = defaultdict(lambda: [999, 0])
    first_weight_date = None
    last_weight_date = None

    for rec in wellness:
        date_str = rec.get("id", "")  # wellness uses "id" as the date
        year = date_str[:4] if date_str else None
        weight = rec.get("weight")

        if year:
            days_by_year[year] += 1

        if weight and weight > 0 and year:
            weight_by_year[year] += 1
            weight_range_by_year[year][0] = min(weight_range_by_year[year][0], weight)
            weight_range_by_year[year][1] = max(weight_range_by_year[year][1], weight)
            if first_weight_date is None or date_str < first_weight_date:
                first_weight_date = date_str
            if last_weight_date is None or date_str > last_weight_date:
                last_weight_date = date_str

    return weight_by_year, days_by_year, weight_range_by_year, first_weight_date, last_weight_date


def find_gaps(dates, threshold_days=14):
    """Find gaps larger than threshold in activity dates."""
    if not dates:
        return []
    unique = sorted(set(dates))
    gaps = []
    for i in range(1, len(unique)):
        d1 = datetime.strptime(unique[i-1], "%Y-%m-%d")
        d2 = datetime.strptime(unique[i], "%Y-%m-%d")
        delta = (d2 - d1).days
        if delta >= threshold_days:
            gaps.append((unique[i-1], unique[i], delta))
    return gaps


# -- Reporting --------------------------------------------------
def print_activity_table(by_year_sport, by_year_total, all_sports):
    """Print activity counts by year and sport."""
    print("=" * 70)
    print("ACTIVITIES BY YEAR AND SPORT")
    print("=" * 70)

    years = sorted(by_year_sport.keys())
    # Show top sports + lump the rest into "Other"
    sport_counts = defaultdict(int)
    for year in years:
        for sport, count in by_year_sport[year].items():
            sport_counts[sport] += count
    top_sports = sorted(sport_counts.keys(), key=lambda s: -sport_counts[s])

    # If more than 8 sports, group minor ones
    if len(top_sports) > 8:
        show_sports = top_sports[:7]
        has_other = True
    else:
        show_sports = top_sports
        has_other = False

    headers = ["Year"] + show_sports + (["Other"] if has_other else []) + ["TOTAL"]
    rows = []
    for year in years:
        row = [year]
        other = 0
        for sport in show_sports:
            row.append(by_year_sport[year].get(sport, 0) or "")
        if has_other:
            for sport in top_sports[7:]:
                other += by_year_sport[year].get(sport, 0)
            row.append(other or "")
        row.append(by_year_total[year])
        rows.append(row)

    # Totals row
    totals = ["TOTAL"]
    for sport in show_sports:
        totals.append(sport_counts[sport])
    if has_other:
        totals.append(sum(sport_counts[s] for s in top_sports[7:]))
    totals.append(sum(by_year_total.values()))
    rows.append(totals)

    print(tabulate(rows, headers=headers, tablefmt="simple", stralign="right"))
    print()


def print_weight_table(weight_by_year, days_by_year, weight_range, first_date, last_date):
    """Print weight data coverage by year."""
    print("=" * 70)
    print("WEIGHT DATA COVERAGE BY YEAR")
    print("=" * 70)

    if first_date:
        print(f"  First weight record: {first_date}")
        print(f"  Last weight record:  {last_date}")
    else:
        print("  [WARN]  No weight data found!")
        print()
        return

    years = sorted(set(list(weight_by_year.keys()) + list(days_by_year.keys())))
    headers = ["Year", "Wellness days", "Days w/ weight", "Coverage %", "Weight range (kg)"]
    rows = []
    for year in years:
        w_days = weight_by_year.get(year, 0)
        total_days = days_by_year.get(year, 0)
        pct = f"{100*w_days/365:.0f}%" if w_days else "—"
        if year in weight_range and weight_range[year][0] < 999:
            rng = f"{weight_range[year][0]:.1f} – {weight_range[year][1]:.1f}"
        else:
            rng = "—"
        rows.append([year, total_days, w_days, pct, rng])

    print(tabulate(rows, headers=headers, tablefmt="simple", stralign="right"))
    print()


def print_gaps(gaps):
    """Print activity gaps."""
    print("=" * 70)
    print(f"ACTIVITY GAPS (>{GAP_THRESHOLD_DAYS} days)")
    print("=" * 70)

    if not gaps:
        print(f"  No gaps >{GAP_THRESHOLD_DAYS} days found — nice consistency!")
    else:
        headers = ["From", "To", "Gap (days)"]
        print(tabulate(gaps, headers=headers, tablefmt="simple"))
    print()


# -- Main -------------------------------------------------------
def main():
    if "YOUR_" in API_KEY or "YOUR_" in ATHLETE_ID:
        print("+==========================================================+")
        print("|  Please set your API key and athlete ID first.         |")
        print("|                                                        |")
        print("|  Option 1: Edit the script (API_KEY and ATHLETE_ID)    |")
        print("|  Option 2: Set environment variables:                  |")
        print("|    export INTERVALS_API_KEY=your_key_here              |")
        print("|    export INTERVALS_ATHLETE_ID=i12345                  |")
        print("+==========================================================+")
        sys.exit(1)

    print()
    print("intervals.icu Account Completeness Audit")
    print(f"Athlete: {ATHLETE_ID}   Date range: {OLDEST} -> {NEWEST}")
    print()

    # Fetch data
    activities = fetch_activities()
    wellness = fetch_wellness()

    # Analyse
    by_year_sport, by_year_total, all_sports, dates = analyse_activities(activities)
    weight_by_year, days_by_year, weight_range, first_w, last_w = analyse_wellness(wellness)
    gaps = find_gaps(dates, GAP_THRESHOLD_DAYS)

    # Report
    print_activity_table(by_year_sport, by_year_total, all_sports)
    print_weight_table(weight_by_year, days_by_year, weight_range, first_w, last_w)
    print_gaps(gaps)

    # Quick summary
    total_acts = sum(by_year_total.values())
    total_weight_days = sum(weight_by_year.values())
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total activities:    {total_acts}")
    print(f"  Sport types:         {', '.join(all_sports)}")
    print(f"  Year span:           {min(by_year_total.keys())} – {max(by_year_total.keys())}")
    print(f"  Weight data days:    {total_weight_days}")
    if gaps:
        print(f"  Gaps >{GAP_THRESHOLD_DAYS}d:           {len(gaps)}  (longest: {max(g[2] for g in gaps)}d)")
    else:
        print(f"  Gaps >{GAP_THRESHOLD_DAYS}d:           None")
    print()


if __name__ == "__main__":
    main()
