# Handover: Dashboard Restyle + Zones Integration
## Date: 2026-02-10

---

## What to do next

Revamp `generate_dashboard.py` to:
1. **Dark theme** matching zones_dashboard_v3.html (DM Sans, #0f1117 bg, #1a1d27 cards)
2. **Add Training Zones section** with HR/Power/Race Effort views
3. **Add Race Readiness section** with planned race cards
4. Keep all existing 10 chart sections, restyled

---

## Zones dashboard prototype (complete, validated)

The zones_dashboard_v3.html is the reference implementation. Key design decisions:

### Zone definitions (from 662 clean-surface runs)
- **HR Zones**: LTHR 178, Max HR 192 → Z1 <144, Z2 144-160, Z3 160-169, Z4 169-178, Z5 178-192
- **Power Zones**: CP-relative → Z1 <258, Z2 258-283, Z3 283-301, Z4 301-322, Z5 322-400
- Power zones scale dynamically with RFL_Trend × PEAK_CP

### Race power targets (from config.py constants)
- Sub-5K: CP × 1.07 = 359W (3K reference)
- 5K: CP × 1.05 = 352W
- 10K: CP × 1.00 = 335W
- HM: CP × 0.95 = 318W
- Marathon: CP × 0.90 = 302W
- ±3% bands for race effort classification

### Race effort zones
- **By power**: Sub-5K = above 5K band top (>363W), then ±3% bands per distance, rest = Other
- **By HR**: Sub-5K 182+, 5K 178-182, 10K 174-178, HM 169-174, Mara 162-169, Other <162

### Structured session detection (summary-level heuristic)
Sessions with avg_hr > 148 AND avg_power 265-330W are classified as structured (intervals).
- 40% of duration estimated at 118% of avg power (the hard portions)
- 60% classified as Other (warmup/cooldown)
- This correctly assigns Tuesday club sessions to HM/10K/5K effort bands
- **Production version** should use per-second NPZ data for accurate time-in-zone

### Planned races (for athlete.yml)
```yaml
planned_races:
  - date: "2026-02-27"
    name: "5K London"
    distance_km: 5.0
  - date: "2026-04-25"
    name: "HM Stockholm"
    distance_km: 21.097
```

### Sub-5K colour: #4ade80 (green), distinct from 5K pink and Mara blue

---

## Current dashboard structure (generate_dashboard.py, 2701 lines)

### Data flow
1. `load_and_process_data()` reads Master_FULL_GPSQ_ID_post.xlsx
2. ~15 data functions extract chart-specific datasets
3. `generate_html()` builds single self-contained HTML with inline JSON + Chart.js
4. Output: single `index.html` for GitHub Pages

### Existing sections (all to keep, restyle to dark)
1. Stats grid (RFL, CP, TSB, CTL, weight, age grade)
2. RFL chart (90d/180d/1y/2y/3y/5y toggles)
3. RFL 14-day trend with projection
4. RFL all-time weekly
5. Training Load (CTL/ATL/TSB)
6. Volume (weekly/monthly/yearly toggles)
7. Weight
8. Race Predictions trend
9. Age Grade trend
10. Recent Runs table
11. Top Race Performances table

### New sections to add
12. Training Zones (HR/Power/Race Effort weekly volume + per-run)
13. Race Readiness (planned race cards with target power, predicted time, specificity minutes)

### Styling changes needed
- Background: #f5f5f5 → #0f1117
- Cards: white → #1a1d27 with #2e3340 border
- Text: #333 → #e4e7ef
- Accent: #2563eb → #818cf8
- Font: system → DM Sans + JetBrains Mono
- Chart grid: rgba(0,0,0,0.1) → rgba(255,255,255,0.04)
- Toggle buttons: restyle to pill toggles

---

## Files for next session

Upload to Claude:
1. `checkpoint_v51_<tag>.zip` (from make_checkpoint.py)
2. `Master_FULL_GPSQ_ID_post.xlsx` (for testing)
3. `zones_dashboard_v3.html` (reference for zones section styling)
4. This handover document

---

## Phase 1 status (athlete.yml / config.py)

Phase 1 YAML files are on disk but NOT wired in — config.py is still v51 original.
Files present: athlete.yml, athlete_config.py, gap_power.py, add_gap_power.py
These are dormant and harmless. No need to revert.

The dashboard restyle does NOT touch config.py — it only modifies generate_dashboard.py.

---

## NPZ cache sync status

Implemented and working. CI uploads individual NPZ files to Dropbox after each run.
Dropbox desktop app syncs to local. Full rebuild completed today with clean pandas <3.0 data.
