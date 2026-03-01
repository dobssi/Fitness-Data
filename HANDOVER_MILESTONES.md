# Handover: Milestones Feature Implementation
## Date: 2026-03-01

---

## Summary

Added two new dashboard sections to `generate_dashboard.py`: **Recent Achievements** and **All-Time Milestones**. Both deployed to CI and tested. One CI failure (numpy int64 JSON serialization) caught and fixed during session.

---

## What was built

### 1. Recent Achievements (position: after Stats Cards, before Training Zones)

Scans last 60 days for noteworthy performances using cascading lookback windows (1y, 2y, 3y, 5y, 10y, all-time):

- **Age Grade best-in-period**: e.g. "Best Age Grade in 5 years: 77.9%"
- **Race time best-in-period**: e.g. "Fastest 5K in 3 years: 19:42"  
- **Surface-specific PBs**: e.g. "Indoor 3K PB: 11:21" (compares only within same surface)
- **RFL Trend peaks**: e.g. "Highest Fitness in 6 months: 91.1%"

Sorted by date (newest first), then significance as tiebreaker. Deduplicated: one entry per distance (best window wins), one AG entry, surface PBs kept separate.

**Current output for Paul (as of 2026-03-01):**
- 🎖️ Best Age Grade in 5 years: 77.9% — LFOTM Feb 2026 (27 Feb)
- ⚡ Fastest 5K in 3 years: 19:42 — LFOTM Feb 2026 (27 Feb)
- ⚡ Fastest 3K in 5 years: 11:21 — Vinthundsvintern (24 Jan)
- 🏅 Indoor 3K PB: 11:21 — Vinthundsvintern (24 Jan)
- 📈 Highest Fitness in 6 months: 91.1% (6 Jan)

Section is hidden if no recent achievements exist.

### 2. All-Time Milestones (position: after Top Races, bottom of dashboard)

Tabs: **PBs** (default) | Volume | Fitness | All

**PBs tab includes:**
- Progressive time PBs per distance (e.g. 15 progressive 5K PBs from 23:11→17:52)
- Current AG PBs per distance (🏆 5K AG: 81.1%, 10K AG: 78.8%, etc.)
- Surface-specific time PBs (🏅 Indoor 3K PB: 11:21, Track 5K PB: 18:25)
- Surface-specific AG PBs (🏆 Indoor 3K AG: 71.8%, Track Mile AG: 75.4%)
- Current PBs highlighted with gold left border

**Volume tab:** Distance milestones (1K-30K km), run counts (#100-3000), race counts (#50-300), yearly volume firsts

**Fitness tab:** RFL threshold firsts, peak fitness, best AG thresholds, consistency streaks

**All tab:** Everything chronologically

**Next Milestones:** Progress bars at top showing upcoming milestones (e.g. "🚀 30,000 km Total Distance — 1773 km to go ~249d")

**Dynamic distance list:** Core distances (3K, 5K, 10K, HM, Marathon) always included. Others (Mile, 1500m, 1K, 15K, 10M) included if 2+ races at that distance.

**Surface groups:** road (default/NaN), indoor (INDOOR_TRACK), track (TRACK). Trail/snow grouped with road for AG purposes.

---

## Key design decisions

| Decision | Rationale |
|---|---|
| 60-day lookback (not 30) | 30 days missed Vinthundsvintern 3K from Jan 24 |
| Cascade fix for empty windows | When no races exist in a window, continue to wider windows instead of breaking |
| Surface PBs separate from overall | Indoor 3K 11:21 is a genuine PB even though road 3K is 10:49 |
| Dates instead of stars | Stars were noise — window badge already shows significance |
| No "Top" tab | PBs as default is cleaner, other filters (Volume/Fitness/All) sufficient |
| Dedup by distance | Only show best achievement per distance in Recent Achievements |
| numpy sanitization | `json.dumps` fails on numpy int64/float64 — recursive sanitize before return |

---

## Bugs found and fixed during session

1. **Cascade break bug**: When no races exist in a lookback window (e.g. no 3K in past year), code broke instead of continuing to wider windows. Fixed: empty window = "no competition, continue".

2. **numpy int64 serialization**: Milestone dicts contained numpy types from DataFrame reads. `json.dumps()` in `_generate_milestones_html` failed on CI. Fixed: recursive sanitize converting numpy types to Python native before returning from `get_milestones_data()`.

3. **3K missing from dist_names**: `get_milestones_data` had `dist_names` without 3.0 key, producing "3.0km" labels. Fixed: added 3.0:'3K' and included 3K in PB tracking loops.

---

## Files changed

Only `generate_dashboard.py` — from 5205 lines (pre-session) to 5777 lines (+572 lines).

New functions added:
- `get_recent_achievements(df, lookback_days=60)` — Recent Achievements computation
- `get_milestones_data(df)` — All-time milestones computation
- `_generate_recent_achievements_html(milestone_data)` — Recent Achievements HTML/JS
- `_generate_milestones_html(milestone_data)` — Milestones HTML/JS with tab filters

Integration points in `main()`:
- After `get_zone_data(df)`: calls `get_milestones_data(df)`, passes to `generate_html()`
- `generate_html()` signature: added `milestone_data=None` parameter
- Recent Achievements inserted after Stats Cards, before Training Zones
- Milestones inserted after Top Races, before footer
- Milestone CSS added before `.footer` styles

---

## Standalone test file

`milestones_feature.py` — generates test dashboards at 8 different cutoff dates showing milestone progression from 2014 (87 runs) to current (3117 runs). Useful for visual testing but NOT needed for production. Not integrated into pipeline.

---

## TODOs identified

### For next session
- **(a) Onboarding PB entry page**: Times, distances, optional dates → override `elapsed_time_s` in matching races. This would fix data quality issues like Ian's 94.4% AG mile.
- **(b) AG sanity check**: Flag any AG >85% for non-elite review. Ian's 94.4% mile at Ladywell Track is likely GPS undershoot on tight bends → moving_time_s too fast for actual distance.
- **(c) Race readiness 2-week window**: Awaiting Ian's feedback on whether the activity window for A races is useful or noise.

### Pre-existing TODOs unchanged
- Surface-specific effort specificity for race readiness cards
- Auto-flag Stryd power outliers (RE z<-2.5σ)
- gap_equiv_time_s from Minetti integral
- Refactor athlete folders to numeric IDs

---

## For next Claude

"Milestones feature is complete and deployed. Two sections added to generate_dashboard.py: Recent Achievements (after stats cards) shows best-in-period races from last 60 days with surface PBs. All-time Milestones (after top races) has PBs/Volume/Fitness/All tabs with progressive time PBs, AG PBs per distance/surface, and next-milestone progress bars. One CI bug fixed (numpy int64 JSON serialization). See HANDOVER_MILESTONES.md for full context. Next TODOs: onboarding PB entry page, AG sanity check (>85%), race readiness feedback from Ian."
