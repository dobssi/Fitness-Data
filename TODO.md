# Pipeline TODO — v52
## Updated: 2026-03-04

---

## Current State Summary

v52 is a mature three-mode pipeline (Stryd/GAP/SIM) processing 3,900+ runs from 2013–2026. Four athletes: Paul (A001, Stryd), Ian (A002, GAP), Nadi (A003, GAP), Steve (A004, GAP), Paul Test (A005, GAP). Dark theme dashboard with training zones, race readiness cards (incl. long run tail metrics), milestones, mode switching. CI via GitHub Actions + Dropbox. INITIAL mode now uses two-job structure to beat GitHub 6h limit.

### What's working
- Three parallel RF models (Stryd/GAP/SIM), >0.989 correlation
- Race readiness cards: target power/pace, predicted time, effort mins (14d/42d), long run tail mins + Z3+ tail mins (14d/42d), taper window, race week plan
- Milestones: recent achievements (60d) + all-time PBs/volume/fitness tabs
- Solar radiation → effective temperature adjustment
- Banister CTL/ATL with HR normalization across athletes
- Safety upload in CI (weather cache + Master survive timeouts)
- INITIAL: two-job workflow (rebuild job → auto-chains stepb_deploy)

---

## Active TODOs

### Dashboard features

**Race History section** *(own session)*
Two stacked card slots with distance-filtered search/select. Cards match Race Readiness design: actual time, TSB on race day, 42d/14d effort mins, long run mins + Z3+ beyond threshold. Side-by-side comparison of any two past races. ~~Decide: show predicted-at-the-time vs actual only.~~ **DONE** — shows actual time, pace, HR, nAG, TSS, CTL/ATL/TSB, RFL, effort mins (14d/42d), conditions (temp/terrain/surface). Delta comparison panel when both slots selected. Predicted-at-the-time deferred to Phase 2.

**Activity search + override editor** *(own session)*
In-dashboard search of activity log, edit modal for overrides, connects to existing GitHub Actions override dispatch. Replaces spreadsheet editing for athletes.

**Scheduled workout planner** *(own session)*
Onboarding form: add planned sessions (date, type, duration). `athlete.yml`: `planned_sessions` block. Pipeline: estimate TSS per session (duration × intensity factor by type, using athlete zones). Project CTL/ATL/TSB forward using Banister. Dashboard: extend CTL/ATL chart with dashed projected line + show projected TSB on race day card.

**Athlete page** *(own session, phase 2 after workout planner)*
`athlete.html` — second pipeline output alongside `index.html`. Forward-looking: training plan + projected CTL/ATL, race calendar with readiness status, goals/target times, profile (zones, thresholds). Fed from `athlete.yml` + projected training data.

### Onboarding

**PB entry page** — times, distances, optional dates → override `elapsed_time_s` in matching races

**AG sanity check** — flag AG >85% for non-elite review (Ian's 94.4% mile likely GPS/timing error)

**HR zones** — auto-generate from LTHR/max HR in onboarding form, display for manual adjustment. Store as explicit zone boundaries in `athlete.yml`. Pipeline prefers explicit, falls back to auto-calc.

**Strava activities.csv re-upload** — add to return visit page so athletes can update their export periodically

### Pipeline / data quality

**Race HR thresholds** — make (3K:0.98, 5K:0.98, 10K:0.97, 10M:0.95, HM:0.94, 30K:0.90, Marathon:0.88) the defaults in `classify_races.py` and `onboard_athlete.py` template

**GAP equiv time** — `gap_equiv_time_s` from Minetti integral

**Rename nAG%** — cleaner column name

**Stryd outlier auto-flag** — RE z < −2.5σ vs speed → set factor=0, use GAP RF instead

**Athlete folder refactor** — rename all athlete folders/paths to numeric IDs (A001–A004) across Dropbox, workflows, GH Pages, cache keys

**Recovery operations** — run `--refresh-weather-solar` for Ian (A002), Nadi (A003), Steve (A004)

### Low priority / backlog

**Surface-specific trail specificity** — race readiness long run metrics: count only hilly runs (undulation_score > threshold) for trail race cards

**PS Floor bias** — still ~0.7% generous. 64%→~40% of races trigger floor. Not urgent.

**Bannister mile surface override** — tagged TRACK, was road. Fix in `activity_overrides.yml`.

**Simulation power 3-4% too low** in pre-Stryd/v1_late eras. Mitigated by GAP era overrides.

---

## Recently completed (this session — 2026-03-04)

- **Race History comparison section** — two side-by-side card slots, distance-filtered race select, full training context (CTL/ATL/TSB, RFL, effort mins, conditions). Delta comparison panel shows time diff, nAG, CTL, TSB, effort, HR changes between any two races.
- **Long run tail metrics** on race readiness cards (HM+): time beyond 80% threshold + Z3+ in tail, 14d/42d windows. NPZ tail-slice for accuracy.
- **Tooltips** on all race card metric cells (.ws-tip CSS class)
- **All race card windows** unified to 14d / 42d
- **INITIAL two-job CI** — `rebuild` (350min) auto-chains `stepb_deploy` (30min) on success. Safety upload preserves weather cache + Master on timeout.
- **Dropbox upload progress counter** — silent per-file, progress every 50, summary at end

## Recently completed (prior sessions — for reference)

### 2026-03-01
- VLM temp override reviewed/kept pending solar validation
- Solar backfill working in UPDATE mode

### 2026-02-27
- shift(1) on RFL_Trend predictions (all modes)
- Solar radiation → Temp_Adj (+1°C per 200 W/m²)
- --refresh-weather-solar flag

### 2026-02-26
- Taper window detection + race week plan (HM taper sessions)
- Adaptive taper: TSB target range from race priority

### 2026-02-25
- Paul Test (A005) onboarded end-to-end
- ci/initial_data_ingest.py auto-distinguishes Strava vs Garmin exports

### 2026-02-23
- Ian, Nadi, Steve pipelines live (GAP mode)
- Cross-athlete HR normalization for TSS

### 2026-02-21
- Milestones section: recent achievements + all-time tabs
- Progressive time PBs, AG PBs, next milestone progress bars

### 2026-02-15
- Planned races: A/B/C priority, chart markers, taper alerts
- Surface-adjusted predictions (continuous P-D curve × surface multipliers)

### 2026-02-11
- Dark theme dashboard, DM Sans font
- Training zones section (HR/Power/Race Effort views)
- Race Readiness section (planned race cards)
