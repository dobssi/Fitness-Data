# Pipeline TODO — v53
## Updated: 2026-03-11

---

## Current State Summary

v53 is a mature multi-athlete running analytics pipeline (Stryd/GAP/SIM modes) processing 3,900+ runs from 2013–2026. Seven athletes: Paul (A001, Stryd), Ian (A002, GAP), Nadi (A003, GAP, dormant), Steve (A004, GAP), PaulTest (A005, GAP), Paul Stryd (A006, Stryd portability test), Johan (A007, GAP, Polar JSON ingest). Dark theme dashboard with training zones, race readiness cards, race history comparison, milestones, mode switching. CI via GitHub Actions + Dropbox. All active athletes now on two-job workflow template.

### What's working
- Three parallel RF models (Stryd/GAP/SIM), >0.989 correlation
- Race readiness cards: target power/pace, predicted time, effort mins (14d/42d), long run tail mins + Z3+ tail mins (14d/42d), taper window, race week plan
- Race History: two-card comparison, distance-filtered, effort zones, delta panel
- Milestones: recent achievements (60d) + all-time PBs/volume/fitness tabs
- Solar radiation → effective temperature adjustment
- Banister CTL/ATL with HR normalization across athletes
- Safety upload in CI (weather cache + Master survive timeouts)
- INITIAL: two-job workflow (rebuild job → auto-chains stepb_deploy)
- A001 migrated to `paul_collyer_pipeline.yml` with two-job template
- classify_races_mode: "skip" / "auto" per athlete.yml
- initial_fit_source: "local_only" protects curated FIT history on INITIAL
- RF_Trend min_periods: 10 (prevents sparse early data inflating peak)
- Mode-aware prediction charts for GAP athletes
- Polar JSON → FIT ingest (ci/polar_ingest.py) with custom FIT binary writer
- NPZ incremental upload to Dropbox in A001 workflow

### What's in progress
- **A001 INITIAL rebuild** — running on paul_collyer_pipeline.yml. Monitor for completion, verify row count (~3125), check dashboard at `docs/paul_collyer/`.

---

## Active TODOs

### High priority

**Onboard workflow generation refactor** *(PRIORITY)*
`onboard_athlete.py` builds workflow YAML from inline f-strings — produces broken single-job workflow missing: two-job INITIAL split, `--full` fetch, `--cache-full`, `--weight-oldest`, weather cache dir, classify_races step, safety uploads, correct bash `${EXIT_CODE:-0}`. Fix: use Paul Test (A005) workflow as a template file with placeholder substitution. Test with Johan A007 re-onboarding.

**Retire old `pipeline.yml`**
Legacy A001 single-job workflow. Keep as backup until `paul_collyer_pipeline.yml` INITIAL is validated, then remove.

**Ian/Nadi/Steve folder renames**
`IanLilley/` → `A002/`, `NadiJahangiri/` → `A003/`, `SteveDavies/` → `A004/`. Requires: Dropbox folder move, workflow YAML updates (`ATHLETE_DIR`, `DB_BASE`), GH Pages path updates, cache key updates. Also rename workflow files to match (`a002_pipeline.yml` etc.).

**Ian/Steve/Nadi INITIAL rebuild**
Need full INITIAL with fresh NPZ cache to get `gps_bbox_m2` and other new columns. Migrate to Paul Test two-job workflow template first.

**Remove root-level `athlete.yml`**
After A001 migration is validated, the root `athlete.yml` is redundant — A001's workflow reads from `athletes/A001/athlete.yml`. Remove to avoid confusion.

### Dashboard features

**Race History — move effort/tail to StepB** *(performance + mode toggle)*
Currently effort mins and long run tail are computed at dashboard build time by loading NPZ files for every race's 42d window (~2 min for 328 races). Move to StepB: pre-compute per-run columns (`effort_14d_pw`, `effort_42d_pw`, `effort_14d_hr`, `lr_tail_14d`, `lr_z3_tail_14d` etc.). Dashboard reads master columns instead of scanning NPZ. Enables JS mode toggle to switch power vs HR effort client-side.

**Race History — <=3km section**
3km category should be "<=3km" to include shorter races (1500m, mile, 1000m).

**Race History — short race 60+ min zone bug**
Short races showing last row with 60+ min zone data. Needs troubleshooting.

**Prediction chart "adjust for conditions"**
Condition-adjusted prediction columns likely need same `_pred_col()` mode-aware treatment added for Johan.

**Activity search + override editor** *(own session)*
In-dashboard search of activity log, edit modal for overrides, connects to existing GitHub Actions override dispatch. Replaces spreadsheet editing.

**Scheduled workout planner** *(own session)*
Planned sessions → TSS estimates → Banister projection → dashed CTL/ATL chart extension + projected TSB on race day card. Foundation partially built with `PlannedSession` dataclass and `planned_sessions` YAML block.

**Athlete page** *(own session, phase 2 after workout planner)*
`athlete.html` — forward-looking complement to dashboard. Training plan + projected CTL/ATL, race calendar, goals, profile.

**Race tooltip on table rows**
Hover popup (distance label, AG%, temp, CTL/ATL/TSB) on Recent Runs, Top Race Performances, Recent Achievements, Milestones. Pure UI.

**GPS route map** *(own session)*
Leaflet.js + OpenStreetMap, no API key. Per-second lat/lon from NPZ. Show route on click for any run. Enables split analysis.

### Pipeline / data quality

**HR spike filter for mid-session pauses**
Currently only handles HR spikes at session start. Extend to detect timestamp gaps > 60s in per-second array and apply spike cleanup after each resume. Found during F9 session analysis (11-min pause caused HR 101→186 spike).

**classify_races.py < 3km**
Missing 2× track mile, 1× track 1500, 1× road mile, 1× road 1000m in Paul Test.

**Rename nAG%**
Cleaner column name.

**Stryd outlier auto-flag**
RE z < −2.5σ vs speed → set factor=0, use GAP RF instead.

**Surface-specific trail specificity**
Race readiness long run metrics: count only hilly runs (undulation_score > threshold) for trail race cards.

**stryd_mass_kg**
Add to athlete.yml as separate field from `mass_kg`. Pipeline uses `stryd_mass_kg` for RE calculations in Stryd mode, `mass_kg` for GAP power and age grading. Johan's Stryd is 87kg, actual 91kg. Paul's Stryd is 76kg.

**NPZ upload for other workflows**
`paul_pipeline.yml` (A005), `paul_stryd_pipeline.yml` (A006), and other two-job workflows also missing incremental NPZ upload on UPDATE/FULL. Only A001's new workflow has it.

**A006 RE p90 dilution fix**
Combined s4+s5 anchor era diluting RE p90 → slightly slow 5K prediction (19:59 vs 19:41 actual).

**detect_eras.py for A001**
Integrate into A001 workflow. Separate session.

**Info sheet in Master from athlete.yml**
Add athlete config summary as first sheet in Master XLSX. Separate session.

### Low priority / backlog

**PS Floor bias** — still ~0.7% generous. 64%→~40% of races trigger floor. Not urgent.

**GAP equiv time** — `gap_equiv_time_s` from Minetti integral.

**Simulation power 3-4% too low** in pre-Stryd/v1_late eras. Mitigated by GAP era overrides.

**1.3% systematic RF_gap_median offset** between A005 (GAP) and A006 (Stryd) on identical data. Constant, not era/recovery/HR/weather/mass dependent. Impact minimal (0.03% RFL, 1s on 5K pred). Needs instrumented debug run.

**AG-driven speed cap** — once AG% established, auto-calibrate `max_avg_speed_mps` in athlete.yml.

**SIM mode** — drop from athlete-facing dashboards (keep for dev validation only). Own session.

**Ramp rate metric** — future pipeline metric and alert enhancement.

**Main pipeline `--refresh-weather-solar`** — run once then remove. Consider removing VLM 25°C temp override now solar boost handles it.

**Johan Strava export** — waiting on Johan to request from Strava. ~110km of treadmill runs missing from Polar export (known Polar limitation). Pipeline will deduplicate by timestamp.

### Cleanup / housekeeping

**Handover file consolidation**
40 handover markdown files in repo root from Feb 7 – Mar 11. Consider archiving older ones into a `docs/handovers/` folder, keeping only the most recent.

**Delete stale files**
Candidates: `cleanup_v51.bat`, `cleanup_v52.bat`, `automation_plan.md`, `handover_portability_session.md`, `multi_athlete_planning.md` (superseded by implementation), `PATCHES_A007_FIXES.md` (applied), `PREDICTION_BIAS_ANALYSIS.md` (done), `DESIGN_MILESTONES.md` (shipped).

**Consolidate CLAUDE.md → CLAUDE_RUNNING_PROJECT_OVERVIEW.md**
`CLAUDE.md` is a stale subset. Delete it and rename `CLAUDE_RUNNING_PROJECT_OVERVIEW.md` back to `CLAUDE.md`, or add a pointer.

---

## Recently completed (2026-03-11)

- **A001 two-job workflow** — `paul_collyer_pipeline.yml` created, INITIAL rebuild triggered. New fields: `classify_races_mode: skip`, `initial_fit_source: local_only`.
- **RF_Trend min_periods** — Changed from `valid.any()` to `valid.sum() >= RF_TREND_MIN_PERIODS` (10) in Stryd/GAP/Sim trend blocks. Fixes Johan's inflated 2015 peak.
- **Prediction chart mode-awareness** — `_pred_col()` helper resolves mode-specific prediction columns for GAP athletes.
- **Stryd firmware analysis** — ~1.2% power calibration shift identified (Feb 28 firmware update), below detect_eras.py 3% threshold. 2.3% genuine fitness dip from accumulated load post-5K race.
- **v53 factor boost** — Additive `1000 × (min((PS/CP)^25, 4) - 1)` replacing flat multiplicative boost, merged into StepB.

## Recently completed (2026-03-08 – 2026-03-10)

- **classify_races.py overhaul** — data-driven `max(2%, 300m)` GPS tolerance, 1500m/Mile added, `gps_distance_km`, bespoke distance detection, pace override for named short races, per-run 5K prediction lookup. 262 races classified for PaulTest.
- **Planned sessions feature** — `PlannedSession` dataclass, YAML block, Banister forward projection, dashed CTL/ATL/TSB lines, projected race-day arrival card.
- **Race History** — two-card comparison with effort zones, long run tail, delta panel, NPZ caching.
- **A006 Stryd portability** — `detect_eras.py` created, PEAK_CP bootstrapped from 95 races, multi-sport filtering automatic.

## Recently completed (2026-03-04 – 2026-03-06)

- **Stat card tooltips** — all 16 dashboard stat cards have hover tooltips.
- **Onboarding HR zone preview** — live zone preview, editable boundaries, exports to YAML.
- **Onboarding data uploads** — any file combination, weight CSV, return visit panel.
- **Race HR thresholds in templates** — commented-out block in generated YAML.
