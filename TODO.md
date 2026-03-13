# Pipeline TODO — v53
## Updated: 2026-03-12

---

## Current State Summary

v53 is a mature multi-athlete running analytics pipeline (Stryd/GAP/SIM modes) processing 3,900+ runs from 2013–2026. Seven athletes: Paul (A001, Stryd), Ian (A002, GAP), Nadi (A003, GAP, dormant), Steve (A004, GAP), PaulTest (A005, GAP), Paul Stryd (A006, Stryd portability test), Johan (A007, GAP, Polar + Strava TCX ingest). Dark theme dashboard with training zones, race readiness cards, race history comparison, milestones, mode switching. CI via GitHub Actions + Dropbox. All active athletes now on two-job workflow template.

### What's working
- Three parallel RF models (Stryd/GAP/SIM), >0.989 correlation
- Race readiness cards: target power/pace, predicted time, effort mins (14d/42d), long run tail mins + Z3+ tail mins (14d/42d), taper window, race week plan
- Race History: two-card comparison, distance-filtered, effort zones, delta panel
- Milestones: recent achievements (60d) + all-time PBs/volume/fitness tabs
- Solar radiation → effective temperature adjustment
- Banister CTL/ATL with HR normalization across athletes
- Safety upload in CI (weather cache + Master survive timeouts)
- INITIAL: two-job workflow (rebuild job → auto-chains stepb_deploy)
- A001 migrated to `paul_collyer_pipeline.yml` — INITIAL validated (3,125 rows, 328 races, RFL 87.2%, 5K 20:22). Dashboard at `docs/paul_collyer/`
- classify_races_mode: "skip" / "auto" per athlete.yml
- initial_fit_source: "local_only" protects curated FIT history on INITIAL
- RF_Trend min_periods: 10 (prevents sparse early data inflating peak)
- Mode-aware prediction charts and headline stats for GAP athletes
- Polar JSON → FIT ingest (ci/polar_ingest.py) with custom FIT binary writer
- Strava TCX ingest: treadmill speed from distance deltas, encoding-safe CSV parsing
- Multi-source ingest: initial_data_ingest.py processes Polar + Strava zips together
- Rebuild timestamp dedup: ±600s + distance match, prefers FIT over TCX
- RE_Adj era-normalised (divides by power_adjuster_to_S4 before comparison)
- StepB distance sanitisation: per-second pace cross-check for rogue Polar distances
- NPZ incremental upload to Dropbox in A001 workflow
- s4/s5 eras merged for A001 (firmware shift <1.2%, below noise floor)

### What's in progress
- **Johan A007 INITIAL** — rebuilding with dedup fix. Previous runs had 59 duplicate pairs (Polar FIT + Strava TCX), inflating volume/CTL. Dashboard predictions and headline stats also being fixed.

---

## Active TODOs

### High priority

**Ian/Nadi/Steve folder renames**
`IanLilley/` → `A002/`, `NadiJahangiri/` → `A003/`, `SteveDavies/` → `A004/`. Requires: Dropbox folder move, workflow YAML updates (`ATHLETE_DIR`, `DB_BASE`), GH Pages path updates, cache key updates. Also rename workflow files to match (`a002_pipeline.yml` etc.). Note: Ian/Steve/Nadi INITIAL rebuilds already completed (two-job template, fresh NPZ cache).

**Rename `power_adjuster_to_S4` → `power_era_adjuster`**
Current name references a specific era (s4) that may not exist for all athletes. Cross-codebase rename: rebuild, StepB, config, detect_eras, master columns. Own session.

### Dashboard features

**Age Grade Rating / AG-relative RFL** *(favourite feature idea)*
Continuously updated age grade rating on the dashboard (already have AG% on stat card). Express age-adjusted RFL as current AG% divided by peak AG%. Gives a "how fit am I relative to my age-adjusted best" metric that's more meaningful than raw RFL for ageing athletes. The AG% trend already exists — this is presenting it as a relative fitness signal.

**Prediction chart trend line missing for GAP athletes** *(bug)*
No green prediction line or dashed adjusted line on Race Predictions chart for GAP athletes (A005, A007, etc.). Root cause: JS looks for `trend_values_gap` key and `predicted_gap` array in predData JSON — both are null/missing. The `_pred_col()` fix in `generate_dashboard.py` resolved headline stats and master columns, but the prediction chart JS dataset builder at line ~2084 needs the same fallback: if `trend_values_gap` is empty/missing, use `trend_values` (which contains the Stryd-based trend). Similarly `predicted_gap` needs to fall back to `predicted`. Reproducing on A005 and likely all GAP mode dashboards.

**RE condition-adjusted prediction scaling**
RE_Adj era bias fixed (era-normalised). But RE% still maps ~1:1 to time%, which may be too aggressive even after normalisation. A 4.7% RE improvement on flat Battersea gave adjusted 5K of ~17:00 (pre-fix). Monitor after next rebuild to see if era normalisation alone is sufficient, or if attenuation (×0.5?) is still needed.

**Race History — move effort/tail to StepB** *(performance + mode toggle)*
Currently effort mins and long run tail are computed at dashboard build time by loading NPZ files for every race's 42d window (~2 min for 328 races). Move to StepB: pre-compute per-run columns (`effort_14d_pw`, `effort_42d_pw`, `effort_14d_hr`, `lr_tail_14d`, `lr_z3_tail_14d` etc.). Dashboard reads master columns instead of scanning NPZ. Enables JS mode toggle to switch power vs HR effort client-side.

**Race History — <=3km section**
3km category should be "<=3km" to include shorter races (1500m, mile, 1000m).

**Race History — short race 60+ min zone bug**
Short races showing last row with 60+ min zone data. Needs troubleshooting.

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
Only missing Golden Stag track mile (2019) in Paul Test. Other sub-3km races now classified.

**Rename nAG%**
Cleaner column name.

**GAP equiv time** — `gap_equiv_time_s` from Minetti integral.

**Stryd outlier auto-flag** — DONE (`rf_stryd_gap_sub` in StepB). RE z < −2.0 + Stryd/GAP ratio >2% above EMA → substitutes GAP RF for Stryd RF. 59 runs flagged on A006 including Battersea parkrun. Threshold is −2.0 not −2.5 as originally spec'd.

**Surface-specific trail specificity**
Race readiness long run metrics: count only hilly runs (undulation_score > threshold) for trail race cards.

**stryd_mass_kg**
Add to athlete.yml as separate field from `mass_kg`. Pipeline uses `stryd_mass_kg` for RE calculations in Stryd mode, `mass_kg` for GAP power and age grading. Johan's Stryd is 87kg, actual 91kg. Paul's Stryd is 76kg.

**NPZ upload for other workflows**
`paul_pipeline.yml` (A005), `paul_stryd_pipeline.yml` (A006), and other two-job workflows also missing incremental NPZ upload on UPDATE/FULL. Only A001's new workflow has it.

**A006 RE p90 dilution fix**
Combined s4+s5 anchor era diluting RE p90 → slightly slow 5K prediction (19:59 vs 19:41 actual).

**Info sheet in Master from athlete.yml**
Add athlete config summary as first sheet in Master XLSX. Separate session.

### Low priority / backlog

**PS Floor bias** — still ~0.7% generous. 64%→~40% of races trigger floor. Not urgent.

**Simulation power 3-4% too low** in pre-Stryd/v1_late eras. Mitigated by GAP era overrides.

**1.3% systematic RF_gap_median offset** between A005 (GAP) and A006 (Stryd) on identical data. Constant, not era/recovery/HR/weather/mass dependent. Impact minimal (0.03% RFL, 1s on 5K pred). Needs instrumented debug run.

**AG-driven speed cap** — once AG% established, auto-calibrate `max_avg_speed_mps` in athlete.yml.

**SIM mode** — drop from athlete-facing dashboards (keep for dev validation only). Own session.

**Ramp rate metric** — future pipeline metric and alert enhancement.

**Main pipeline `--refresh-weather-solar`** — run once then remove. Consider removing VLM 25°C temp override now solar boost handles it.

**detect_eras.py for A001** — ran detection this session: 5 eras detected vs 7 manual. Key finding: era_3 sub-split (Feb-Jul 2019) not in manual config. s4/s5 merged (shift below noise floor). Not urgent — manual eras working well, detection is validation only.

### Cleanup / housekeeping

**Delete stale `CLAUDE.md`**
Stale subset of `CLAUDE_RUNNING_PROJECT_OVERVIEW.md`. Delete if still present — the overview doc is the canonical reference.

---

## Recently completed (2026-03-13)

- **Onboard workflow generation refactor** — Replaced 520-line inline f-string workflow generator with `ci/workflow_template.yml` template file + simple `{{PLACEHOLDER}}` substitution. Template based on proven Paul Test (A005) two-job workflow. Fixes all 20 gaps vs old generator: two-job INITIAL split, `--full` fetch, `--cache-full`, `--weight-oldest`, weather cache dir, classify_races step, safety uploads, correct bash `${EXIT_CODE:-0}`, `gpx_tcx_summaries.csv` support, operator precedence, PYTHONUNBUFFERED, mass from YAML, etc. Schedule commented out by default (enable after validation). Added `--athlete-id` and `--slug` CLI args for re-onboarding existing athletes. Old generator preserved as `_generate_workflow_yml_inline()` fallback.

## Recently completed (2026-03-12)

- **A001 INITIAL validated** — 3,125 rows, 328 curated races preserved (classify_races_mode: skip), PEAK_CP bootstrapped 368W from 102 races. Dashboard deployed to `docs/paul_collyer/`.
- **s4/s5 era merge** — Removed s5 boundary from A001 athlete.yml. Firmware shift <1.2%, below detect_eras noise floor (~3%). Both eras now get adjuster 1.0.
- **detect_eras.py analysis on A001** — 5 eras detected (vs 7 manual). Compared with A006 (4 eras). Differences traced to mass corrections in A001 affecting nPower_HR ratios. Manual eras retained.
- **TCX/Strava pipeline support** — `strava_ingest.py`: treadmill speed from distance deltas, UTF-8 encoding fix. `rebuild_from_fit_zip.py`: `--extra-summaries` merge, NaT weather fix, zero-FIT pipeline support. `initial_data_ingest.py`: multi-zip support (Polar primary + Strava supplementary).
- **Rebuild timestamp dedup** — ±600s + distance match (±10%), prefers FIT over TCX/summary rows. Uses `date` column (always present) instead of `start_time_utc` (dropped in final master).
- **RE_Adj era normalisation** — Divides RE_avg by `power_adjuster_to_S4` before comparing to race median reference. Removes systematic era bias (v1 was 4% off, now ~1.5%).
- **StepB distance sanitisation** — When session-level speed exceeds per-second pace by >50%, replaces `distance_km` with pace-derived value. Fixes rogue Polar stride sensor distances.
- **StepB zero division fix** — `moving_time_s = 0` guard in RF window distance calculation.
- **Dashboard GAP headline predictions** — Falls back to `pred_5k_s_gap` etc. when Stryd columns are NaN.
- **Dashboard GAP mode CP** — Race readiness and modeStats now read `CP_gap` / `effective_peak_cp_gap` from master instead of using Stryd PEAK_CP × GAP RFL.

## Recently completed (2026-03-11)

- **A001 two-job workflow** — `paul_collyer_pipeline.yml` created, INITIAL rebuild completed.
- **RF_Trend min_periods** — Changed to `valid.sum() >= 10`. Fixes Johan's inflated 2015 peak.
- **Prediction chart mode-awareness** — `_pred_col()` helper for GAP athletes.
- **Stryd firmware analysis** — ~1.2% power calibration shift (Feb 28 firmware), below detect_eras 3% threshold.
- **v53 factor boost** — Additive formula merged into StepB.
- **Johan A007 Polar ingest** — 1,439 FIT files from Polar JSON. Three rounds of FIT binary format debugging.

## Recently completed (2026-03-08 – 2026-03-10)

- **classify_races.py overhaul** — data-driven GPS tolerance, 1500m/Mile added, per-run 5K prediction lookup.
- **Planned sessions feature** — Banister forward projection, dashed CTL/ATL/TSB lines.
- **Race History** — two-card comparison with effort zones, long run tail, delta panel.
- **A006 Stryd portability** — `detect_eras.py`, PEAK_CP bootstrap from 95 races.
