# Pipeline TODO — v53
## Updated: 2026-03-14

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
- **A001 workflow missing `merge_user_data.py`** — Bespoke workflow doesn't download/process `user_data/` from Dropbox. Training plan not picked up. Root cause identified, fix not yet applied. Need to add step matching template-generated workflows.
- **A001 TSS z5_frac cache miss** — Mar 14 parkrun TSS was 103.5 (should be ~137). NPZ cache wasn't found during StepB, so z5_frac=NaN and Z5 intensity boost skipped. UPDATE triggered — verify TSS corrected on next session start.
- **Johan A007 INITIAL** — rebuilding with dedup fix. Previous runs had 59 duplicate pairs (Polar FIT + Strava TCX), inflating volume/CTL. Dashboard predictions and headline stats also being fixed.
- **athlete.yml Dropbox drift** — ✅ FIXED. Removed athlete.yml from all Dropbox sync steps. PEAK_CP bootstrap now commits back to git via `[skip ci]` commit. Stale DataPipeline copies deleted. Git is sole source of truth.

---

## Active TODOs

### High priority

**Rename `power_adjuster_to_S4` → `power_era_adjuster`**
Current name references a specific era (s4) that may not exist for all athletes. Cross-codebase rename: rebuild, StepB, config, detect_eras, master columns. Own session.

### Dashboard features

**Upcoming Sessions — first week TSS includes completed runs** ✅
Done. Week total now sums actual + planned TSS.

**Upcoming Sessions — CTL/ATL/TSB projection columns** ✅
Done. Post-workout values projected from Daily sheet. TSB colour-coded green/red.

**Upcoming Sessions — week total alignment**
Week total label should left-align under Session column, not centred. Minor styling.

**Intervals.icu RACE tag ingestion**
When athlete tags a run as RACE on intervals.icu, pipeline should pick it up. Currently not in FIT sub_sport (always generic) or activities.csv. Need to check intervals.icu API for tags field.

**Age Grade Rating / AG-relative RFL** *(favourite feature idea)*
Continuously updated age grade rating on the dashboard (already have AG% on stat card). Express age-adjusted RFL as current AG% divided by peak AG%. Gives a "how fit am I relative to my age-adjusted best" metric that's more meaningful than raw RFL for ageing athletes. The AG% trend already exists — this is presenting it as a relative fitness signal.

**Prediction tuning — Stryd mode possibly over-pessimistic**
Paul's Stryd mode predictions feel too slow. GAP predictions seem closer. Revisit Stryd-specific tuning in own session.

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

**Scheduled workout planner** *(mostly done)*
Training plan parser (`ci/parse_training_plan.py`) parses PDF/TXT plans dropped into `user_data/`. Supports Swedish + English (auto-detected, translated to English for dashboard). Classifies sessions, estimates TSS (calibrated from athlete history), outputs `planned_sessions.yml`. StepB reads it for Banister projection. Upcoming Sessions card shows plan with CTL/ATL/TSB. Race Week Plan card uses plan sessions when available.
Remaining:
- Wire plan sessions into Race Week Plan taper solver (use real sessions instead of generic taper when dates overlap)
- Option 3 (parameter-driven generic taper) still available for athletes without coach plans

**Athlete page** *(own session, phase 2 after workout planner)*
`athlete.html` — forward-looking complement to dashboard. Training plan + projected CTL/ATL, race calendar, goals, profile.

**Race tooltip on table rows**
Hover popup (distance label, AG%, temp, CTL/ATL/TSB) on Recent Runs, Top Race Performances, Recent Achievements, Milestones. Pure UI.

**GPS route map** *(own session)*
Leaflet.js + OpenStreetMap, no API key. Per-second lat/lon from NPZ. Show route on click for any run. Enables split analysis.

### Pipeline / data quality

**HR spike filter for mid-session pauses** — settling fix done, overshoot TODO
Settling time now scales with pause duration: `max(min(pause, 60), min(pause*0.3, 180)) + HR_LAG_S`. Unchanged for pauses ≤200s. Long pauses get longer settling (686s → 195s, was 75s). Handles ramp-up phase.
**Still open:** post-pause HR overshoot (HR peaks ~15bpm above equilibrium, then settles). Ratio-based filter attempted but 5% threshold false-positives on hills/effort changes — 66/3127 runs lost >50% valid RF data. Needs more surgical approach (own session). Example: Mar 10 run, HR 191 vs equilibrium 175 after 4-min pause.

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

## Recently completed (2026-03-14, session 2 — Claude Code)

- **HR spike filter — settling time fix** — Swiss cheese settling now scales with pause duration for long pauses (>200s). Formula: `max(min(pause, 60), min(pause*0.3, 180)) + HR_LAG_S`. 686s pause: 75s → 195s, was 75s. Overshoot filter attempted but reverted (5% threshold too aggressive, 66 runs lost >50% data). Overshoot deferred to own session.
- **TSS discrepancy diagnosed** — A001 parkrun TSS 103.5 vs A005/A006 ~168. Root cause: z5_frac=NaN from NPZ cache miss during StepB. Base TSS identical (~71.7); gap is Z5 intensity boost (×1.9). UPDATE triggered to fix.

## Recently completed (2026-03-14, session 1 — Claude Code)

- **GAP prediction chart trend line fix** — JS fallback from `trend_values_gap` → `trend_values` and `predicted_gap` → `predicted`. Green trend line now renders for all GAP athletes.
- **Upcoming Sessions card polish** — Added CTL/ATL/TSB projected columns (post-workout values from Daily sheet). TSB colour-coded green/red. First week total now includes already-completed TSS. Today's planned races show even when no actual run yet. Styling: CTL/ATL subtle, TSB prominent.
- **Swedish→English translation in training plan parser** — Auto-detects language from Swedish markers (vecka, långpass, vila etc.). Translates session descriptions to English while preserving technical notation (F10:, 21KP, rep schemes). Translation map covers ~30 common training terms.
- **A001 workflow fix** — `apply_run_metadata.py` arg names corrected (`--overrides` → `--override-file`, `--pending` → `--pending-file`). UPDATE runs with parkrun metadata were failing.
- **A001/A005 athlete.yml planned_races synced** from Dropbox edits.
- **athlete.yml removed from Dropbox sync** — all 7 workflows + template updated. PEAK_CP write-back now commits to git instead of Dropbox round-trip. Stale DataPipeline copies deleted. Prevents config drift between git and Dropbox.
- **HR spike filter — settling time fix** — Swiss cheese settling now scales with pause duration for long pauses (>200s). Formula: `max(min(pause, 60), min(pause*0.3, 180)) + HR_LAG_S`. 686s pause: 75s → 195s. Overshoot filter attempted but reverted (5% threshold too aggressive, 66 runs lost >50% data). Overshoot deferred to own session.

## Recently completed (2026-03-13, session 2 — Claude Code)

- **Training plan PDF/text parser** — New `ci/parse_training_plan.py`. Athletes drop a PDF or text file into `user_data/` on Dropbox. Parser extracts weekly schedules (Swedish + English), classifies sessions, estimates TSS (calibrated from athlete history via `--master`), outputs date-anchored `planned_sessions.yml`. StepB reads it for Banister CTL/ATL projection. Removing the file clears the plan.
- **Upcoming Sessions dashboard card** — Shows planned workouts from Daily sheet: date, description, TSS, grouped by week with weekly totals. Rest days greyed, races gold. Taper progression summary + race countdown. Positioned after Training Load chart.
- **All 7 workflows updated** — Parse training plan step (after merge_user_data, both jobs), `--planned-sessions` flag on all StepB calls, PDF + TXT detection, cleanup when no file present.
- **Root cleanup** — Deleted 15 `.bat` files, `MIGRATION_PLAN.md`, `CLAUDE_RUNNING_PROJECT_OVERVIEW.md`, `dir.txt`, duplicate `gitignore`. Archived 21 `HANDOVER_*.md` to `handovers/`. Fixed stale folder name references.
- **CLAUDE.md refreshed** — TODOs updated from handovers, file structure corrected, A001 noted as bespoke workflow, completed items removed.
- **Athlete guide updated** — Added `user_data/` section (training plan format, FIT uploads, CSV files). Distributed to all 7 athletes' Dropbox folders.
- **Claude Code setup** — Project + user permissions configured (`~/.claude/settings.json`).

## Recently completed (2026-03-13, session 1)

- **Onboard workflow generation refactor** — Replaced 520-line inline f-string workflow generator with `ci/workflow_template.yml` template file + simple `{{PLACEHOLDER}}` substitution. Template based on proven Paul Test (A005) two-job workflow. Fixes all 20 gaps vs old generator: two-job INITIAL split, `--full` fetch, `--cache-full`, `--weight-oldest`, weather cache dir, classify_races step, safety uploads, correct bash `${EXIT_CODE:-0}`, `gpx_tcx_summaries.csv` support, operator precedence, PYTHONUNBUFFERED, mass from YAML, etc. Schedule commented out by default (enable after validation). Added `--athlete-id` and `--slug` CLI args for re-onboarding existing athletes. Old generator preserved as `_generate_workflow_yml_inline()` fallback.
- **Ian/Nadi/Steve folder renames** — `IanLilley/` → `A002/`, `NadiJahangiri/` → `A003/`, `SteveDavies/` → `A004/`. Dropbox folders renamed, workflows updated (6 lines each — ATHLETE_DIR + DB_BASE in both jobs), repo `git mv`. Ian A002 UPDATE validated on CI.
- **`user_data` folder pattern** — New `ci/merge_user_data.py` script. Athletes drop files into `user_data/fits/` (FIT files), `user_data/activities.csv` (Strava re-export), or `user_data/weight.csv` on Dropbox. Pipeline downloads and merges each run: FITs added to `data/fits.zip` (timestamp dedup), activities.csv replaced if larger, weight appended to `athlete_data.csv` (date dedup). Step added to workflow template (both jobs). `user_data/fits/` folders created on Dropbox for all active athletes. Shared Dropbox links sent to Ian, Steve, Johan, Nadi.

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
