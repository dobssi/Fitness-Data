# Handover: Onboarding Refactor + user_data + Fixes
## Date: 2026-03-13

---

## Summary

Major infrastructure session: onboarding workflow refactored to template-based generation, folder renames completed, user_data pattern implemented, classify_races fixes for Swedish keywords and marathon HR logic, strava_ingest fix for non-running TCX leak, dashboard prediction fallback fix. All changes validated on CI.

---

## Files Changed

| File | Status | Changes |
|------|--------|---------|
| `ci/workflow_template.yml` | **NEW** | 651-line workflow template with `{{PLACEHOLDER}}` substitution. Two-job structure, all 20 features. Schedule commented out by default. |
| `ci/merge_user_data.py` | **NEW** | Downloads `user_data/` from Dropbox, merges FITs into fits.zip (timestamp dedup), activities.csv (replace if larger), weight.csv (append to athlete_data.csv, date dedup). |
| `onboard_athlete.py` | **Modified** | Template-based workflow generation. Added `--athlete-id` and `--slug` CLI args. Old generator preserved as fallback. |
| `classify_races.py` | **Modified** | Added `maraton` (Swedish) to RACE_KEYWORDS, RACE_NAME_OVERRIDES, NON_PARKRUN_RACE_KW. Fixed Path B: sustained HR at 20km+ or 4%+ above threshold overrides pace rejection — marathon at 92% LTHR is always a race. |
| `strava_ingest.py` | **Modified** | TCX/GPX processing now filters by activities.csv running filenames. Non-running activities (cycling, swimming etc.) excluded from gpx_tcx_summaries.csv. |
| `generate_dashboard.py` | **Modified** | `_last_valid()` moved to module level. Prediction headline stats and mode toggle fall back to last valid value when latest row has NaN (caused by short jog preceding it). |
| `TODO.md` | **Modified** | Onboard refactor + folder renames + user_data completed. Workout planner feature expanded with Option 2 (LLM PDF import) and Option 3 (parameter-driven). |
| `.github/workflows/*.yml` (x6) | **Modified** | Ian/Steve/Nadi: folder renames (IanLilley to A002, SteveDavies to A004, NadiJahangiri to A003). All 6 workflows: merge_user_data step added to both jobs. |

---

## Validation Results

| Athlete | Mode | Result |
|---------|------|--------|
| Ian A002 | UPDATE | 3,078 rows, folder rename working, all Dropbox paths correct |
| Steve A004 | DASHBOARD | Predictions now showing (21:05 5K, 44:10 10K, 1:37:03 HM) |
| Johan A007 | INITIAL | 1,077 rows, 20 races (incl marathon), 3 spinning rows eliminated, PEAK_CP stable 416W |
| Johan A007 | CLASSIFY_RACES x2 | Marathon classified, PEAK_CP converged (416/416, unchanged on repeat) |

---

## Key Decisions

- **Schedule commented out by default** in workflow template — uncomment after INITIAL validated
- **`_last_valid()` at module level** — both `load_and_process_data()` and `get_summary_stats()` need it
- **Marathon HR logic:** at 20km+ distance, sustained HR above race threshold = race, regardless of pace vs prediction mismatch. Pace rejection only applies at shorter distances.
- **Strava TCX filter:** cross-reference against `activities.csv` filename column (run-only). No activities.csv = no filter (backward compatible).
- **PEAK_CP see-saw:** confirmed harmless +/-1-2W oscillation, converges within 2 runs. Not worth dampening.
- **Nadi is male** (corrected assumption).

---

## Outstanding / Next Session

- **Workout planner (Option 2/3)** — PDF training plan import via LLM or parameter-driven taper generation. Prototype tested on Johan's Hannover HM plan. See expanded TODO.
- **Prediction tuning** — Paul and Ian predictions feel too slow post-pipeline changes. Own session.
- **Johan REVIEW flags** — 15 rows for Johan to review in activity_overrides.xlsx
- **Footpod distance sanitisation** — Polar stride sensor can inflate distances on non-GPS sessions (spinning, treadmill). Current per-second cross-check doesn't catch summary-only rows. Low priority.

---

## For Next Claude

"Major infrastructure session. Onboarding refactored to template-based workflow generation (`ci/workflow_template.yml` + `--athlete-id`/`--slug` args). Ian/Nadi/Steve folders renamed to A002/A003/A004. `ci/merge_user_data.py` implements user_data folder pattern — athletes drop FITs/CSV into Dropbox, pipeline merges. Step added to all 6 workflows. `classify_races.py` fixed for Swedish `maraton` keyword + marathon-distance HR logic (sustained effort at 20km+ overrides pace rejection). `strava_ingest.py` now filters non-running TCX/GPX via activities.csv. `generate_dashboard.py` prediction fallback uses `_last_valid()` at module level. All validated on CI: Ian UPDATE, Steve DASHBOARD, Johan INITIAL x2 + CLASSIFY_RACES x2. Next priorities: workout planner feature (PDF import or parameter-driven taper), prediction tuning. See HANDOVER_ONBOARDING_REFACTOR.md."
