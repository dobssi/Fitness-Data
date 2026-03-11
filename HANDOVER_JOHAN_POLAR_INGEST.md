# Handover: Johan A007 Onboarding + Polar Ingest + GAP RF Fix
## Date: 2026-03-11

---

## Files delivered this session

1. **`StepB_PostProcess.py`** — GAP RF summary fallback (lines 4187–4209)
2. **`johan_pipeline.yml`** — corrected two-job workflow (replaces broken onboard_athlete.py output)
3. **`ci/polar_ingest.py`** — new: converts Polar Flow JSON export to fits.zip + activities.csv
4. **`ci/initial_data_ingest.py`** — updated: detects Polar exports and routes to polar_ingest.py

---

## StepB GAP RF fallback fix

**Problem:** When NPZ cache is missing for a run, StepB skips the per-second loop entirely. RF_gap_median stays NaN because GAP RF is a StepB-only computation (rebuild doesn't compute it). This caused the last row of A006 (Paul Stryd) to have no GAP RFL.

**Root cause:** A006's INITIAL likely timed out before GH Actions cache save, so the last run's NPZ wasn't available in job 2. The Master row existed (from safety upload) but without NPZ, StepB couldn't compute GAP RF.

**Fix:** Lines 4187–4209 in the no-cache path now compute a summary-level `RF_gap_median = gap_power / avg_hr` from `avg_gap_pace_min_per_km` before continuing. Uses same formula as GAP Power Score. ~4% higher than per-second value (whole-run average vs RF window) but the weighted trend absorbs it. Much better than NaN.

**Deploy before tonight's rebuilds** so both A005 and A006 benefit.

---

## Johan A007 workflow fix

**Problem:** `onboard_athlete.py` generates workflows from inline f-strings — produces a broken single-job workflow missing: two-job INITIAL split, `--full` fetch, `--cache-full`, `--weight-oldest`, weather cache dir, classify_races step, safety uploads, correct bash `${EXIT_CODE:-0}`.

**Fix:** Produced `johan_pipeline.yml` by substituting Johan's values into the battle-tested Paul Test two-job template. 606 lines vs the 332-line broken version.

**TODO (priority):** Refactor `onboard_athlete.py` to use Paul Test workflow as a template file with placeholder substitution. Test with Johan.

---

## Polar JSON ingest

**Background:** Johan has 12+ years of Polar data (2010–2026). Polar exports as JSON, not FIT. The zip contains:
- `training-session-*.json` (2,446 files) — per-exercise 1Hz data
- `activity-*.json` (3,832 files) — daily activity summaries (ignored)
- `training-target-*.json` — planned workouts (ignored)

**Polar JSON structure (per training-session):**
- 1Hz samples: HEART_RATE, SPEED (km/h), ALTITUDE, DISTANCE (cumulative m), CADENCE, TEMPERATURE
- Power: `LEFT_CRANK_CURRENT_POWER` — Stryd or Polar native, recorded at half-value (Polar treats it as single-sided cycling power). Multiply by 2 for total watts. Summary stat confirms: per-sec mean × 2 = summary avg.
- GPS: wayPoints array with lat/lon/altitude/elapsedMillis (separate from samples, needs time-alignment)
- Sport ID: 1 = running, 27 = trail running/jogging. Both accepted.
- Names: `name` field exists but often empty on newer sessions. Some Swedish names preserved (e.g. "Löpning").

**Power timeline:**
- Pre-2021: No power (Polar V800)
- 2021–2024: Power present but source is Polar native wrist power (Vantage V2/INW4A), NOT Stryd
- 2025-01-22 onwards: Stryd connected (set to 87kg, actual mass 91kg)
- Both sources use same `LEFT_CRANK_CURRENT_POWER` channel — indistinguishable in data
- GAP mode is the right choice: ignores power entirely, avoids the contaminated power column

**`ci/polar_ingest.py`:**
- Reads Polar zip, filters to running sport IDs, extracts per-second data
- Produces valid FIT files (minimal writer: file_id + session + record messages)
- GPS waypoints interpolated to 1Hz and embedded in records
- Power scaled by `--power-scale 2` (default) to get total watts
- Generates Strava-compatible `activities.csv` for name matching
- Tested on 5 sample sessions (2016, 2021, 2025, 2×2026): all valid FIT, correct sizes/CRCs

**`ci/initial_data_ingest.py` update:**
- Added Polar detection: checks for `training-session-*.json` inside zip
- Priority order: Strava (has `activities/` folder) → Polar (has training-session JSON) → Garmin (has FIT files)
- Routes to `polar_ingest.py` with `--out-dir` pointing to the data directory

---

## A005 vs A006 fresh rebuild comparison

Both rebuilt from scratch with same StepB code. Key findings:

- **Last row NaN fixed:** Both now have RF_select_code=1.0 (StepB NPZ), no more NaN
- **Systematic 1.3% RF_gap_median offset persists:** A006 RF_gap_median is 1.013× A005 on every single row. Constant across all eras (pre-Stryd and post-Stryd). Not caused by recovery filter, HR correction, temperature, mass, or distance correction. Impact minimal: 0.03% RFL, 1s on 5K prediction. Added to TODO for instrumented debug session.
- **RFL_gap_Trend delta:** 0.0003 (0.03 percentage points) — negligible
- **Peak RF_gap_Trend:** A005=2.1282, A006=2.1627 (ratio 1.016)

---

## Johan INITIAL — what to do

1. Upload `Johan_Polar.zip` to Dropbox: `/Running and Cycling/DataPipeline/athletes/A007/data/`
2. Deploy all 4 files to repo:
   - `StepB_PostProcess.py` (root)
   - `.github/workflows/johan_pipeline.yml`
   - `ci/polar_ingest.py` (new)
   - `ci/initial_data_ingest.py` (updated)
3. Run INITIAL via workflow dispatch
4. Flow: `initial_data_ingest.py` detects Polar → `polar_ingest.py` converts 2,446 sessions → `fits.zip` → rebuild → StepB → dashboard
5. Schedule is commented out — uncomment when ready for daily runs

**Expect:** ~2,446 training sessions, of which maybe 1,500–1,800 are running (rest filtered by sport ID). The two-job split gives 350 min for rebuild — should be enough.

**Activity names:** Sparse. Polar `name` field is often empty. No Strava `activities.csv`. Runs will show as dates/locations. Johan can request Strava export later for name backfill.

---

## TODOs added this session

- **PRIORITY:** Refactor `onboard_athlete.py` workflow generation to use template file instead of f-strings
- Investigate 1.3% systematic RF_gap_median offset between A005/A006 (low priority)
- Add `stryd_mass_kg` to `athlete.yml` as separate field from `mass_kg` (low priority)
