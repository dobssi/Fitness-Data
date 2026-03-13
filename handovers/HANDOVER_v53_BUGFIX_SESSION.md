# Handover: v53 Pipeline Bugfix Session — Johan + Steve
## Date: 2026-03-13

---

## Session Summary

Multi-athlete debugging session. Johan A007 INITIAL resolved after 5+ crash cycles. Steve A004 Zwift runs diagnosed as unfetchable from intervals.icu. Dedup, race classification, encoding, and dashboard fixes across multiple files.

---

## Johan A007

### INITIAL crash history (resolved)

| Crash | Cause | Fix |
|-------|-------|-----|
| 1 (prev session) | `UnicodeDecodeError` in strava_ingest.py | `encoding='latin-1'` |
| 2 (prev session) | Same in rebuild load_strava | `encoding='latin-1'` |
| 3 (prev session) | Flexible column lookup for Strava CSV | `_col()` helper |
| 4 (prev session) | tz-naive/aware mismatch in match_strava | tz safety check |
| 5 (this session) | **Same tz crash — old commit checked out** | Verified push, plus new fixes below |

### Root causes found this session

1. **Corrupted `activities.csv`**: `initial_data_ingest.py` was overwriting the good CSV (from intervals.icu ingest) with a Mac resource fork file (`__MACOSX/Johan export/._activities.csv`) extracted from the Strava zip. The `._` file is binary junk that pandas reads as 1 column, 0 rows.
   - **Fix**: `strava_ingest.py` now skips `__MACOSX/` and `._` files during zip extraction.
   - **Fix**: `initial_data_ingest.py` sanity-checks the extract CSV before overwriting — won't replace if empty/malformed (<3 columns or 0 rows).

2. **match_strava crash on bad CSV**: Even with a bad CSV, the pipeline shouldn't crash. 
   - **Fix**: `match_strava()` early-exits when strava data has 0 valid dates.

3. **Encoding**: Swedish characters (ö, ä, å) corrupted — `latin-1` encoding misreads UTF-8.
   - **Fix**: All CSV reads now try UTF-8 first, fall back to latin-1. (`rebuild_from_fit_zip.py`, `strava_ingest.py`)

4. **Workflow hardcoded weight**: Johan's workflow had `--mass-kg 75` (from broken `onboard_athlete.py`). His actual weight is 91kg.
   - **Fix**: Added "Extract athlete mass from YAML" step to both jobs. All `--weight`/`--mass-kg` now use `${{ steps.athlete.outputs.mass_kg }}`.

### Dedup issue — TCX/FIT timezone duplicates

**Problem**: Johan has both FIT files (from intervals.icu, timestamps in UTC) and TCX files (from Strava export, timestamps in local Stockholm time). The ~1 hour offset between UTC and CET caused the ±600s dedup to miss duplicates → master had 1324 rows (should be ~1120).

**First fix attempt**: `nearest_hour` — snap to nearest whole-hour offset, match ±600s around it. Handled all timezones. But had a critical flaw: **no upper bound**. A TCX at 16:47 and a FIT at 09:43 the next day (17h apart) would match because `round(17h) = 17h`, `abs(0) < 600`. This killed 301 of 302 legitimate TCX-only runs, dropping master from 1324 to 1024.

**Final fix** (deployed, awaiting INITIAL result):
- **Cross-source pairs** (FIT vs TCX): `nearest_hour` matching but **capped at 14 hours** (`diff_s <= 50400`). Covers UTC-12 to UTC+2.
- **Same-source pairs** (FIT vs FIT, TCX vs TCX): tight ±600s only. Two genuine 5km runs 2 hours apart won't be falsely merged.
- Applied to both main dedup and extra-summaries dedup.
- `is_time_match` initialised to `False` to prevent undefined variable when cross-source diff > 14h.

### Marathon not classified as race

**Activity**: "Stockholm maraton - 3:31:22" (42.3km, HR 90.8% LTHR)

**Root cause chain**:
1. Swedish spelling "maraton" not in `RACE_KEYWORDS` — but Paul correctly pointed out keywords shouldn't be needed.
2. HR 90.8% passes base marathon threshold (88%) but fails the unnamed +4% uplift (92%).  
3. Pace ratio 1.089 (8.9% slower than predicted marathon pace) fails the 5% tolerance.
4. Result: neither HR-only nor HR+pace paths classify it.

**Fix**: Distance-scaled HR uplift for unnamed runs. Shorter distances need more evidence (high HR could be intervals), longer distances need less (nobody does 42km at 90% LTHR as training).

```
5K:  +4.0% → 102% LTHR unnamed threshold
10K: +3.6% → 100.6%
HM:  +2.7% → 96.7%
30K: +2.0% → 92.0%
Marathon: +1.0% → 89.0%
```

Formula: `scale = max(0.25, 1.0 - (distance_km - 5.0) / 50.0)`, `uplift = 0.04 * scale`

### Anti-keyword vs HR fix

Step 3 in classify_races (anti-keyword only → training) was blocking runs where HR was above race threshold. Changed: anti-keyword only classifies as training when HR is BELOW threshold. If HR is above, falls through to step 5 (HR + pace/distance evidence).

### Dashboard fixes

- **AG RFL capped at 100%**: Johan's predicted AG (74.1%) > best race AG (69.84%) → 106% displayed. Now `min(..., 100.0)`.
- **Prediction tabs greyed out**: Distances with no races get `opacity: 0.3`, `pointerEvents: none`. Auto-selects first distance with data.

### INITIAL status

Running with all fixes (dedup cap, encoding, marathon classification, strava_ingest Mac resource fork skip). Awaiting result for distance audit.

---

## Steve A004

### Zwift runs not fetching

**Problem**: March 3 and March 7 Zwift runs missing from dashboard.

**Diagnosis chain**:
1. intervals.icu shows the activities as "Virtual Run" (with a space).
2. `fetch_fit_files.py` matched `"VirtualRun"` (no space) → miss. **Fixed**: case-insensitive, whitespace-stripped matching.
3. But the API returns them as type `"Unknown"` (not "Virtual Run"). **Fixed**: added `"unknown"` to `RUNNING_TYPES`.
4. But `act.get("type", "")` returns empty string for missing keys, not `"Unknown"`. The debug print used `"Unknown"` as default, masking the real issue. **Fixed**: filter default changed to `"Unknown"`.
5. 35 activities found, 11 new downloads attempted — **all 11 failed with 422 Unprocessable Entity**. intervals.icu can't serve FIT files for Zwift activities. The metadata exists but no downloadable file.

**Resolution**: intervals.icu cannot provide Zwift FIT files. Options:
1. **`user_data` folder on Dropbox** (recommended) — athlete drops FIT files manually, pipeline picks them up.
2. **`fetch_zwift_fits.py`** — Python `zwift-client` library can authenticate with username/password and download FIT files directly from Zwift. Requires athlete's Zwift credentials as GitHub secrets.
3. **Strava API** — can't serve original FIT files (only streams). Deferred.

### Predictions missing

Steve's UPDATE appended a new run but predictions on the last row are NaN. The dashboard reads predictions from the last row. Needs a FROM_STEPB to recompute, or dashboard should look backwards for last valid prediction (similar to AG fix).

---

## Files Changed This Session

### `rebuild_from_fit_zip.py`
- UTF-8 first encoding for CSV reads
- `match_strava` early exit on bad strava data (0 valid dates)
- Cross-source dedup: `nearest_hour` matching capped at 14h, same-source stays ±600s
- Extra-summaries dedup: same 14h cap

### `StepB_PostProcess.py`
- (Carried from previous session — AG NaN-truthy fix, Factor ^5 for non-races)

### `generate_dashboard.py`
- AG RFL capped at 100%
- Prediction tabs: grey out distances with no races, auto-select first available
- (Carried from previous session — prediction chart fixes, tooltip, virtual exclusion)

### `classify_races.py`
- Anti-keyword falls through to step 5 when HR above threshold
- Distance-scaled unnamed HR uplift (4% at 5K → 1% at marathon)
- (Carried: virtual run deflag)

### `strava_ingest.py`
- Skip `__MACOSX/` and `._` resource fork files in zip extraction
- UTF-8 first encoding

### `fetch_fit_files.py`
- Case-insensitive, whitespace-stripped type matching
- `"unknown"` added to `RUNNING_TYPES`
- Default for missing `"type"` key changed from `""` to `"Unknown"`
- Debug logging: prints all activity types found

### `ci/initial_data_ingest.py`
- Sanity check before overwriting `activities.csv` — won't replace with empty/malformed extract

### `.github/workflows/johan_pipeline.yml`
- Dynamic mass from `athlete.yml` (was hardcoded 75kg → reads 91kg)
- "Extract athlete mass from YAML" step added to both jobs

---

## Pending

| Item | Status |
|------|--------|
| Johan INITIAL | Running with all fixes — awaiting distance audit |
| Steve Zwift FITs | Not fetchable from intervals.icu — need user_data folder or Zwift API |
| Steve predictions | Need FROM_STEPB after Zwift runs resolved |

---

## New TODOs (for TODO.md)

- **`user_data` folder pattern**: Shared Dropbox folder per athlete (`user_data/fits/`, `user_data/activities.csv`, `user_data/weight.csv`). Pipeline checks each run, merges new data, never deletes. Solves FIT import, Strava re-upload, and weight history in one pattern.
- **`fetch_zwift_fits.py`**: Download FIT files from Zwift via `zwift-client` Python library. Username/password auth. Requires athlete Zwift credentials as GitHub secrets.
- **GPS course files**: Upload GPX/KML for target races → Minetti grade costs per segment → course-specific prediction.
- **Dashboard prediction fallback**: Look backwards for last valid prediction row, not just `iloc[-1]` (same pattern as AG fix for auto-excluded rows).

---

## Recommended Next Session Priorities

1. **Audit Johan INITIAL result** — verify weekly distances match Strava, marathon classified, Swedish names correct
2. **Onboarding refactor** — `onboard_athlete.py` workflow generation uses Paul Test template with placeholder substitution
3. **`user_data` folder** — implement for all athletes, solves Steve's Zwift problem immediately
4. **Steve FROM_STEPB** — after Zwift runs added, fix predictions

---

## For Next Claude

"Session fixed Johan A007 INITIAL (Mac resource fork corruption, encoding, dedup timezone bugs, marathon classification). Steve A004 Zwift runs unfetchable from intervals.icu (422 error) — need user_data folder pattern or Zwift API. Dedup now uses cross-source nearest_hour capped at 14h. Unnamed race HR uplift scales with distance (4% at 5K → 1% at marathon). Johan INITIAL running with all fixes, awaiting result. See HANDOVER_v53_BUGFIX_SESSION.md."
