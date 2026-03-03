# Handover: Race Classification Calibration + Pipeline Fixes
## Date: 2026-03-03

---

## Summary

Calibrated race classification HR thresholds from actual data, eliminated false positives, fixed intervals.icu sync gap, added weight history backfill, and added CLASSIFY_RACES workflow mode.

---

## Race Classification Changes (classify_races.py)

### HR Thresholds — data-driven calibration

Old thresholds (from initial guess) were too high for some distances, too low for others, and not monotonic. Analysed all of Paul's races vs training by distance band to find the natural HR gap.

| Distance | Old | New | Evidence |
|----------|-----|-----|----------|
| 3K | 1.01 | **0.98** | Only 2 real 3K races (98%, 102%). "Highbury Fields 3k" at 93% was a tempo (13.6min for 3km). |
| 5K | 1.00 | **0.98** | Parkruns range 53-102%. Names do the heavy lifting. |
| 10K | 1.00 | **0.97** | Lowest real 10K race = 96%. Training well below. |
| 10M | 0.98 | **0.95** | Must sit between 10K and HM (monotonic). |
| HM | 0.97 | **0.94** | Royal Parks Half 2018 at 94% is lowest real race. Training maxes at 93% (Hackney Half steady, caught by name). |
| 30K | 0.94 | **0.90** | Lidingöloppet races 90-96%. |
| Marathon | 0.93 | **0.88** | Stockholm Marathon at 89% is lowest. |

Key principle: thresholds must be **monotonically decreasing** with distance — you can sustain higher HR% in shorter races.

### "race (low)" verdict eliminated

Previously, keyword match + low HR → "race (low)" with race_flag=0. This was confusing — it looked like a race in the verdict column but wasn't flagged.

**Now**: keyword match + low HR → "training (medium)" with reason "below race threshold". No ambiguity. Low HR = not a race, end of story.

### Step 2 HR discount removed

Steps 1 and 2 both used `named_hr = min_hr - 0.05` (a 5% HR discount). This let "Värmdö - Hemmesta-Stromma half marathon distance test run" (90% LTHR) slip through as a race because "half marathon" is a keyword and 90% ≥ 89% (= 94% - 5%).

**Now**: Only step 1 (specific RACE_NAME_OVERRIDES like "Göteborgsvarvet", "Royal Parks Half") gets the 5% discount. Step 2 (generic keywords like "half", "race", "marathon", "parkrun") uses the **full threshold**. Zero real races affected by this change.

### Keyword and regex fixes (from earlier in session)

- Fixed typo: `premiärhalvan` → `premiärhalv` (matches both -halvan and -halven)
- Added `\bhalf\b` to RACE_KEYWORDS and NON_PARKRUN_RACE_KW
- Added missing names to RACE_NAME_OVERRIDES: Royal Parks, Oxford Half, Hampton Court, Göteborgsvarvet, Premiärhalven, Broloppet, Örebro, Big Half, Reading Half, Willow 10k
- Step 1 restructured: named races beat generic anti-keywords (e.g. "Hackney Half...steady") UNLESS a decisive training pattern is present ("week \d+", "sandwich", "sim", "calibrat", "prep", "warm-up...cool-down")

### Parkrun structural detection tightened

For nameless athletes (future onboarding), parkrun detection by structure:
- Distance: 4.8–5.1 km (was 4.8–5.5)
- Day: Saturday
- Start time: 9:00–9:10 or 9:30–9:40 (was 8:00–10:30)

### Test results

Post-changes: 299 races, 450 training. Zero "race (low)". All previously missing races (Göteborgsvarvet, Premiärhalven ×3, Royal Parks ×3, Oxford Half, Hampton Court, Broloppet, VLM 2018, Stockholm Marathon) now classified correctly. The two false HM positives (Värmdö, hilly training run) now correctly classified as training.

---

## Pipeline Fixes

### intervals.icu sync gap (paul_pipeline.yml)

**Problem**: INITIAL mode ran fetch_fit_files.py which set `fit_sync_state.json` last_sync to March 3. The Strava export only covered through ~Feb 24. Post-ingest fetch also read the same sync state and found nothing new. Result: ~1 week of runs (Feb 25 – Mar 3) missing from Master.

**Fix in workflow**: Post-ingest fetch (INITIAL only) now:
1. Deletes `fit_sync_state.json` before running
2. Uses `--full` flag to fetch all history from intervals.icu
3. Existing FIT files on disk are deduplicated, so only the gap downloads

**For current state**: The sync state on Dropbox still says March 3. Need to manually delete `fit_sync_state.json` from Dropbox (`/Running and Cycling/DataPipeline/athletes/A005/fit_sync_state.json`), then run UPDATE. After that, daily UPDATEs will work normally.

### Weight data backfill (sync_athlete_data.py)

**Problem**: `INTERVALS_WEIGHT_START = "2026-01-01"` was hardcoded — only fetched weight from intervals.icu from Jan 2026 onwards (pre-2026 came from BFW for Paul A001). New athletes like A005 had no weight history before 2026.

**Fix**: Added `--weight-oldest` CLI argument. Default unchanged (2026-01-01) for backward compatibility. PaulTest workflow now passes `--weight-oldest 2020-01-01` to backfill from intervals.icu.

The parameter flows through to `fetch_weight_from_intervals()`, `merge_data()`, and `postprocess_weight()`.

### CLASSIFY_RACES workflow mode (paul_pipeline.yml)

Added `CLASSIFY_RACES` to the mode dropdown. When selected:
- **Runs**: checkout, setup, download from Dropbox, classify_races.py (without --skip-if-classified), upload to Dropbox, summary
- **Skips**: rebuild, GAP power, StepB, dashboard, deploy to Pages
- Forces full re-classification of all runs (no --skip-if-classified flag)

---

## Files Changed

| File | Changes |
|------|---------|
| `classify_races.py` | HR thresholds, race(low)→training, step 2 discount removed, parkrun tightened, keywords fixed |
| `sync_athlete_data.py` | `--weight-oldest` CLI arg, plumbed through fetch/merge/postprocess |
| `paul_pipeline.yml` | CLASSIFY_RACES mode, post-ingest sync reset + --full, --weight-oldest 2020-01-01 |
| `athlete.yml` (A005) | Updated race_hr_thresholds_pct to match new defaults |

All delivered as files in outputs. `classify_races_v2.py` → rename to `classify_races.py` on deploy.

---

## Deployment sequence

1. Commit all four files
2. Delete `fit_sync_state.json` from Dropbox for A005
3. Run CLASSIFY_RACES → regenerates overrides (done, validated)
4. Run UPDATE → picks up missing recent runs + weight backfill
5. Future daily UPDATEs work normally

---

## Outstanding TODOs

- **Make HR thresholds the defaults** in classify_races.py DEFAULT_RACE_HR_THRESHOLDS and onboard_athlete.py template for new athletes (stored in memory)
- **Race (low) review**: 20 runs now correctly training(medium) — some may need manual override if genuinely raced at low HR (DNS/injury/pacing). Check the overrides spreadsheet.
- **Ian Test pipeline**: Can onboard Ian as a test using the same infrastructure. Or re-run Paul with Strava-only data (activities.csv + FIT files from intervals.icu)
- **Sync gap detection**: Consider adding logic to UPDATE mode that detects when Master's last date is significantly before the sync state's last_sync, and forces a broader fetch window automatically

---

## Key Learnings

- **HR thresholds must be monotonically decreasing with distance** — physiological constraint, not a tuning choice
- **Low HR + race keyword = training**, never a race. A parkrun jogged at 83% LTHR is an event (parkrun=1) but not a race (race_flag=0)
- **Named race discount (step 1) vs keyword discount (step 2)**: specific race names deserve HR leniency (you might jog a race you registered for). Generic keywords like "half marathon" do not — "half marathon distance training run" is not a race
- **Sync state persistence across modes**: INITIAL creates sync state that UPDATE inherits. If INITIAL's fetch window doesn't cover the full data gap, UPDATE will never backfill it. Must reset sync state between modes or detect the gap.
