# Handover: Race Classification Calibration + PaulTest Fixes
## Date: 2026-03-03

---

## What happened this session

Calibrated classify_races.py HR thresholds using Paul's 3118 curated runs as ground truth, then validated against Ian's 3067-run dataset. Fixed three PaulTest pipeline issues. Updated all athlete.yml files.

---

## Key findings

### HR threshold calibration

**Counter-intuitive finding confirmed then corrected**: Initial statistical optimisation produced thresholds where 10K > 5K (non-monotonic). Root cause: ~15 10K races miscurated as training in Paul's ground truth (Stockholm's Bästa, Midnattsloppet, Kvartsmarathon at 97-100% LTHR). Optimizer compensated for dirty data.

**Paul's actual race HR ranges:**
- 5K: 97-104% LTHR (bulk at 99-101%, first km drags avg down)
- 10K: 95-102% LTHR
- HM: 94-100% LTHR  
- Marathon: 90-93% LTHR

**Paul's thresholds (set by Paul, not optimizer):**
```
3K: 1.01  (180 bpm)
5K: 1.00  (178 bpm)
10K: 1.00 (178 bpm)
HM: 0.97  (172 bpm)
Marathon: 0.93 (165 bpm)
```

Interpolated: 10M: 0.98, 30K: 0.94

### Validated on Ian's data (LTHR 157)

Paul's thresholds catch 21/28 of Ian's 5K races. The 7 missed are at 95-100% LTHR — BUT Ian's race/training HR completely overlaps at 5K (training P90 = 102%, race min = 95%, gap = -6%). No HR threshold can cleanly separate Ian's races from training. Keywords must do the heavy lifting for Ian.

**Ian's LTHR may be too low** — regularly hits 170+ (110% LTHR) in "Morning Run" training sessions. Worth flagging.

### Architecture decision: thresholds in athlete.yml

Thresholds are stored as `%LTHR` in `athlete.yml` under `athlete.race_hr_thresholds_pct`. Pipeline defaults match Paul's values. Athletes override in their YAML; those not set fall back to defaults.

### Parkrun ≠ Race

`parkrun=1` marks the event (always set if it's a parkrun). `race_flag` is set independently by HR. A jogged parkrun is `parkrun=1, race_flag=0`. The `race(low)` verdict (race keyword but easy HR) now gives `race_flag=0` with a REVIEW note.

---

## Files produced / modified this session

All in `/mnt/user-data/outputs/`:

| File | Deploy to | What changed |
|---|---|---|
| `classify_races.py` | root | Thresholds from athlete.yml, race(low)→flag=0, parkrun decoupled, championship override, mästerskap keyword, buggy decisive anti, monotonic thresholds |
| `paul_pipeline.yml` | `.github/workflows/paul_pipeline.yml` | 3 fixes (see below) |
| `athlete.yml` | root (Paul A001) | Added `race_hr_thresholds_pct` block |
| `A005_athlete.yml` | `athletes/A005/athlete.yml` | Added thresholds + fixed easy_rf (hr_max was 0, hr_min was 137→120) |
| `IanLilley_athlete.yml` | `athletes/IanLilley/athlete.yml` | Added thresholds commented out + LTHR note |
| `NadiJahangiri_athlete.yml` | `athletes/NadiJahangiri/athlete.yml` | Added thresholds commented out |
| `SteveDavies_athlete.yml` | `athletes/SteveDavies/athlete.yml` | Added thresholds commented out + fixed easy_rf (hr_max was 0→141) |

---

## PaulTest pipeline fixes (paul_pipeline.yml)

### Fix 1: fits.zip not uploaded to Dropbox after INITIAL mode

**Root cause**: Upload step only fired when `steps.fetch.outputs.new_runs == 'true'`. During INITIAL, fetch_fit_files runs BEFORE initial_data_ingest (which creates fits.zip from Strava export), so the flag is never set.

**Fix**: Upload condition now includes `|| env.PIPELINE_MODE == 'INITIAL' || env.PIPELINE_MODE == 'FULL'`.

### Fix 2: Missing recent intervals.icu activities

**Root cause**: fetch_fit_files.py runs at step 7 (before initial_data_ingest at step 13). In INITIAL mode, fits.zip doesn't exist yet at step 7, so fetch can't merge new FITs. After initial_data_ingest creates fits.zip from Strava, there's no second fetch to pick up intervals.icu runs after the Strava export date.

**Fix**: Added step 14 "Fetch intervals.icu FITs after ingest (INITIAL only)" — runs fetch_fit_files.py again after initial_data_ingest, so it can merge intervals.icu FITs into the newly created fits.zip.

### Fix 3: classify_races.py not in workflow

**Root cause**: Script existed in repo but was never wired into any workflow. Was run manually during onboarding.

**Fix**: Added step 18 "Classify races" between GAP power and StepB. Runs on FULL/INITIAL mode with `--from-master --skip-if-classified` (won't overwrite athlete's manual edits on subsequent runs).

### Updated step order:
```
 1-9.   Setup, fetch, gate
10-12.  Download data, sync intervals.icu
13.     Initial data ingest (INITIAL only)
14.     Fetch intervals.icu FITs after ingest (INITIAL only) ← NEW
15-16.  Rebuild from FIT files (FULL or UPDATE)
17.     Add GAP power
18.     Classify races (FULL/INITIAL) ← NEW
19.     StepB post-processing
20.     Generate dashboard
21.     Save persec cache
22.     Upload fits.zip (INITIAL/FULL/new FITs) ← FIXED condition
23-26.  Upload results, deploy Pages, summary
```

---

## What to do next

### Immediate: Deploy and re-run PaulTest

1. Copy files to repo at paths listed above
2. Push to GitHub
3. Trigger paul_pipeline with mode=INITIAL
   - This rebuilds fits.zip from Strava export
   - Fetches intervals.icu FITs after the Strava cutoff
   - Runs classify_races.py on the full Master
   - Runs StepB + dashboard
4. Check dashboard for: race classifications, recent runs from intervals.icu, correct race HR thresholds

### Then: Paul's main pipeline

- Paul's `athlete.yml` has the new `race_hr_thresholds_pct` block
- classify_races.py isn't in Paul's main `pipeline.yml` (only paul_pipeline.yml for PaulTest)
- Decide whether to add it to main pipeline too, or keep manual for Paul since his overrides are already curated

### Then: Other athletes

- Ian/Nadi/Steve workflows also need the classify_races step if they don't have it
- Their athlete.yml thresholds are commented out (defaults apply)
- Ian's LTHR should be reviewed (157 seems low for someone hitting 170+ regularly)

### Curation fixes for Paul's main data

~15 races miscurated as training in Paul's Master should be fixed in activity_overrides.xlsx:
- Stockholms Kvartsmarathon (98% LTHR, 10.4km)
- Midnattsloppet 2015 (97% LTHR, 10km)
- Midnattsloppet 2024 (99% LTHR, 5km)
- Assembly League Crystal Palace (98% LTHR, 4.9km)
- Stockholm's Bästa Race #3 and #7 (100% LTHR, 10km)
- Big Half 2018 (98% LTHR, 21.1km)
- CityRun hour challenge (97% LTHR, 14.7km)
- Brooks 5k in 18'33 (94% LTHR, 4.9km)
- Spåret 5000 (101% LTHR, 5.1km)
- Burgess Park parkrun (97% LTHR, 5km)

### PaulTest easy_rf fixes

PaulTest and Steve had `easy_rf.hr_max: 0` which would break easy RF calculation. Fixed to 148 (Paul) and 141 (Steve = round(157 * 0.90)). PaulTest also had `hr_min: 137` which was too high — fixed to 120.

---

## classify_races.py architecture summary

### Decision tree
1. **Named race override** (VLM, London Marathon, etc.) + no anti → race if HR ≥ named threshold
2. **Race keyword** (parkrun, race, championship, etc.) + no anti → race if HR ≥ named threshold
3. **Anti keyword only** (tempo, interval, easy, etc.) → training
4. **Both race + anti** → check decisive keywords:
   - Championship/mästerskap overrides anti → race if HR ≥ threshold
   - Decisive anti (sandwich, tempo, session, interval, FKS, buggy, prep) → training
   - Otherwise HR decides
5. **No keywords** → HR-only at full threshold

### HR discount for named races
Named threshold = threshold - 5%. So for 5K: unnamed needs 100% LTHR, named parkrun needs 95% LTHR.

### Parkrun handling
- `parkrun` keyword only counts as race keyword at 5K distance
- At non-5K distances (sandwiches), only NON_PARKRUN_RACE_KW matches count
- `is_parkrun()` caps at 5.1km distance
- parkrun=1 always set if it's a parkrun event, race_flag set by HR independently

### Validation results on Paul's data
- TP=305, FP=20, FN=15
- Precision 93.8% (all 20 FP are genuinely miscurated races)
- Real false positives: 0

---

## For next Claude

"classify_races.py is calibrated with athlete-specific HR thresholds from athlete.yml. PaulTest needs INITIAL re-run after deploying 3 workflow fixes (fits.zip upload, post-ingest fetch, classify_races step). All athlete.yml files updated with race_hr_thresholds_pct. See HANDOVER_CLASSIFY_RACES.md."
