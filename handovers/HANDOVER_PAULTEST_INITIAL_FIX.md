# Handover: Paul Test INITIAL Pipeline Fix + Log Analysis
## Date: 2026-03-06

---

## Summary

Analysed CI logs from Paul Test INITIAL run (workflow 59584041660). Diagnosed why the master stopped at 2026-02-24 despite runs existing through 2026-03-05. Root cause was a race condition in the INITIAL workflow between the first `fetch_fit_files.py` and `initial_data_ingest.py`. Applied one-line fix to `paul_pipeline.yml`.

---

## The bug

### Symptom
Paul Test INITIAL completed successfully (6h12m, both jobs green), but Master_FULL only contained runs through 2026-02-24. The 10 most recent runs (Feb 25 – Mar 5) were missing.

### Root cause: FIT_downloads pollution before Strava ingest

The INITIAL workflow had three FIT-fetching stages:

1. **First fetch** (line 132): `fetch_fit_files.py` downloads 25 recent FITs from intervals.icu into `FIT_downloads/` directory
2. **Strava ingest** (line 212): `initial_data_ingest.py` extracts Strava export → builds `fits.zip` with 3094 FIT files (Strava export ends at Feb 24)
3. **Second fetch** (line 220): `fetch_fit_files.py --full` scans all intervals.icu history, finds 106 FITs missing from `fits.zip`, appends them

The problem: in step 3, the 25 recent runs from step 1 were already sitting in `FIT_downloads/`. The deduplication logic in `fetch_fit_files.py` saw them as "already present" (3085 of 3191 total) and didn't re-download or append them to `fits.zip`. The 106 it did add were older gaps (pre-2014 files, short runs, etc.) that existed on intervals.icu but not in the Strava export.

Result: `fits.zip` had 3200 files but was missing the 25 most recent runs. Rebuild processed all 3200 → master ends at Feb 24.

### The fix

One-line change to `paul_pipeline.yml`:

```yaml
# Before:
if: ${{ github.event.inputs.sync != 'false' }}

# After:  
if: ${{ github.event.inputs.sync != 'false' && env.PIPELINE_MODE != 'INITIAL' }}
```

This skips the first fetch for INITIAL mode. The first fetch is only needed for UPDATE mode (to detect new runs and decide whether to continue). For INITIAL, the workflow always continues via `workflow_dispatch` gate, and the second fetch (post-ingest, with `--full`) handles all intervals.icu FIT downloads from scratch with an empty `FIT_downloads/` directory.

### Why the gate still works

The "Decide whether to continue" step (line 153) checks:
1. If `workflow_dispatch` → always continue (INITIAL is always dispatched)
2. Else if `steps.fetch.outputs.new_runs == true` → continue
3. Else → stop

Since INITIAL is always `workflow_dispatch`, it hits branch 1 regardless of the fetch step being skipped.

---

## Log analysis details

### Rebuild job timeline (from logs)
- 05:39 — Job starts, fits.zip not found on Dropbox (expected — data was deleted)
- 05:39 — First fetch: 25 FITs downloaded to FIT_downloads/
- 05:40 — Strava ingest: extracts 3931 FIT + 282 GPX + 51 TCX from export zip
  - 3094 running FIT files → fits.zip
  - 1 GPX converted (with HR), 265 GPX skipped (no HR or empty)
  - 0 TCX converted (all XML parse errors)
- 06:49 — Second fetch (--full): 3191 runs on intervals.icu, 3085 already in FIT_downloads, 106 new
  - 733 matched by timestamp proximity (Strava/intervals.icu timezone dedup)
  - 106 appended to fits.zip → 3200 total
- 06:50–09:52 — Rebuild processes 3200 FITs (3h02m)
  - 5 FitHeaderError failures (old corrupted files)
  - 3 non-running skipped (cycling)
  - 1 IndexError on manual 332-byte placeholder FIT
  - 3192 rows in master, ending at 2026-02-24
- 09:52 — Weather: 319 groups, 3162 filled, 30 skipped (no GPS)
- 09:54 — GAP power: 783 runs with GAP, 2409 with Stryd, 3187 total
- 09:54 — classify_races: 223 races, 555 training, 205 parkruns, 30 surfaces detected

### Key stats from this INITIAL run
- **Total FIT files processed**: 3200 (3094 Strava + 106 intervals.icu gap-fill)
- **Master rows**: 3192 (8 failures/skips)
- **NPZ cache**: 3192 files written, saved to GH Actions cache
- **Persec cache upload**: Skipped in INITIAL mode (by design — timeout protection)
- **Weather cache**: Built from scratch, 3162 locations filled
- **Race classification**: 223 races detected (first pass, no predictions available)

---

## Files delivered this session

### Modified
- **`paul_pipeline.yml`** — First fetch skipped for INITIAL mode (one-line condition change)

### Previously delivered (still needed for next INITIAL)
- **`rebuild_from_fit_zip.py`** — lat/lon in NPZ, float32/uint8 dtype optimisation, gps_bbox_m2
- **`classify_races.py`** — GPS bbox track/indoor detection, elapsed/moving ratio demotion, no-HR pace fallback
- **`StepB_PostProcess.py`** — auto_exclude + indoor→INDOOR_TRACK normalisation + race preservation
- **`generate_dashboard.py`** — auto_exclude filter with race preservation

---

## What to expect from next INITIAL run

With the fix deployed, the next INITIAL should:

1. **Skip first fetch** — FIT_downloads stays empty
2. **Strava ingest** — builds fits.zip from export (3094 FITs)
3. **Second fetch (--full)** — finds ALL intervals.icu runs not in fits.zip, including recent ones
4. **Rebuild** — processes all FITs, master should extend to current date

### New features active in this rebuild
- **NPZ lat/lon arrays** (float32, ~0.03m precision) — enables future GPS route maps
- **NPZ dtype optimisation** — float32/uint8 instead of all-float64, ~30% smaller cache
- **gps_bbox_m2** — GPS bounding box area computed per run
- **GPS bbox track detection** — classify_races auto-detects TRACK (<50,000 m²) and INDOOR_TRACK (<5,000 m² or no GPS)
- **Elapsed/moving ratio demotion** — races with ratio >1.5 demoted to training
- **Auto-exclude** — 67 junk runs flagged (but races preserved)
- **No-HR pace fallback** — allows no-HR races through if at race distance + race pace

### Expected timing
- Rebuild job: ~5h (FIT ingest + full rebuild + weather fetch)
- StepB job: ~1h (double-pass for INITIAL: StepB → re-classify with predictions → StepB second pass + dashboard)
- Total: ~6h (previous INITIAL took 6h12m)

### Persec cache upload
The rebuild job skips persec cache upload to Dropbox for INITIAL (timeout protection — it's already close to the 350-min limit). The stepb_deploy job uploads with `--cache-dir` unconditionally, so persec cache is synced to Dropbox during stepb_deploy. No follow-up UPDATE run needed for cache sync.

---

## Open items from this + prior sessions

### Immediate (ready to deploy)
- All code changes from prior session are in the delivered files — no separate deployment needed

### TODO list (unchanged from prior session)
- Race tooltip on table rows (#23)
- GPS route map feature (lat/lon now in NPZ)
- Activity search + override editor (own session)
- Scheduled workout planner (own session)
- Race History card effort mode toggle
- NPZ scanning move to StepB for performance
- Surface-specific trail specificity metric
