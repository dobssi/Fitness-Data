# v50 Handover — Code Review & Optimisations

## Summary

v50 focuses on pipeline performance optimisations and code quality improvements. The headline change is a **quick mode** that skips Steps 2-4 when UPDATE finds no new FIT files, reducing the typical daily run from ~11 minutes to ~2 minutes. Additional optimisations target the cache index startup time and StepB's iterrows-heavy processing.

## Changes

### 1. Quick Mode — No new runs detected → skip Steps 2-4

**Problem:** UPDATE mode always ran all 4 pipeline steps even when Step 1 (rebuild) found 0 new FIT files. This wasted ~9 minutes of processing time on GitHub Actions for no-op runs.

**Solution:**
- `rebuild_from_fit_zip.py` now returns exit code **2** (instead of 0) when append mode finds no new FITs
- `run_pipeline.py` detects exit code 2 in UPDATE mode and skips Steps 2-4 entirely
- Dashboard generation still runs (it reads existing `_post.xlsx`)
- FULLPIPE mode is unaffected (always runs all steps)

**Impact:** UPDATE with no new runs: ~11 min → ~2 min (GitHub), ~3 min → <1 min (local)

### 2. Fast Cache Index — `_cache_index.json`

**Problem:** Every step that needs per-second cache data (StepA, StepB, build_re_model) opens ALL ~3000 .npz files to read their timestamp metadata for the bisect index. This takes ~30 seconds each time, repeated across 3 steps = ~90 seconds wasted.

**Solution:**
- New `_cache_index.json` file maintained alongside .npz files
- Maps `{filename_stem: epoch_timestamp}` for instant lookup
- Written/updated when new .npz files are created during rebuild
- Auto-rebuilt from scratch if missing or stale (>5% drift from .npz count)
- StepA and StepB import `build_cache_index_fast()` from rebuild module
- Fallback to slow .npz scan if import fails (backward compatible)

**New functions in `rebuild_from_fit_zip.py`:**
- `_load_cache_index(cache_dir)` — Load JSON index
- `_save_cache_index(cache_dir, index)` — Write JSON index
- `_update_cache_index_entry(cache_dir, npz_filename, epoch_s)` — Add/update single entry
- `rebuild_cache_index(cache_dir)` — Full rebuild from .npz scan
- `build_cache_index_fast(cache_dir)` — Public API: returns sorted (keys, paths) for bisect

**Impact:** Cache index build: ~30s → <1s per step

### 3. Vectorized Temp_Trend Calculation

**Problem:** Temp_Trend was calculated in a separate `iterrows()` loop using `mask = (dfm['date'] >= cutoff_date)` per row — O(n²) complexity with ~3000 rows.

**Solution:**
- Convert dates to epoch seconds once
- Use `np.searchsorted` to find window boundaries in O(log n) per row
- Eliminated one full iterrows pass (~3000 iterations)
- Merged into the existing adjustment factor pass

**Impact:** Temp_Trend + adjustment calc: ~2 passes → 1 pass, ~30% faster for this section

### 4. Dead Code Removal

**`rebuild_from_fit_zip.py`:** Removed ~40 lines of unreachable code after `return wx` in `compute_weather_averages()`. This was an old version of the function body left behind after refactoring to use `compute_weather_averages_from_hourly()`.

### 5. Deprecation Fix

**`rebuild_from_fit_zip.py`:** Replaced `pd.Timestamp.utcnow()` with `pd.Timestamp.now('UTC')` to fix the pandas deprecation warning.

### 6. Version Bumps

All files updated from v49 to v50:
- `rebuild_from_fit_zip.py` — header, cache index additions, dead code removal
- `StepB_PostProcess.py` — header, fast cache index, merged Temp_Trend pass
- `StepA_SimulatePower.py` — header, fast cache index
- `run_pipeline.py` — PIPELINE_VER=50, quick mode logic
- `Run_Full_Pipeline.bat` — version refs
- `StepB_PostProcess.bat` — version refs

## Performance Summary

| Scenario | v49 | v50 | Saving |
|----------|-----|-----|--------|
| UPDATE, no new runs (GitHub) | ~11 min | ~2 min | **9 min** |
| UPDATE, no new runs (local) | ~3 min | <1 min | **2 min** |
| UPDATE, 1 new run (GitHub) | ~11 min | ~11 min | Cache index ~90s |
| FULLPIPE rebuild (GitHub) | ~2h 42m | ~2h 40m | Cache index ~90s |
| Cache index build per step | ~30s | <1s | **~29s × 3 steps** |

## Migration

1. Copy v50 files to pipeline directory
2. Run any mode — `_cache_index.json` will auto-generate on first run
3. No full rebuild required (no algorithm changes)

## Files Changed

- `rebuild_from_fit_zip.py` — Cache index infrastructure, dead code removal, deprecation fix
- `StepB_PostProcess.py` — Fast cache index, vectorized Temp_Trend, merged iterrows passes
- `StepA_SimulatePower.py` — Fast cache index
- `run_pipeline.py` — Quick mode (exit code 2 handling), version bump
- `Run_Full_Pipeline.bat` — Version refs
- `StepB_PostProcess.bat` — Version refs
- `v50_handover.md` — This file (replaces placeholder)

## TODO for v51+

From v49/v50 TODO list, still outstanding:
1. **DataFrame fragmentation** in StepB — could further reduce by collecting per-row results in dicts and bulk-assigning. The `PerformanceWarning` should be reduced by v50 changes but not eliminated.
2. **Weather cache persistence on GitHub** — SQLite DB to Dropbox for faster re-runs
3. **Persec cache on Dropbox** — tar.gz upload for resilience beyond GH Actions cache
4. **Scheduled daily runs** — cron trigger ready to enable
5. **Dashboard-only workflow** — regenerate without pipeline run
6. **Smarter UPDATE mode** — detect code changes since last run
