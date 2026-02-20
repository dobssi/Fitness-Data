# Handover: Ian CI Pipeline + Multi-Athlete Fixes
## Date: 2026-02-19

---

## Summary

Got Ian's pipeline running on GitHub Actions end-to-end. Multiple bugs found and fixed relating to path mismatches between local and CI environments, Python version compatibility, and NPZ file matching for zone data.

---

## What was done

### 1. GitHub Actions workflow created (`ian_pipeline.yml`)
- Manual dispatch with FULL / FROM_STEPB / DASHBOARD modes
- Separate concurrency group (`ian-pipeline`) — can run alongside Paul's pipeline
- Timeout 240 minutes
- `actions/cache/restore@v4` + `actions/cache/save@v4` with `if: always()` for persec cache
- Upload results on failure too (`if: always()`)
- GitHub Pages deployment to `docs/ian/` → `dobssi.github.io/Fitness-Data/ian/`

### 2. Bugs found and fixed

**a) `master_template.xlsx` not in repo**
- FIT rebuild needs this as a template for writing Master_FULL.xlsx
- Was gitignored by `Master_*.xlsx` pattern
- Fix: added `!master_template.xlsx` exception to `.gitignore`

**b) Python 3.11 vs 3.12 f-string parsing (SyntaxError)**
- `generate_dashboard.py` uses nested triple-quoted strings inside f-strings
- Python 3.12 (PEP 701) handles this; 3.11 does not
- Fix: changed `python-version: '3.12'` in both `ian_pipeline.yml` and `pipeline.yml`

**c) `⚖️` emoji in generate_dashboard.py**
- Composite Unicode character caused SyntaxError on CI
- Fix: removed emoji, line 2828 now just says "Weight"

**d) `ATHLETE_CONFIG_PATH` not set globally in workflow**
- Only the dashboard step had it; StepB didn't see Ian's `athlete.yml`
- StepB defaulted to Stryd mode → used Garmin wrist power for TSS
- Garmin power values were garbage (5,178W!) → inflated CTL from 45 to 84
- Fix: moved `ATHLETE_CONFIG_PATH` to global env block

**e) File paths didn't match Dropbox structure**
- Workflow assumed `output/activity_overrides.xlsx` but actual location is `athletes/IanLilley/activity_overrides.xlsx` (root)
- Same for `persec_cache/` (root, not `output/persec_cache/`)
- Fix: aligned all workflow paths with actual Dropbox structure
- Also fixed `run_ian.bat`: `CACHE_DIR` changed from `%OUTPUT_DIR%\persec_cache` to `%ATHLETE_DIR%\persec_cache`

**f) Dashboard couldn't find NPZ cache**
- Dashboard searches relative to master file location (`output/`)
- NPZ cache is one level up (`athletes/IanLilley/persec_cache/`)
- Fix: added `os.path.join(_master_dir, '..', 'persec_cache')` to search paths

**g) Dashboard NPZ matching by activity ID**
- NPZ files named by activity ID from FIT filename (e.g. `18502680351.npz`)
- Dashboard tried `run_id` column (doesn't exist in Ian's master)
- Fix: added fallback to extract ID from `file` column (`18502680351.fit` → `18502680351`)
- This also matches by date string and date prefix as before

**h) Paul's pipeline.yml wiping Ian's dashboard**
- `peaceiris/actions-gh-pages` without `keep_files: true` deletes all existing files
- Fix: added `keep_files: true` to Paul's pipeline deploy step

### 3. `dropbox_sync.py` — `--local-prefix` argument (from previous session)
- Allows CI to download/upload with path prefix: `--local-prefix "athletes/IanLilley"`
- Maps `data/fits.zip` → `athletes/IanLilley/data/fits.zip` locally
- Backward compatible — Paul's pipeline calls without `--local-prefix`

---

## Current state

### Ian's pipeline on CI: WORKING
- FULL rebuild: ✅ (3,053 runs processed, 158 weather groups, 89 races flagged)
- FROM_STEPB: ✅ (CTL ~45, correct GAP mode TSS)
- Dashboard: ✅ (pending push of NPZ matching fix for zone charts)
- GitHub Pages: `dobssi.github.io/Fitness-Data/ian/`
- NPZ cache: 3,053 files on Dropbox + GitHub Actions cache
- Weather cache: seeded from Paul's cache, all 158 groups fetched

### Files modified
- `.github/workflows/ian_pipeline.yml` — new workflow
- `.github/workflows/pipeline.yml` — added `keep_files: true` for Pages deploy
- `.gitignore` — `!master_template.xlsx` exception, `athletes/*/persec_cache/`
- `ci/dropbox_sync.py` — `--local-prefix` argument
- `generate_dashboard.py` — NPZ cache path search (parent dir), NPZ file matching (file column fallback), removed ⚖️ emoji
- `run_ian.bat` — `CACHE_DIR` path fix

### Dropbox structure (actual)
```
athletes/IanLilley/
  athlete.yml                    # config
  athlete_data.csv               # input (may not exist yet)
  activity_overrides.xlsx        # input — race flags etc.
  data/
    fits.zip                     # 1,598 FIT files
    activities.csv               # Strava export
  persec_cache/                  # NPZ files (3,053)
  output/
    Master_FULL.xlsx             # rebuild output
    Master_FULL_post.xlsx        # StepB output
    dashboard/index.html         # dashboard
    _weather_cache_openmeteo/    # weather SQLite
```

---

## Outstanding / next session

### Immediate
- [ ] **Push generate_dashboard.py NPZ fix** and run DASHBOARD for Ian — zone charts should then show multi-zone bars
- [ ] **Add intervals.icu for Ian** — he has an account now:
  - Athlete ID: `i512712`
  - API key: `5olfh3q3nozqih9p595hunb2s`
  - Add as GitHub secrets: `IAN_INTERVALS_API_KEY`, `IAN_INTERVALS_ATHLETE_ID`
  - Add sync steps to `ian_pipeline.yml` (between download and rebuild)
  - Enables daily automated runs + weight/wellness data

### New athlete (3rd)
- Paul has FIT files for a new athlete ready to onboard
- Can duplicate Ian's setup as template
- Key steps: create Dropbox folder, athlete.yml, activity_overrides.xlsx, workflow file

### Pipeline improvements identified
- [ ] Consider packing persec_cache as tar.gz for Dropbox transfer (6,107 individual file uploads is slow)
- [ ] `run_ian.bat` and `ian_pipeline.yml` should stay perfectly aligned — any path changes need both updated

---

## Key files for checkpoint

Include all of these:
- `.github/workflows/ian_pipeline.yml`
- `.github/workflows/pipeline.yml`
- `.gitignore`
- `ci/dropbox_sync.py`
- `generate_dashboard.py`
- `run_ian.bat`
- `master_template.xlsx`
- All existing v51 scripts (unchanged)
- `athletes/IanLilley/athlete.yml`

---

## For next Claude

"Ian's CI pipeline is working on GitHub Actions. Last fix pending: push updated generate_dashboard.py which adds NPZ file matching via the `file` column (activity ID from FIT filename). After that, add intervals.icu sync for Ian — secrets are ready (ID: i512712). Also have a 3rd athlete with FIT files to onboard. See HANDOVER_IAN_CI.md for full context."
