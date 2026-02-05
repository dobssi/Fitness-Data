# Pipeline Automation Plan — Off-Laptop, intervals.icu Integration

## Current State (v47, laptop)

```
[Garmin Watch] → [manual FIT export] → TotalHistory.zip
                                              │
[BFW Excel] → export_athlete_data.py → athlete_data.csv
                                              │
    Run_Full_Pipeline_v47.bat ────────────────┤
    ├── rebuild_from_fit_zip_v47.py           │ (reads TotalHistory.zip)
    ├── build_re_model_s4.py                  │
    ├── StepA_SimulatePower_v47.py            │
    ├── StepB_PostProcess_v47.py              │ (reads athlete_data.csv)
    ├── generate_dashboard.py                 │
    └── push_dashboard.py                     │
                                              │
    Outputs: Master_FULL_GPSQ_ID_post.xlsx    │
             persec_cache_FULL_v47/            │
             dashboard → GitHub Pages         │
```

**Manual steps today:**
1. Export FIT files from Garmin Connect → copy to TotalHistory folder → recreate zip
2. Export weight/TSS from BFW Excel → athlete_data.csv  
3. Run batch file on laptop
4. Activity overrides: edit activity_overrides.xlsx manually

---

## Target State (automated)

```
[Garmin Watch] → [Garmin Connect] → [intervals.icu] ← auto-sync (minutes)
                                           │
                                    GitHub Actions trigger
                                           │
                         ┌─────────────────┤
                         ▼                 ▼
              intervals.icu API    Dropbox API
              ├─ new FIT files     ├─ persec_cache/
              ├─ weight data       ├─ Master Excel
              └─ non-run TSS       └─ athlete_data.csv
                         │                 │
                         ▼                 ▼
                    GitHub Actions runner
                    ├── fetch_new_activities.py  (NEW)
                    ├── rebuild_from_fit_zip_v47.py (or folder mode)
                    ├── build_re_model_s4.py
                    ├── StepA_SimulatePower_v47.py
                    ├── StepB_PostProcess_v47.py
                    ├── generate_dashboard.py
                    └── push_dashboard.py
                              │
                    ┌─────────┤
                    ▼         ▼
              Dropbox     GitHub Pages
              (updated    (dashboard)
               cache +
               Master)
```

**Manual steps remaining:**
- Activity overrides (race flags, surfaces, distances) — via YAML in repo, editable on mobile

---

## Phased Implementation

### Phase 1: intervals.icu as Data Source (replace BFW)
**Goal:** Eliminate BFW dependency. Pipeline still runs on laptop but reads from intervals.icu.

**Deliverables:**
1. `intervals_fetch.py` — library module for intervals.icu API calls
2. `sync_athlete_data.py` — replaces `export_athlete_data.py` 
   - Weight: intervals.icu wellness API (Jun 2023+), athlete_data.csv (pre-2023)
   - Non-running TSS: intervals.icu activities with hrTSS for all sport types
   - Output: same `athlete_data.csv` format (backward compatible)
3. Updated batch file to call `sync_athlete_data.py` before pipeline

**intervals.icu API calls needed:**
| Endpoint | Purpose | Frequency |
|----------|---------|-----------|
| `GET /athlete/{id}/wellness?oldest=...&newest=...` | Weight data | Each run |
| `GET /athlete/{id}/activities?oldest=...&newest=...` | Non-running TSS (hrTSS by sport) | Each run |

**Weight merge strategy:**
- Pre-2023: keep existing athlete_data.csv values (manual BFW export, one-time)
- 2023+: overwrite with intervals.icu wellness data on each sync
- Interpolation unchanged (handled in StepB)

**Non-running TSS merge strategy:**
- All years: intervals.icu activities API returns `icu_training_load` (hrTSS) for every sport
- Filter: exclude `Run` and `VirtualRun` (those are handled by the pipeline itself)
- Sum by date → `non_running_tss` column in athlete_data.csv

### Phase 2: FIT File Auto-Pull (replace manual export)
**Goal:** No more manual FIT file export. Pipeline pulls new files from intervals.icu.

**Deliverables:**
1. `fetch_fit_files.py` — downloads new FIT files from intervals.icu
   - Tracks last-synced activity ID in a state file
   - Downloads only new activities since last sync
   - Saves to TotalHistory folder (or equivalent)
2. Modify `rebuild_from_fit_zip_v47.py` to accept `--fit-dir` as alternative to `--fit-zip`
   - Or: script creates a temporary zip from the folder (simpler, no rebuild changes)
3. `add_run.py` updated to optionally skip FIT copy (file already in intervals.icu)

**intervals.icu API calls:**
| Endpoint | Purpose |
|----------|---------|
| `GET /athlete/{id}/activities?oldest=...&newest=...` | List new activities |
| `GET /activity/{id}/file` | Download FIT file (gzip compressed) |

**Incremental strategy:**
- State file: `sync_state.json` → `{"last_activity_id": "i12345_67890", "last_sync": "2026-02-04T12:00:00Z"}`
- On each run: fetch activities newer than last_sync, download their FIT files
- After successful pipeline run: update state file

**Strava activity names:**
- Currently from `activities.csv` (Strava export)
- intervals.icu has activity names too (synced from Garmin)
- Phase 2 option: use intervals.icu names, or keep Strava CSV as override layer

### Phase 3: GitHub Actions Automation
**Goal:** Pipeline runs in the cloud. No laptop required.

**Architecture:**
```yaml
# .github/workflows/pipeline.yml
name: Run Pipeline
on:
  workflow_dispatch:         # Manual trigger (button press)
    inputs:
      mode:
        type: choice
        options: [UPDATE, FULL]
        default: UPDATE
  schedule:
    - cron: '0 20 * * *'   # Daily at 20:00 UTC (22:00 Stockholm)

jobs:
  pipeline:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: pip install pandas numpy openpyxl fitparse requests
      
      - name: Download cache from Dropbox
        env:
          DROPBOX_TOKEN: ${{ secrets.DROPBOX_TOKEN }}
        run: python ci/dropbox_sync.py download
      
      - name: Fetch new activities from intervals.icu
        env:
          INTERVALS_API_KEY: ${{ secrets.INTERVALS_API_KEY }}
        run: python fetch_fit_files.py
      
      - name: Sync athlete data
        env:
          INTERVALS_API_KEY: ${{ secrets.INTERVALS_API_KEY }}
        run: python sync_athlete_data.py
      
      - name: Run pipeline
        run: python run_pipeline.py ${{ github.event.inputs.mode || 'UPDATE' }}
      
      - name: Upload results to Dropbox
        env:
          DROPBOX_TOKEN: ${{ secrets.DROPBOX_TOKEN }}
        run: python ci/dropbox_sync.py upload
      
      - name: Deploy dashboard
        run: |
          python generate_dashboard.py
          # Push to gh-pages branch
```

**Key considerations:**

1. **Compute time:** GitHub Actions free tier = 2,000 min/month. UPDATE run should take 2-5 min. Full rebuild ~30-60 min. Daily updates = ~150 min/month, well within budget.

2. **Cache transfer:** The persec_cache folder is hundreds of MB. For UPDATE mode, we only need to download the cache, process new files, upload the diff. For FULL rebuild, we need the entire cache.
   - Option A: Dropbox API (download/upload zip of cache)
   - Option B: GitHub Actions cache (14-day retention, 10GB limit) — good for hot cache
   - Recommended: Dropbox as primary, GH Actions cache as optimization

3. **Secrets needed:**
   - `INTERVALS_API_KEY`: `tgbv1rdzgciobj8meem47cml`
   - `INTERVALS_ATHLETE_ID`: `i224884`
   - `DROPBOX_TOKEN`: OAuth2 token with file access
   - `GITHUB_TOKEN`: Built-in, for Pages deployment

4. **Python replacement for .bat:** Create `run_pipeline.py` that does the same as `Run_Full_Pipeline_v47.bat` but cross-platform.

5. **Error handling:** GitHub Actions should notify on failure (email or Slack webhook).

### Phase 4: Mobile Override Editing
**Goal:** Edit activity overrides from phone after a race.

**Options (ranked):**
1. **YAML in repo + GitHub mobile app** — edit `overrides.yml`, commit triggers re-run
   - Pro: Simple, version controlled, triggers pipeline automatically
   - Con: GitHub mobile editing is clunky for structured data
   
2. **Simple web form → GitHub API** — static HTML page that commits to repo
   - Pro: Clean mobile UX
   - Con: More to build and maintain

3. **Telegram/Signal bot** — message bot with structured commands
   - Pro: Already on phone, natural
   - Con: Most complex to build

**Recommended: Option 1** (YAML + GitHub mobile), with Option 2 as a future enhancement.

**Override YAML format:**
```yaml
# activity_overrides.yml
overrides:
  - file: "2026-02-01_10-30-00.FIT"
    race: true
    distance_km: 42.195
    notes: "Stockholm Marathon"
    
  - file: "2026-01-25_09-00-00.FIT"
    race: true
    parkrun: true
    distance_km: 5.0
    
  - file: "2026-01-20_15-00-00.FIT"
    surface: TRAIL
    surface_adj: 1.035
```

---

## Detailed API Reference (intervals.icu)

### Authentication
```python
session = requests.Session()
session.auth = ("API_KEY", "tgbv1rdzgciobj8meem47cml")
```

### Activity List (for TSS + FIT download)
```
GET /api/v1/athlete/i224884/activities
  ?oldest=2026-01-01&newest=2026-02-04
```
Returns JSON array. Key fields per activity:
- `id`: activity ID (e.g., `"i224884:1234567890"`)
- `start_date_local`: ISO datetime
- `type`: sport type (`"Run"`, `"Ride"`, `"Workout"`, etc.)
- `icu_training_load`: hrTSS (the non-running TSS value we need)
- `name`: activity name
- `distance`: distance in meters
- `moving_time`: seconds
- `elapsed_time`: seconds

### FIT File Download
```
GET /api/v1/activity/{activity_id}/file
```
Returns gzip-compressed FIT file. Decompress with `gzip.decompress()`.

### Wellness (Weight)
```
GET /api/v1/athlete/i224884/wellness
  ?oldest=2023-06-01&newest=2026-02-04
```
Returns JSON array. Key fields:
- `id`: date string (e.g., `"2024-01-15"`)
- `weight`: weight in kg (float, or null)

### Webhook (future, for real-time trigger)
```
POST /api/v1/athlete/i224884/webhook
{
  "url": "https://api.github.com/repos/..../dispatches",
  "events": ["ACTIVITY_CREATED"]
}
```

---

## Migration Checklist

### Phase 1 Prerequisites
- [x] intervals.icu account confirmed working (3,896 activities)
- [x] API key and athlete ID confirmed
- [x] Weight data availability mapped (Jun 2023+)
- [x] Non-running sport types identified (15 types)
- [ ] Fix sync gap: Sep 2025 – Jan 2026 (Garmin resync pending)
- [ ] Freeze pre-2023 weight data in athlete_data.csv (one-time BFW export)

### Phase 1 Deliverables
- [ ] `intervals_fetch.py` — API client library
- [ ] `sync_athlete_data.py` — weight + TSS sync
- [ ] Test: sync output matches BFW export (diff check)
- [ ] Update batch file to call sync before pipeline

### Phase 2 Deliverables
- [ ] `fetch_fit_files.py` — incremental FIT download
- [ ] `sync_state.json` — tracking last sync point
- [ ] Test: full pipeline with intervals.icu-sourced FIT files
- [ ] Verify: no regressions vs laptop pipeline output

### Phase 3 Deliverables
- [ ] `run_pipeline.py` — cross-platform pipeline orchestrator
- [ ] `ci/dropbox_sync.py` — cache upload/download
- [ ] `.github/workflows/pipeline.yml` — GitHub Actions workflow
- [ ] Secrets configured in GitHub repo
- [ ] Test: UPDATE run completes in <10 min
- [ ] Test: dashboard deploys correctly

### Phase 4 Deliverables
- [ ] `activity_overrides.yml` — YAML format override file
- [ ] `convert_overrides.py` — XLSX → YAML migration
- [ ] StepB updated to read YAML overrides
- [ ] Test: override edit → commit → pipeline trigger → updated output

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| intervals.icu API rate limits | Pipeline fails mid-run | Implement backoff, batch requests |
| Sync gap (Sep–Jan 2026) not resolved | Missing 139 days of data | Manual FIT upload as fallback |
| GitHub Actions compute limit | Monthly quota exceeded | Limit to triggered runs, not scheduled |
| Dropbox token expiry | Cache sync fails | Use refresh token, alert on 401 |
| Cache size exceeds GH Actions limits | Slow transfers | Compress cache, use incremental sync |
| FIT file format changes | Parse failures | Pin fitparse version, error alerting |
| Strava names lost | Activities unnamed | Use intervals.icu names as fallback |

---

## Recommended Starting Point

**Build Phase 1 first:** `intervals_fetch.py` + `sync_athlete_data.py`

This delivers immediate value (no more BFW dependency) with minimal risk (pipeline still runs on laptop, just with a new data source). It also validates the intervals.icu API integration before committing to the bigger Phase 2/3 changes.

The `intervals_fetch.py` module becomes the foundation that Phase 2 (FIT download) and Phase 3 (GitHub Actions) build on.
