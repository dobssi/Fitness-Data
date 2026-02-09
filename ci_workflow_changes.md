# CI Workflow Changes: Fix TotalHistory.zip and pending_activities sync

## Problem
CI downloads TotalHistory.zip, fetches new FITs, appends them, processes them — 
but never uploads the updated zip back to Dropbox. This means:
- Dropbox TotalHistory.zip falls behind (missing recent FITs)
- Local full rebuilds miss runs only processed by CI
- pending_activities.csv dispatch metadata is lost after CI run

## Changes to `.github/workflows/pipeline.yml`

### In the "Upload results to Dropbox" step, add these items:

```yaml
      - name: Upload results to Dropbox
        run: |
          python ci/dropbox_sync.py upload \
            --items Master_FULL_GPSQ_ID.xlsx \
                    Master_FULL_GPSQ_ID_simS4.xlsx \
                    Master_FULL_GPSQ_ID_post.xlsx \
                    athlete_data.csv \
                    re_model_s4_FULL.json \
                    sync_state.json \
                    TotalHistory.zip \
                    fit_sync_state.json \
                    pending_activities.csv \
                    activity_overrides.xlsx
```

The new items are:
- `TotalHistory.zip` — keeps Dropbox in sync with CI (chunked upload handles >150MB)
- `fit_sync_state.json` — so next CI run knows what was already fetched
- `pending_activities.csv` — preserves dispatch metadata names
- `activity_overrides.xlsx` — preserves dispatch metadata overrides

### Also add to download step (if not already there):

```yaml
      - name: Download pipeline data from Dropbox
        run: |
          python ci/dropbox_sync.py download \
            --items Master_FULL_GPSQ_ID.xlsx \
                    Master_FULL_GPSQ_ID_simS4.xlsx \
                    Master_FULL_GPSQ_ID_post.xlsx \
                    athlete_data.csv \
                    activities.csv \
                    activity_overrides.xlsx \
                    Master_Rebuilt.xlsx \
                    re_model_s4_FULL.json \
                    sync_state.json \
                    fit_sync_state.json \
                    pending_activities.csv \
                    TotalHistory.zip
```

New additions to download: `fit_sync_state.json`, `pending_activities.csv`
(TotalHistory.zip should already be in the download list)

## Impact
- TotalHistory.zip upload is ~230MB but uses chunked upload (50MB chunks)
- Adds ~30-60 seconds to CI upload step
- Keeps Dropbox and CI perfectly in sync
- Local Daily_Update will see the same FITs as CI
