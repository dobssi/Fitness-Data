# A001 Migration Plan: Root → athletes/A001 + Two-Job Workflow

## Overview

Move Paul (A001) from the legacy single-job `pipeline.yml` (files in repo root) to the standard two-job template in `athletes/A001/`, matching the A005/A006/A007 pattern.

## Folder renames

| Old | New |
|-----|-----|
| `athletes/IanLilley/` | `athletes/A002/` |
| `athletes/NadiJahangiri/` | `athletes/A003/` |
| `athletes/SteveDavies/` | `athletes/A004/` |
| (root files) | `athletes/A001/` |

## A001 folder structure (new)

```
athletes/A001/
  athlete.yml                    # From root athlete.yml (+ new fields)
  activity_overrides.xlsx        # From root activity_overrides.xlsx  
  athlete_data.csv              # From root athlete_data.csv
  output/                       # Created at runtime
    Master_A001_FULL.xlsx
    Master_A001_FULL_post.xlsx
    dashboard/index.html
  persec_cache/                 # Created at runtime
  data/
    TotalHistory.zip            # Curated FIT history
    activities.csv              # Strava activity names
```

## File renames (Dropbox paths must match)

The Dropbox base for A001 becomes: `/Running and Cycling/DataPipeline/athletes/A001`

| Old (root) | New (in A001 folder) |
|------------|---------------------|
| `Master_FULL_GPSQ_ID.xlsx` | `output/Master_A001_FULL.xlsx` |
| `Master_FULL_GPSQ_ID_post.xlsx` | `output/Master_A001_FULL_post.xlsx` |
| `Master_FULL_GPSQ_ID_simS4.xlsx` | `output/Master_A001_FULL_simS4.xlsx` |
| `activity_overrides.xlsx` | `activity_overrides.xlsx` |
| `athlete_data.csv` | `athlete_data.csv` |
| `re_model_s4_FULL.json` | `re_model_s4_FULL.json` |
| `sync_state.json` | `sync_state.json` |
| `fit_sync_state.json` | `fit_sync_state.json` |
| `pending_activities.csv` | `pending_activities.csv` |
| `TotalHistory.zip` | `data/TotalHistory.zip` |
| `activities.csv` | `data/activities.csv` |
| `persec_cache_FULL/` | `persec_cache/` |
| `index.html` | `output/dashboard/index.html` |

## New athlete.yml fields

```yaml
pipeline:
  classify_races_mode: "skip"   # "auto" = run classify_races; "skip" = trust overrides only
```

## Workflow changes

### New: `.github/workflows/paul_collyer_pipeline.yml`
- Two-job template (from paul_pipeline.yml / A005)
- `ATHLETE_DIR: athletes/A001`
- `DB_BASE: /Running and Cycling/DataPipeline/athletes/A001`
- `ATHLETE_CONFIG_PATH: athletes/A001/athlete.yml`
- Secret prefix: (use existing INTERVALS_API_KEY / INTERVALS_ATHLETE_ID)
- Master filename: `Master_A001_FULL.xlsx` / `Master_A001_FULL_post.xlsx`
- **classify_races step**: reads `classify_races_mode` from athlete.yml, skips if "skip"
- **NPZ upload**: Add `--cache-dir` to Dropbox upload step (UPDATE + FULL modes)
- **detect_eras.py**: Add to rebuild step for Stryd mode
- **Info sheet**: Add athlete.yml summary as first sheet in Master

### Retire: `.github/workflows/pipeline.yml`
- Keep as backup initially, remove later

### Update other workflows
- Ian: `ATHLETE_DIR: athletes/A002`, `DB_BASE: .../athletes/A002`
- Nadi: `ATHLETE_DIR: athletes/A003`, `DB_BASE: .../athletes/A003`  
- Steve: `ATHLETE_DIR: athletes/A004`, `DB_BASE: .../athletes/A004`

## Dropbox migration (manual steps)

Paul needs to:
1. Create `athletes/A001/` folder tree in Dropbox
2. Move/copy files from root to new paths
3. Rename Master files
4. First run should be INITIAL to rebuild with new paths

## classify_races_mode logic

In the workflow:
```bash
CLASSIFY_MODE=$(python -c "
import yaml
with open('$ATHLETE_CONFIG_PATH') as f:
    cfg = yaml.safe_load(f)
mode = cfg.get('pipeline', {}).get('classify_races_mode', 'auto')
print(mode)
")
if [ "$CLASSIFY_MODE" != "skip" ]; then
    python classify_races.py ...
fi
```
