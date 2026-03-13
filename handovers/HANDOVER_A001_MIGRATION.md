# Handover: A001 Migration to Two-Job Workflow
## Date: 2026-03-11

---

## Summary

Migration of Paul (A001) from legacy single-job `pipeline.yml` (files in repo root) to standard two-job template in `athletes/A001/`, matching A005/A006/A007 pattern.

## Files produced

| File | Purpose |
|------|---------|
| `.github/workflows/paul_collyer_pipeline.yml` | New two-job workflow for A001 |
| `athletes/A001/athlete.yml` | A001 config with `classify_races_mode: "skip"` |
| `HANDOVER_A001_MIGRATION.md` | This document |
| `MIGRATION_PLAN.md` | Detailed plan with Dropbox paths |

## Key features of the new workflow

1. **Two-job template** — rebuild (350 min) + stepb_deploy (30 min), each with own 6h clock
2. **classify_races_mode: "skip"** — new field in athlete.yml. When "skip", both the rebuild-job and stepb_deploy-job classify_races steps are skipped entirely. Your curated `activity_overrides.xlsx` is the sole source of race classifications.
3. **NPZ incremental upload** — `--cache-dir` added to the UPDATE/FULL Dropbox upload step. Fixes the missing-NPZ-on-Dropbox issue we identified this session.
4. **Dispatch inputs** — run_name, race, parkrun, distance_km, surface, surface_adj, notes, run_date, run_number all preserved from original pipeline.yml.
5. **TotalHistory.zip** — FULL/INITIAL rebuilds use `data/TotalHistory.zip` (your curated archive), UPDATE mode uses `data/fits.zip` (incremental intervals.icu downloads).
6. **GH Pages path** — `docs/paul_collyer/index.html`

## Athlete folder renames (also needed)

| Old | New | Workflow to update |
|-----|-----|--------------------|
| `athletes/IanLilley` | `athletes/A002` | `ian_lilley_pipeline.yml` |
| `athletes/NadiJahangiri` | `athletes/A003` | `nadi_jahangiri_pipeline.yml` |
| `athletes/SteveDavies` | `athletes/A004` | `steve_davies_pipeline.yml` |

Each workflow needs: `ATHLETE_DIR`, `DB_BASE`, cache keys, and GH Pages path updated to use the numeric ID.

## Dropbox migration steps (Paul to do manually)

1. In Dropbox, create folder: `Running and Cycling/DataPipeline/athletes/A001/`
2. Create subfolders: `data/`, `output/`, `persec_cache/`, `output/dashboard/`, `output/_weather_cache_openmeteo/`
3. Move/copy files from root to new paths:
   - `Master_FULL_GPSQ_ID.xlsx` → `output/Master_A001_FULL.xlsx`
   - `Master_FULL_GPSQ_ID_post.xlsx` → `output/Master_A001_FULL_post.xlsx`
   - `activity_overrides.xlsx` → `activity_overrides.xlsx`
   - `athlete_data.csv` → `athlete_data.csv`
   - `TotalHistory.zip` → `data/TotalHistory.zip`
   - `activities.csv` → `data/activities.csv`
   - `persec_cache_FULL/` contents → `persec_cache/`
   - `re_model_s4_FULL.json` → `re_model_s4_FULL.json`
   - `sync_state.json` → `sync_state.json`
   - `fit_sync_state.json` → `fit_sync_state.json`
   - `pending_activities.csv` → `pending_activities.csv`
   - `_weather_cache_openmeteo/` → `output/_weather_cache_openmeteo/`
4. First run: INITIAL mode to rebuild with new paths

## What's NOT in this migration (separate sessions)

- **detect_eras.py integration** — needs testing on A001 data first
- **Info sheet in Master** — adding athlete.yml summary as first sheet in XLSX
- **Ian/Nadi/Steve folder renames** — straightforward but touches their workflows + Dropbox
- **Retiring pipeline.yml** — keep as backup until A001 workflow is validated

## Known issues from this session

1. **Stryd firmware RE shift (~1-1.2%)** — detected but not corrected. GAP mode confirms most of the RFL dip (0.895→0.874) is genuine fatigue, firmware accounts for ~1.2%.
2. **F9 session RF (1.699)** — artificially low due to 11-min pause creating HR spike in first interval rep. Per-second interval-only RF is 1.707 (still lower than F7 1.800 / F8 1.785 due to genuine HR elevation of 7 bpm).
3. **HR spike after mid-session pauses** — pipeline only handles spikes at session start. TODO: extend filter to detect gaps > 60s in timestamp array and apply same spike cleanup after each resume.

---

## For next Claude

"A001 migration files are ready. The workflow (`paul_collyer_pipeline.yml`) and athlete.yml are produced but NOT deployed — Paul needs to set up the Dropbox folder structure first (see HANDOVER_A001_MIGRATION.md for the file mapping). The key new feature is `classify_races_mode: skip` in athlete.yml which prevents classify_races.py from overwriting Paul's curated overrides. The NPZ upload fix is also included. See MIGRATION_PLAN.md for the full architecture."
