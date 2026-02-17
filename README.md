# Running Performance Pipeline — v51

Three-mode fitness analysis pipeline processing GPS watch + Stryd power meter data.

## Pipeline Flow

```
FIT files → rebuild_from_fit_zip.py → StepA_SimulatePower.py → StepB_PostProcess.py → generate_dashboard.py
                                           ↑                         ↑
                                    build_re_model_s4.py        config.py ← athlete.yml
```

## Core Scripts

| Script | Purpose |
|--------|---------|
| `rebuild_from_fit_zip.py` | Parse FIT files → master Excel + NPZ per-second cache |
| `StepA_SimulatePower.py` | Simulate power for non-Stryd eras using RE model |
| `StepB_PostProcess.py` | RF, RFL, predictions, alerts, training load — the main engine |
| `generate_dashboard.py` | Single-file HTML dashboard with Chart.js |
| `config.py` | Central config, loads from `athlete.yml` via `athlete_config.py` |
| `run_pipeline.py` | Orchestrator for full/incremental runs |

## Data Management

| Script | Purpose |
|--------|---------|
| `fetch_fit_files.py` | Download new FIT files from intervals.icu |
| `add_run.py` | Add a run with metadata (name, shoes, flags) |
| `add_override.py` | Add activity overrides (race flag, surface, distance) |
| `sync_athlete_data.py` | Sync weight + non-running TSS from intervals.icu |

## CI/CD

- `.github/workflows/pipeline.yml` — Daily automated run
- `ci/dropbox_sync.py` — Upload/download NPZ cache + master to Dropbox
- `ci/apply_run_metadata.py` — Apply pending activity names and overrides

## Three Modes

- **⚡ Stryd**: Real power from Stryd pod. Primary model.
- **🏃 GAP**: Grade Adjusted Pace / HR. No power meter needed. 0.904 correlation with Stryd.
- **🔬 SIM**: Athlete-specific simulated power from RE model. 0.993 correlation with Stryd.

## Key Files

- `athlete.yml` — Athlete-specific config (mass, DOB, LTHR, planned races, etc.)
- `activity_overrides.yml` — Per-activity flags (race, surface, distance corrections)
- `requirements.txt` — Python dependencies (pandas <3.0)
