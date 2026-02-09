# Multi-Athlete Pipeline: Planning Document
## From v51 (Paul-only) to portable (anyone with FIT files + HR)

*Working document — not a spec, not a task list. For thinking through.*

---

## The Core Decision

The pipeline already has two modes internally: real Stryd power and simulated power. The portable version formalises this as a user-facing choice:

- **Stryd mode**: Real power when available, simulated for gaps/history. Your current pipeline.
- **GAP mode**: Grade Adjusted Pace / HR for everything. No power meter needed.

Both feed into the same downstream engine. The runner sees the same dashboard, same fitness curve, same race predictions. The RF values aren't directly comparable between modes (different units/scale), but RFL (0-100% of personal peak) is universal.

---

## What Needs to Move Where

### Currently Paul-specific → becomes `athlete.yml`

| Item | Current location | Notes |
|---|---|---|
| Mass, DOB, gender | `config.py` constants | Straightforward |
| PEAK_CP_WATTS (372) | `config.py` | Auto-bootstrap for new athletes — see section below |
| Era transition dates | `rebuild_from_fit_zip.py` hardcoded dict | Only relevant for Stryd mode |
| Stryd mass corrections | `config.py` STRYD_MASS_HISTORY | Only relevant for Stryd mode, most athletes won't need |
| RE model reference era | `build_re_model_s4.py` hardcoded "s4" | Only relevant for Stryd mode |
| Simulated power eras | `StepA` / `rebuild` hardcoded sets | Only relevant for Stryd mode |
| Data source credentials | Environment variables | Already externalised, just needs per-athlete .env |

### Currently in code but actually generic (no change needed)

| Item | Why it's already portable |
|---|---|
| Age grading (WMA tables) | Parameterised by age/gender/distance |
| Temperature adjustment | Based on exercise physiology, not athlete-specific |
| Terrain adjustment | Based on undulation physics |
| CTL/ATL/TSB (42/7 day) | Industry standard Banister model |
| Factor weighting | Based on distance × HR intensity |
| RF_Trend quadratic decay | Physiologically motivated, works for any training frequency |
| Alert thresholds | Already in config.py, sensible defaults |
| Dashboard generation | Mostly generic, just needs athlete name |

### Tuning parameters that COULD be athlete-specific but have good defaults

| Parameter | Default | When to change |
|---|---|---|
| RF window (40 min) | Works for 30-90 min runs | Very short or very long typical runs |
| RF trend window (42 days) | Works for 4-5 runs/week | Very high or low frequency |
| Temp baseline (10°C) | Northern Europe | Tropical athlete might use 20°C |
| RE constant (0.92) | Average male runner | Only matters in GAP mode; women ~0.94 |
| Easy RF thresholds | Based on your HR zones | Different max HR athletes |

---

## The Bootstrapping Problem

A new athlete has no history. Several things need bootstrapping:

### PEAK_CP (critical for race predictions)

**Without Stryd**: Can't measure directly. Options:
1. **From a race result** (best): "I ran 5K in 22:00" → back-calculate the CP that produces that prediction at RFL=1.0. One race is enough.
2. **From training data** (after ~3 months): Take peak RF_Trend and multiply by a typical race HR. Rough but functional.
3. **Leave blank initially**: Dashboard works without predictions. Add when first race data comes in.

**With Stryd**: Take highest 20-min rolling power from the data. Available within a few weeks.

**Recommendation**: Make race predictions optional until PEAK_CP is established. The fitness tracking (RF_Trend, RFL, CTL/ATL/TSB) works perfectly without it.

### RE Model (Stryd mode only)

Needs ~50+ runs with real power to train. For a new Stryd user:
- Use the generic Minetti model for the first 50 runs (effectively GAP mode internally)
- Auto-switch to the athlete-specific RE model once enough data exists
- The transition should be invisible to the athlete — RF values shift slightly but RFL recalibrates

### Era Detection (Stryd mode only)

For a new athlete with ONE Stryd:
- There's only one era. No era adjusters needed at all.
- If they later upgrade (e.g. Stryd 4 → Stryd 5), auto-detect from serial number change in FIT files
- Only offer manual era config for athletes migrating from other power meters

### RF_Trend History

First ~5 runs have no trend. This is fine — the pipeline already handles `min_periods=5`. The dashboard just shows individual RF dots until the trend line appears.

---

## Data Sources: What's Realistic

### Tier 1: FIT files in a folder (universal)

Every GPS watch exports FIT files. The athlete:
1. Exports their history (Garmin Connect bulk export, Strava bulk export, or just a folder of files)
2. Points the pipeline at the folder
3. Pipeline parses everything, builds the master

**This is the minimum viable intake.** Everything else is convenience.

### Tier 2: intervals.icu (your current setup)

- FIT file download via API ✅
- Weight/wellness data ✅
- Activity metadata ✅
- Requires: free account + API key
- Good for: ongoing sync (new runs auto-ingested)

### Tier 3: Strava API

- **Cannot** download original FIT files (Strava strips them)
- **Can** fetch per-second streams (HR, speed, altitude, distance, cadence, power)
- **Can** list activities with summary stats
- Requires: OAuth app registration, token refresh flow
- Rate limited: 100 requests per 15 minutes, 1000 per day

**The Strava problem**: Strava's API gives you streams, not FIT files. The pipeline currently expects FIT files as input. Two options:

**Option A — Strava streams as a FIT-equivalent input**: Write a Strava adapter that fetches streams and produces the same intermediate format that `rebuild_from_fit_zip.py` creates from FITs. The pipeline never knows the data came from Strava. This is cleaner architecturally but means re-implementing some of the FIT parsing logic for stream data.

**Option B — Strava bulk export as initial load + API for ongoing**: Athlete does a one-time bulk export (gives real FIT/GPX/TCX files for all history), then uses the API for incremental new activities (streams only). This is pragmatic but has a seam.

**Option C — Just support Strava bulk export**: The athlete downloads their data archive from Strava every few months. No API integration. Low-tech but works.

**Recommendation**: Start with Tier 1 (FIT folder) and Tier 2 (intervals.icu). Add Strava as Option C (bulk export support) since those files are just FIT/GPX/TCX in a zip. The Strava API (Option A) is a significant engineering effort for the OAuth flow and stream-to-intermediate conversion — defer it.

### Tier 4: Garmin Connect

- Bulk export gives FIT files (same as Tier 1)
- API requires developer registration and is poorly documented
- Not worth building a client; just use the exported files

### Weight Data Without intervals.icu

If the athlete doesn't use intervals.icu:
- **Manual entry**: A simple CSV with date,weight_kg columns
- **From Garmin Connect export**: Weight is in the wellness CSV
- **From Apple Health export**: Weight in the XML
- **Or just skip it**: Use a constant weight from `athlete.yml`. Weight tracking is nice-to-have, not core.

---

## The Onboarding Flow

What does a new athlete actually do?

### Minimal setup (GAP mode, FIT folder):

```
1. Copy athletes/template/ to athletes/yourname/
2. Edit athlete.yml: set name, mass_kg
3. Drop FIT files into athletes/yourname/fits/
4. Run: python pipeline.py yourname
5. Open athletes/yourname/dashboard/index.html
```

That's it. Five steps. No API keys, no accounts, no power meter.

### With intervals.icu:

```
1-2. Same as above
3. Set data_source type to "intervals" in athlete.yml
4. Set INTERVALS_API_KEY and INTERVALS_ATHLETE_ID in .env
5. Run: python pipeline.py yourname --sync
```

### With Stryd:

```
1-2. Same as above, but set power_mode: "stryd" in athlete.yml
3-5. Same as either FIT folder or intervals route
```

The pipeline auto-detects Stryd data in the FIT files and handles everything.

---

## Migration Path for Your Pipeline

The current v51 pipeline doesn't need to be rewritten. It needs to be:

1. **Reorganised**: Move Paul-specific constants into `athletes/paul/athlete.yml`
2. **Parameterised**: `config.py` reads from the athlete's YAML instead of hardcoded values
3. **Extended**: Add the GAP/HR calculation alongside the existing power/HR path
4. **Wrapped**: `pipeline.py` accepts an athlete name and resolves all paths

The actual calculation code — `StepB_PostProcess.py`, `sim_power_pipeline.py`, `age_grade.py`, `generate_dashboard.py` — stays essentially the same. The changes are at the edges (config loading, path resolution, RF source selection), not in the core logic.

### What I'd do first (if/when you start)

**Phase 1: Extract athlete config** (~2 sessions)
- Create `athletes/paul/athlete.yml` with all current hardcoded values
- Modify `config.py` to load from YAML, falling back to current constants
- Everything still works identically for you, but the values now come from a file

**Phase 2: Add GAP/HR mode** (~2 sessions)
- Add the Minetti GAP calculation to `rebuild_from_fit_zip.py` (it already computes grade)
- Add a `power_mode` switch that selects real/sim power vs GAP/HR
- Test on your data: run the whole pipeline in GAP mode, compare to real output
- We already know the expected result: 0.90 correlation, 2.1% trend MAE

**Phase 3: Second athlete** (~2 sessions)
- Duplicate your setup for a test athlete (could be you in GAP mode, or someone real)
- Verify the folder structure works with two athletes
- Fix any remaining hardcoded paths

**Phase 4: Strava bulk export support** (~1 session)
- Parse Strava's export zip (FIT/GPX/TCX files + activities.csv)
- Feed into existing pipeline via the FIT folder path
- GPX/TCX parsing is the only new code needed

**Phase 5: Onboarding polish** (~1-2 sessions)
- `onboard.py` interactive setup script
- Template athlete folder
- README with clear instructions
- Handle edge cases: no elevation data, no HR, very sparse data

---

## Open Questions

### 1. Single repo or separate repos?
The core pipeline code is shared. Each athlete has their own data/config/output. Options:
- **Single repo, gitignored data**: Simplest. Athletes fork it and add their folder. Data stays local.
- **Core as a package, athlete repos import it**: Cleaner separation but more infrastructure.
- **Start with single repo**, split later if needed.

### 2. GitHub Actions per athlete?
Your CI/CD runs daily on GitHub Actions. For a second athlete:
- Same repo, separate workflow files? (e.g. `.github/workflows/paul.yml`, `.github/workflows/athlete2.yml`)
- Or one workflow that loops over athletes?
- Or: only you get CI/CD, everyone else runs locally? (Simpler to start.)

### 3. Dashboard hosting?
Currently GitHub Pages from your repo. For multiple athletes:
- Each athlete's dashboard is a separate folder in Pages
- Or: each athlete hosts their own (just an HTML file, can go anywhere)
- Or: a shared site with athlete selector (over-engineering for now)

### 4. RE model sharing?
Your RE model is trained on your Stryd data. For GAP mode athletes it's irrelevant. For another Stryd athlete:
- They train their own model from their own data
- Your model could be a "starter" model until they have enough data
- The generic Minetti model is good enough as a bootstrap

### 5. How to handle GPX/TCX files?
Strava bulk export and some watches produce GPX or TCX instead of FIT. These contain the same data (GPS, HR, elevation, timestamps) but in XML format. Need a parser that produces the same intermediate format. Libraries exist (`gpxpy`, `tcxparser`). ~1 session of work.

### 6. What about athletes with no elevation data?
Some older watches or indoor treadmill runs have no GPS altitude. For GAP mode this means no grade correction — falls back to Approach A (speed/HR). For Stryd mode, power still works fine without elevation. Should handle gracefully, not crash.

---

## Summary: What This Gets Us

A runner with a Garmin watch, no power meter, and a folder of FIT files can:
1. Set their weight in a YAML file
2. Run one command
3. Get a fitness dashboard with RF trends, training load, and (once they race) predictions

The same pipeline, same dashboard, same metrics as the 51-version power-based system — just using GAP/HR instead of power/HR, at 90% of the accuracy, with 0% of the hardware requirements and about 30% of the code complexity.

And if they later buy a Stryd, they flip one switch in their config and the pipeline seamlessly starts using real power, filling in the historical gap with simulation.
