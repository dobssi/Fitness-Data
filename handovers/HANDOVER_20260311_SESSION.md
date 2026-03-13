# Handover: Data Analysis + A001 Migration + A007 Fixes
## Date: 2026-03-11

---

## Summary

Multi-topic session: Stryd firmware analysis on Paul's data, A001 migration to two-job workflow, and three A007 (Johan) fixes deployed. A001 INITIAL rebuild is currently running.

---

## 1. Stryd Firmware Analysis (Paul / A001)

### Finding
Paul's Stryd received two firmware updates: one around Feb 8 (post-flu), one around Feb 21 (post-Battersea parkrun). The second update, first used around Feb 28 or Mar 1, caused a measurable power calibration shift.

### Evidence
- **RE (running economy)** on Stockholm-only easy runs: 0.889 baseline (Dec–Jan) → 0.909 post-firmware (Mar 3–11). RE = speed×mass/power, so higher RE = Stryd reading **less power** per unit speed.
- **Stryd/GAP ratio** (RF_adj / RF_gap_adj): ~1.00 in Jan → ~0.97–0.98 from late Feb. Confirms Stryd power dropped ~2–2.5% relative to physics-model GAP.
- **But GAP mode RFL also dipped**: 0.895 → 0.874 (−2.3%). This is genuine fatigue visible without any Stryd involvement.
- **Net**: ~2.3% genuine fitness dip + ~1.0–1.2% firmware artefact. The firmware effect is smaller than initially estimated (was thinking 2.4%, actual ~1.2%).

### F9 Session (Mar 10) NPZ Analysis
- 11-minute pause between warmup and intervals caused HR to drop to 101 bpm
- First interval rep: HR spiked from 101→186 in 3.5 min, creating artificially low RF (1.677)
- Pipeline only handles HR spikes at session start, not after mid-session pauses
- Interval-only RF comparison: F7=1.800, F8=1.785, F9=1.707 — genuine dip from elevated HR (+7 bpm for same power), not just measurement noise
- TODO: extend HR spike filter to detect gaps > 60s in timestamp array

### detect_eras.py
A006 did NOT detect the firmware shift — the ~2.4% Stryd/GAP change is below the ~3% noise floor of the binary segmentation algorithm. Both A001 and A006 carry the shift uncompensated.

---

## 2. A001 Migration to Two-Job Workflow

### Status: INITIAL REBUILD RUNNING

### What was done
- Created `paul_collyer_pipeline.yml` from Paul Test (A005) template
- Key adaptations for A001:
  - `ATHLETE_DIR: athletes/A001`, `DB_BASE: .../athletes/A001`
  - Dispatch inputs preserved (run_name, race, parkrun, distance, surface, notes)
  - `apply_run_metadata.py` step added
  - `TotalHistory.zip` used for FULL/INITIAL rebuild (not fits.zip)
  - **`classify_races_mode: "skip"`** — new athlete.yml field, skips classify_races.py to protect curated overrides
  - **`initial_fit_source: "local_only"`** — new athlete.yml field, prevents intervals.icu from re-adding curated-out activities during INITIAL
  - **NPZ incremental upload** added to Dropbox upload step (fixes the missing-NPZ-on-Dropbox issue)
  - GH Pages path: `docs/paul_collyer/`

### Dropbox structure
Paul created `athletes/A001/` in Dropbox and moved files from root to new paths per MIGRATION_PLAN.md. Master files renamed from `Master_FULL_GPSQ_ID*.xlsx` to `Master_A001_FULL*.xlsx`.

### Files delivered
- `paul_collyer_pipeline.yml` → `.github/workflows/`
- `athletes/A001/athlete.yml` (with classify_races_mode, initial_fit_source, rf_trend_min_periods: 10)

### Not yet done
- Retire old `pipeline.yml` (keep as backup until A001 validated)
- Ian/Nadi/Steve folder renames (IanLilley→A002, etc.)
- detect_eras.py integration for A001
- Info sheet in Master from athlete.yml

---

## 3. A007 (Johan) Fixes

### Fix 1: RF_Trend min_periods (DEPLOYED)
**Problem:** RF_gap_Trend peak set from first run (2015-08-15, RF=2.258) based on 1 data point. Distorted RFL to 84.8%.
**Fix:** `valid.any()` → `valid.sum() >= RF_TREND_MIN_PERIODS` in three places (Stryd, GAP, Sim trend blocks). `RF_TREND_MIN_PERIODS` imported from config.py, all athlete.yml files updated from 5 to 10.
**Result:** Peak moved to 2025-10-12 (RF=1.956). RFL now 97.4%. Predictions shifted from 20:00 to 20:10 (5K) due to PEAK_CP recalibration from 384→329.

### Fix 2: Prediction chart mode-awareness (DEPLOYED)
**Problem:** `get_prediction_trend_data()` in generate_dashboard.py hardcoded `pred_5k_s`. GAP athletes only have `pred_5k_s_gap`.
**Fix:** Added `_pred_col()` helper that tries mode-specific column first, falls back through alternatives.
**Result:** Prediction chart now populated with 706 data points. Known remaining issue: "adjust for conditions" dashed line doesn't appear (likely needs same mode-aware treatment for the adjusted prediction columns).

### Fix 3: Missing treadmill runs — INVESTIGATION COMPLETE
**Problem:** ~110km missing from Jan-Feb 2026 vs Strava.
**Investigation:** `scan_polar_sports.py` (standalone script) scanned all 2446 sessions in Johan's Polar export. Sport ID 95 (Treadmill) has only 6 sessions. The missing treadmill runs are NOT in the Polar export at all — they're recorded on the V3 but not included in the Polar Flow data export zip (known Polar limitation).
**Fix:** Johan needs to request a **Strava bulk export**. Pipeline will process both Polar and Strava exports, deduplicate by timestamp. `initial_data_ingest.py` auto-detects both export types in the data folder.

---

## 4. Files Changed This Session

| File | Change |
|------|--------|
| `StepB_PostProcess.py` | Added `RF_TREND_MIN_PERIODS` import + three trend blocks gated on min_periods |
| `generate_dashboard.py` | `_pred_col()` helper for mode-aware prediction columns |
| `athletes/*/athlete.yml` (all 8) | `rf_trend_min_periods: 5` → `10` |
| `athlete_template.yml` | `rf_trend_min_periods: 5` → `10` |
| `.github/workflows/paul_collyer_pipeline.yml` | NEW — A001 two-job workflow |
| `athletes/A001/athlete.yml` | NEW — with classify_races_mode, initial_fit_source |
| `scan_polar_sports.py` | NEW — standalone Polar sport ID scanner |

---

## 5. Outstanding TODOs (from this session)

- **A001 INITIAL completion** — monitor, verify row count matches expectations (~3125), check dashboard deploys to `docs/paul_collyer/`
- **HR spike filter for mid-session pauses** — extend to detect timestamp gaps > 60s and apply spike cleanup after each resume
- **Prediction chart "adjust for conditions"** — condition-adjusted columns likely need same `_pred_col()` treatment
- **Johan Strava export** — waiting on Johan to request from Strava (takes a few hours to prepare)
- **Ian/Nadi/Steve folder renames** — IanLilley→A002, NadiJahangiri→A003, SteveDavies→A004
- **NPZ upload fix for other workflows** — paul_pipeline.yml (A005), paul_stryd_pipeline.yml (A006), and other two-job workflows also missing incremental NPZ upload on UPDATE/FULL
- **detect_eras.py for A001** — separate session
- **Info sheet in Master from athlete.yml** — separate session

---

## For Next Claude

"A001 is mid-INITIAL rebuild on the new two-job workflow (paul_collyer_pipeline.yml). Key new athlete.yml fields: `classify_races_mode: skip` (protects curated overrides) and `initial_fit_source: local_only` (prevents intervals.icu from re-adding excluded activities on INITIAL). StepB and generate_dashboard.py have been patched: RF_Trend now requires 10 valid runs before computing (fixes Johan's inflated 2015 peak), and prediction charts now resolve mode-specific columns for GAP athletes. All athlete.yml files updated to rf_trend_min_periods: 10. See HANDOVER_A001_MIGRATION.md and this handover for full context."
