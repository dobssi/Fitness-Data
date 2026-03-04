# Handover: Race Readiness Long Run Metrics + CI Fixes
## Date: 2026-03-04

---

## What happened this session

### 1. Race Readiness — long run specificity metrics (generate_dashboard.py)

Added two new metrics to HM+ race readiness cards (dist_km ≥ 15):

**"Run time ≥ N min"** — a section separator label showing the rounded threshold (80% of predicted finish time, nearest 10 min). E.g. for Ian's HM (~100min predicted) → "Run time ≥ 80 min".

**Long run Nd** — total minutes spent running **beyond** the threshold in qualifying runs. E.g. a 95min run with 80min threshold contributes 15min, not 95min.

**Z3+ in long Nd** — minutes at Z3+ effort (HR ≥ LTHR×0.90) within the tail portion only (minutes 80–95 in the example above).

Both metrics shown for **14d and 42d** windows, matching the race effort spec windows.

**Implementation details:**
- Threshold computed from predicted finish time (same logic as race card), rounded to nearest 10min
- `_lr_data[race_idx]` dict: `threshold_label`, `total_14`, `total_42`, `z3_14`, `z3_42`
- Z3+ tail computation:
  - **With NPZ path**: slices `hr_bpm` array from `threshold_s` onwards, counts seconds ≥ z3_hr_floor directly — fully accurate
  - **With hz but no path**: proportional estimate — `Z3_full × (tail_min / total_min)`
  - **No zone data**: avg HR proxy, same proportional approach
- `_npz_path` stored on `run_entry` when NPZ loads successfully (to enable tail-slice)
- `.ws-tip` CSS class added (separate from `.ws` bar chart class) for hover tooltips

**Tooltips** added to all 6 race card metric cells using `.ws-tip` hover pattern:
- Target, Predicted, 14d/42d at effort (existing metrics)
- Long run 14d/42d: "Time spent running beyond N mins in last X days"
- Z3+ in long 14d/42d: "Time spent running in Z3 or harder beyond N mins in last X days"

**All windows now 14d / 42d** throughout race cards (effort mins, long run mins, Z3+ mins). JS `calcSpecificity` window also updated to 42d.

---

### 2. dropbox_sync.py — upload progress counter

`sync_cache_incremental_upload` previously printed one `✓ Uploaded: ...` line per file — produces hundreds of log lines on INITIAL. Changed to:
- `silent=True` on `dropbox_upload` calls within the cache loop
- Progress line every 50 files: `↑ 50/847 files uploaded...`
- Final summary: `✓ Cache upload complete: 847 uploaded, 0 already in sync`
- Failures reported in summary: `, 3 failed` if any

Both upload functions updated (lines ~400 and ~960 in the file).

---

### 3. paul_pipeline.yml — two-job structure for INITIAL

**Problem:** INITIAL rebuild takes ~5h. GitHub Actions hard limit is 6h per job. The previous run timed out after fits.zip upload, before the final results upload and GitHub Pages deploy.

**Solution:** Split workflow into two jobs, each with their own 6h clock:

**Job 1: `rebuild`** (`timeout-minutes: 350`)
- Handles UPDATE / FULL / INITIAL / CLASSIFY_RACES + scheduled check
- For INITIAL: FIT ingest + full rebuild + weather fetch + persec NPZ upload
- StepB + dashboard + deploy run here for UPDATE / FULL
- Safety upload (`if: always()`) saves weather cache + Master even on timeout
- FROM_STEPB and DASHBOARD skip this job entirely

**Job 2: `stepb_deploy`** (`timeout-minutes: 30`)
- `needs: [rebuild]`
- Auto-triggered after INITIAL rebuild succeeds
- Also triggered directly by FROM_STEPB and DASHBOARD modes
- Downloads from Dropbox, runs StepB + dashboard + deploy
- Condition: `always() && (rebuild succeeded + mode==INITIAL) || mode==FROM_STEPB || mode==DASHBOARD`

**Mode routing summary:**

| Mode | rebuild | stepb_deploy |
|---|---|---|
| UPDATE | ✅ full pipeline | ⏭ skipped |
| FULL | ✅ full pipeline | ⏭ skipped |
| INITIAL | ✅ rebuild only | ✅ auto-chains |
| FROM_STEPB | ⏭ skipped | ✅ runs directly |
| DASHBOARD | ⏭ skipped | ✅ runs directly |
| CLASSIFY_RACES | ✅ classify only | ⏭ skipped |

---

### 4. Paul Test (A005) INITIAL run status

- **First INITIAL attempt**: hit 6h GitHub limit mid-upload (before deploy)
- **Scheduled UPDATE** ran after and deployed correctly — dashboard is live
- **A005 current state**: fully operational, GAP mode, 3,926 runs, CTL=71.0, RFL=69.9%
- **Plan**: delete A005 Dropbox data tonight, run INITIAL overnight using new two-job workflow

---

## Files changed this session

| File | Change |
|---|---|
| `generate_dashboard.py` | Long run tail metrics, tooltips, 42d windows |
| `ci/dropbox_sync.py` | Silent upload + progress counter for cache uploads |
| `.github/workflows/paul_pipeline.yml` | Two-job structure, safety upload, timeout fix |

---

## New TODOs captured

- **Race History section** — two stacked card slots, each with distance-filtered search/select. Cards match Race Readiness design. Side-by-side comparison. Own session.
- **Scheduled workout planner** — planned_sessions in athlete.yml, TSS estimation, CTL/ATL projection, dashed line on chart. Own session.
- **Athlete page** (athlete.html) — forward-looking complement to dashboard. Phase 2 after workout planner.
- **Surface-specific trail specificity** for race readiness long run metrics (future, low priority for now)

---

## For next Claude

"This session added long run tail metrics to race readiness cards (time beyond 80% threshold + Z3+ in that tail, 14d/42d), fixed the INITIAL 6h timeout by splitting into two sequential jobs, and added a Dropbox upload progress counter. Paul Test (A005) dashboard is live and clean. Plan is to delete A005 data and run INITIAL overnight using the new two-job paul_pipeline.yml. Check the dashboard output carefully: race classification, long run metrics on Ian's Paddock Wood Half card, and the stepb_deploy job auto-chaining after rebuild completes."
