# Handover: Onboarding + Dashboard + Documentation Session
## Date: 2026-03-06

---

## Summary

Onboarding improvements, dashboard stat card tooltips, race HR threshold defaults, project documentation, and Stryd onboarding logic fix. No pipeline logic changes. PaulTest INITIAL rebuild running in background (~6h).

---

## Files modified this session

### `onboard.html` (838 → 1118 lines)
- **HR zone preview panel** — appears in Heart Rate section when LTHR is entered. Auto-calculates 5 zones using same formula as `generate_dashboard.py` (0.81, 0.90, 0.955, 1.0 × LTHR). Editable boundaries with inline inputs, accent highlight on modified values, reset button. Exports `zones.hr_zones` array to YAML/JSON config only when customised (otherwise pipeline uses formula fallback).
- **Strava/data upload guidance** — export instructions rewritten to explain any file combination works (whole zip, individual FITs, activities.csv standalone, mix). Pipeline auto-detects — no checkboxes or declarations needed.
- **Weight CSV format guide** — collapsible section with format example. Accepts flexible date formats. Upload via same Dropbox link.
- **Return visit panel** — returning users see "Upload new data" box with Dropbox link and reminder they can upload updated exports, activities.csv, or weight CSVs at any time.

### `generate_dashboard.py` (6686 → 6704 lines)
- **Stat card tooltips** — all 16 cards in the stats grid now have hover tooltips using existing `.ws-tip` pattern. Covers CTL/ATL/TSB/Weight, RFL/RFL14d/EasyRF/AG, predictions, and volume. TSB tooltip references actual target range (5–25% of CTL). Stat cards get `cursor: help` and subtle accent border on hover.

### `onboard_athlete.py` (1458 → 1484 lines)
- **Race HR thresholds** — commented-out `race_hr_thresholds_pct` block added to generated `athlete.yml` (3K:0.98 through Marathon:0.88). Documents the defaults for athletes who want to customise.
- **Stryd onboarding logic fix** — replaced the blunt "no power data → downgrade to GAP" with threshold-based logic:
  - 50+ power runs + user declared Stryd → Stryd mode
  - 10–49 power runs + declared → Stryd mode (with note)
  - 1–9 power runs + declared → GAP mode, YAML comment to switch at 10+ runs
  - 0 power runs + declared → GAP mode, YAML comment to switch when data appears
  - No declaration → existing auto-detect (50% threshold + Garmin CV check)

### `athlete_template.yml` (79 → 94 lines)
- **Race HR thresholds** — same commented-out block added for manual setup path.

### New documentation files
- **`README.md`** — full project overview: what it does, three pipeline stages, three fitness modes (Stryd/GAP/SIM), current athletes, project structure, CI/CD modes, onboarding process, key design decisions. Replaces old setup-only README.
- **`ATHLETE_GUIDE_STRYD.md`** — dashboard reading guide for Stryd athletes. Covers all sections with power-specific language (power/HR ratio, RE p90, power zones, target wattage).
- **`ATHLETE_GUIDE_GAP.md`** — same guide for GAP athletes. Uses pace-specific language (GAP/HR ratio, pace zones, target pace, bootstrapped predictions).
- **`TODO.md`** — updated with session completions and new items.

### `HANDOVER_PAULTEST_INITIAL_FIX.md`
- Corrected timing (was "~4h", now "~6h") and persec cache upload info (stepb_deploy uploads unconditionally, no follow-up UPDATE needed).

---

## PaulTest INITIAL rebuild

Running in background at time of handover. Key fix: first fetch skipped for INITIAL mode (one-line change in `paul_pipeline.yml` from prior session). Expected to include all runs through current date. ~6h total.

---

## Key findings this session

- **RE_p90 = 0.9246** for Paul's Stryd predictions (90th percentile of last 50 runs). LFOTM 5K on 27 Feb had RE_avg = 0.9240 — nearly identical, confirming p90 correctly reflects race-day efficiency.
- **Banister model uses exponential decay** (`alpha = 1 - exp(-1/tau)`), not the simple `1/tau` approximation. Memory/docs updated accordingly.
- **TSB race targets are % of CTL**, not absolute values. A-race 5K: 5–25% of CTL.
- **`classify_races.py` already had correct default HR thresholds** — the TODO was only about adding them to templates.

---

## TODO items added

- **Drop SIM mode from new athlete dashboards** — SIM mode is a dev validation tool (proves GAP ≡ simulated power). Confusing for athletes. Keep in pipeline for dev, hide from athlete-facing: generated `athlete.yml` shouldn't mention SIM, workflow templates skip SIM flags, dashboard mode toggle shows only Stryd ↔ GAP. Own session.

---

## Files to deploy

All files are in `/mnt/user-data/outputs/`. To deploy:

1. `onboard.html` → replace in repo root
2. `generate_dashboard.py` → replace in repo root
3. `onboard_athlete.py` → replace in repo root
4. `athlete_template.yml` → replace in repo root
5. `README.md` → replace in repo root
6. `ATHLETE_GUIDE_STRYD.md` → add to repo root
7. `ATHLETE_GUIDE_GAP.md` → add to repo root
8. `TODO.md` → replace in repo root
9. `HANDOVER_PAULTEST_INITIAL_FIX.md` → replace in repo root

None of these affect the running pipeline — all are documentation, onboarding UI, or template changes. Safe to deploy while INITIAL runs.

---

## For next session

- Check PaulTest INITIAL results — should now include runs through current date
- Drop SIM mode from athlete-facing dashboard/templates (own session)
- Sub-3km race classification bug (`classify_races.py` missing track miles, 1500m etc)
- Race History effort/tail move to StepB (performance + mode toggle)
- Activity search + override editor (own session)
