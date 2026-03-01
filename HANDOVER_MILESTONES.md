# Handover: Milestones Feature + Bug Fixes
## Date: 2026-03-01

---

## What to do next

### 1. Implement Milestones feature (new)

Full design spec in `DESIGN_MILESTONES.md` (attached). Summary:

**Hero banner** above Race Predictions section showing the most recent significant milestone (gold/silver tier, 28-day expiry). **Inline badges** in Recent Runs table for all milestone tiers.

**Four milestone types per run:**
1. **Overall AG** — best age grade across all distances/surfaces ("Best AG in 5y 8m")
2. **Distance+surface AG PB** — "5K AG PB", "trail HM AG PB"
3. **Distance+surface time PB** — "Marathon time PB" (a PB is a PB, clock time matters)
4. **Distance+surface condition-adjusted time PB** — only flagged when it differs from #3 (suppressed when clock time was also a PB). Uses duration-scaled Temp_Adj × surface_adj.

**Three tiers:**
- 🏆 Gold: all-time PB or best in 5+ years → banner + badge
- ⭐ Silver: best in 3+ years → banner + badge
- ✨ Bronze: best in 1+ year → badge only

**Key rules:**
- Applies to any run with valid `age_grade_pct` (races + parkruns)
- Surface categories: road (default/NaN), trail (TRAIL/SNOW/HEAVY_SNOW), track (TRACK), indoor (INDOOR_TRACK)
- 2-attempt rule: distance PBs need ≥1 prior attempt at same distance+surface (2nd marathon can be a PB, 1st cannot)
- Overall AG needs ≥5 prior AG runs
- Banner: 28-day expiry, most recent banner-worthy milestone wins
- Badge: show highest-tier milestone only per row
- "Previously" reference line in banner: date, name, AG%, time of the run that was bettered

**Validated against Paul's data:** ~3 banners/year, ~10 badges/year across 60 AG runs in 2 years. Not spammy.

**Implementation:** ~175 lines, all in `generate_dashboard.py`. No StepB/config changes needed.

---

### 2. Fix: Prediction chart nearby-fallback for mode predictions (ready to deploy)

**Bug:** Race prediction chart dashed line (condition-adjusted) diverges wildly from the solid line, especially for Steve's marathon view and Paul's GAP mode.

**Root cause:** `get_race_prediction_chart_data()` builds `predicted_gap` and `predicted_sim` arrays WITHOUT the 7-day nearby fallback that the main `predicted` array has. For Steve's 7 marathon races, only 3 have `pred_marathon_s_gap` on the race row itself — the other 4 are null. Chart.js Bézier tension:0.3 interpolating through 3 points over 8 years creates wild curves.

**Fix:** Added nearby-fallback to the mode prediction loop (lines ~1255-1276). Fixed file: `generate_dashboard.py` in outputs.

**The fix is already in the `generate_dashboard.py` output file from this session.**

---

### 3. Bug: Race readiness card shows past races + wrong prep window

Two issues with the Race Readiness section:

**a) Race card still shows after the race has been run.** LFOTM on 27 Feb was still showing in the race readiness section on the day (after the race). The card should disappear once the race date has passed, or at least once a race-flagged activity exists on that date.

**b) Prep window is race-7 days, should be race-6 days.** The specificity/preparation calculation looks back 7 days from race day, but the useful training window is the 6 days before the race (race-6 to race-1). Race day itself and 7 days back are both wrong boundaries. Warmup runs on race day (before the race) should count towards preparation.

**c) Warmup runs not included in prep.** Runs on race day that aren't the race itself (e.g. a warmup jog flagged as a separate activity) should count towards the preparation specificity minutes. Currently only runs before race day are included.

These are in `generate_dashboard.py` in the race readiness / planned races section. Look for `calcSpecificity()` in the JS and the planned_races card rendering logic.

---

## Files for next session

Upload to Claude:
1. `checkpoint_v52_*.zip` (latest checkpoint with all scripts)
2. `DESIGN_MILESTONES.md` (this session's output — the feature spec)
3. `generate_dashboard.py` (this session's output — contains the prediction chart fix)
4. This handover document
5. `Master_FULL_GPSQ_ID_post.xlsx` (Paul's master, for testing milestones)
6. Optionally `Master_FULL_post.xlsx` (Steve's master, for testing cross-athlete)

---

## What was done this session

1. **Investigated Steve's dashboard Temp_Adj discrepancy** — London Marathon 2018 tooltip showing +1.3% vs Excel 5.2%. Traced to stale dashboard HTML generated before FULL rebuild completed. New master has correct Temp_Adj=1.052 with solar radiation (671 W/m², +3.4°C effective temp boost). Dashboard just needs regeneration.

2. **Found and fixed prediction chart nearby-fallback bug** — mode predictions (predicted_gap/predicted_sim) missing 7-day fallback, causing sparse data + Bézier interpolation artefacts.

3. **Analysed Paul's LFOTM AG achievement** — 77.87% AG, best in 5 years 8 months (since Jun 2020 BMAF virtual 5K). This sparked the milestones feature idea.

4. **Designed milestones feature** through iterative discussion:
   - Started with AG-only milestones
   - Paul requested both time and AG PBs ("a PB is a PB")
   - Added surface differentiation (road/trail/track/indoor)
   - Added 2-attempt rule (2nd marathon can be a PB)
   - Added condition-adjusted time PBs (suppressed when clock PB also achieved)
   - Settled on 28-day banner expiry

---

## Current state of generate_dashboard.py

The output file contains ONE fix (prediction chart nearby-fallback) but NOT the milestones feature — that's to be built in the next session from the design spec.
