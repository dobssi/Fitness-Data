# Handover: Taper Solver Fix + Daily Sheet Weight + Onboarding Form
## Date: 2026-02-26

---

## Summary

Fixed a double-counting bug in the taper solver's CTL/ATL projection, added weight and non-running TSS columns to the Daily sheet output, and built a self-service athlete onboarding web form.

---

## Fixes Applied

### 1. Taper solver double-counting bug (generate_dashboard.py)

**Problem**: Training Load chart showed CTL 58.4 / ATL 49.5 for race morning (27 Feb), while Race Readiness card showed CTL 58.0 / ATL 47.7 — ATL gap of 1.8 points.

**Root cause**: `_solve_taper()` started from `df.iloc[-1]` CTL/ATL — the end-of-day values that already include today's TSS (43 from the calibration tuneup). But the forward projection loop (line 1954, `for i_d in range(0, ...)`) re-applied today's TSS on day i_d=0, effectively counting it twice. This pulled the projection toward today's TSS value, creating divergence from the Daily sheet projection which correctly applied each day's TSS only once.

**Mathematical proof**:
- Daily sheet: today CTL=59.8 → tomorrow = 59.8 × 41/42 = 58.4 ✓
- Solver (double-counted): 59.8 + (43-59.8)/42 = 59.4 → 59.4 × 41/42 = 58.0 ✓ (matches the wrong race card value)
- ATL gap larger (1.8 vs 0.4) due to shorter time constant (τ=7 vs τ=42)

**Fix** (lines 1734-1769): Changed `current_ctl`/`current_atl` to use **yesterday's end-of-day** values from the Daily sheet instead of `df.iloc[-1]`. The solver now starts from yesterday's state, applies today's TSS from `recent_runs` exactly once in the forward loop, then projects forward with planned/template TSS values. Falls back to `df.iloc[-1]` if the Daily sheet is unavailable (new athletes without history).

**Additional fix**: `_td` NameError — changed to `timedelta` (which is imported at the top of the file). `_td` was an artifact from a previous refactor.

### 2. Daily sheet: Weight + non-running TSS columns (StepB_PostProcess.py)

**Problem**: Excel analysis required the external `athlete_data.csv` file open for SUMIF formulas pulling non-running TSS and weight — caused "unable to refresh" errors when the CSV wasn't open.

**Fix** (lines 1645-1697, 1735-1746):

**Weight loading**:
- Added `weight_by_date` dict built from `athlete_data.csv` alongside existing `non_running_tss` loading
- Reads `weight_kg` column, groups by date (last value per day)
- Maps to `daily_df['weight_kg']`, forward-fills for continuity
- Days before first measurement stay NaN (correct behaviour)

**Daily sheet output columns** (in order):
- `Date`, `distance_km`, `TSS_Running`, `TSS_Other`, `TSS_Total`, `CTL`, `ATL`, `TSB`, `RF_Trend`, `RFL_Trend`, `Weight_kg` (+ optional `RFL_gap_Trend`, `RFL_sim_Trend`)

**Result**: Excel VLOOKUP formulas can now pull weight and non-running TSS directly from the Daily sheet with no external file dependency.

---

## New: Self-Service Athlete Onboarding Form

### onboard.html

Single-page dark-themed form matching dashboard aesthetic (DM Sans font, #0f1117 background, #1a1d27 cards). Deployed to GitHub Pages via `Push_Onboard.bat`.

**Section 1 — About You**: Name, email, DOB, gender, weight (kg), timezone (9 options covering UK, Europe, US, Australia, Japan)

**Section 2 — Heart Rate**: Max HR and LTHR with "Estimate from my data" checkboxes. Hint text: LTHR ≈ avg HR during hard 10K.

**Section 3 — Your Running Data** (non-exclusive design):
- **FIT File Upload**: Always visible. Accepts Garmin/Strava export zip. Expandable instructions for both platforms. Dropbox file request link for >500MB files.
- **intervals.icu Integration**: Checkbox reveals athlete ID + API key fields. Not mutually exclusive with FIT upload — can provide both. Green checkmarks highlight benefits (weight sync, daily updates, no manual exports).
- **Optional extras**: Strava `activities.csv` upload (helps detect races + name runs), Weight CSV upload (`date,weight_kg` format).

**Section 4 — Races** (optional): Recent race results (date, distance, time) and upcoming races (date, distance, name, priority A/B/C). Dynamic add/remove rows.

**Submission**: Opens mailto with formatted summary including a ready-to-paste `athlete.yml` snippet. Includes data source config. Stores in localStorage as backup. Shows success message.

**Design**: 4 step dots showing progress. Responsive mobile layout. Validation: requires name, email, weight, DOB; gentle confirm if no data source provided.

### Push_Onboard.bat

Batch file to deploy `onboard.html` to the `gh-pages` branch of the `Fitness-Data` repo. Copies from the DataPipeline root (same folder as the bat file), commits, pushes, switches back to `main`.

**Note**: Both `Push_Onboard.bat` and `onboard.html` must be in the same directory (the DataPipeline root, one level above `Fitness-Data/`). If the file content is identical to what's already on gh-pages, git reports "nothing to commit" — this is correct behaviour, not an error.

---

## Placeholders to Fill Before Sharing

The onboard.html has 4 placeholders (2 instances each):
1. `__YOUR_EMAIL__` — mailto recipient address
2. `__DROPBOX_REQUEST_LINK__` — Dropbox file request URL for large uploads

---

## CI Auto-Deploy

To auto-deploy `onboard.html` on pipeline runs, add to `.github/workflows/pipeline.yml` (around line 252, in the gh-pages deploy step):
```yaml
cp -f onboard.html docs/onboard.html 2>/dev/null || true
```

---

## Files Modified

| File | Changes |
|------|---------|
| `generate_dashboard.py` | Taper solver: start from yesterday's Daily sheet CTL/ATL (lines 1734-1769); fix `_td` → `timedelta` |
| `StepB_PostProcess.py` | Load weight from athlete_data.csv, add `Weight_kg` column to Daily sheet output (lines 1645-1697, 1735-1746) |

## Files Created

| File | Purpose |
|------|---------|
| `onboard.html` | Self-service athlete signup form |
| `Push_Onboard.bat` | Deploy onboard.html to gh-pages |

---

## Steve's LTHR

Investigated Steve Davies' identical zone boundaries to Ian Lilley — both have LTHR 157. This is likely a placeholder copied during onboarding (file generated by `onboard_athlete.py on 2026-02-24`). Steve's max HR of 173 suggests LTHR should be ~144-152 (83-88% of max). Needs calibration from his race/training data, or Steve can report his watch-detected LTHR.

---

## Scaling Discussion

Discussed product viability: 1000 users × £3/month = £36k/year if fully automated. Next milestones:
- Prove pipeline with 5-10 athletes (Ian, Steve, Nadi + new signups)
- PWA wrapper for home screen install + offline access (intermediate step before native app)
- Self-service backend (FastAPI + Stripe + cron per athlete) when ready for 50+ athletes
- Future: athletes can add intervals.icu to their pipeline via a settings page (currently a 5-minute manual config change)

---

## For Next Claude

"Fixed taper solver double-counting bug — was starting from end-of-today CTL/ATL then re-applying today's TSS in forward loop. Now starts from yesterday's Daily sheet values, applies today once. Added Weight_kg + TSS columns to Daily sheet output. Built self-service onboarding form (onboard.html) with non-exclusive FIT upload + intervals.icu integration, deployed via Push_Onboard.bat to gh-pages. Placeholders __YOUR_EMAIL__ and __DROPBOX_REQUEST_LINK__ need filling before sharing publicly. See HANDOVER_TAPER_FIX_ONBOARDING_20260226.md."
