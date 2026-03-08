# Handover: Onboarding Fixes — Two-Job Workflows, Zones, HR Thresholds
## Date: 2026-03-08

---

## Summary

Four onboarding fixes plus new workflows for Ian, Steve, and Nadi — all matching the proven A005 (Paul Test) two-job template.

---

## Changes

### 1. `onboard_athlete.py` — two-job workflow generator (1492 → 1641 lines)

**Complete rewrite of `generate_workflow_yml()`** to produce a two-job structure matching A005:

- **Job 1 (rebuild):** UPDATE/FULL/INITIAL/CLASSIFY_RACES modes. 350-min timeout.
  - Lightweight intervals.icu check (gated on `continue==true`)
  - Full rebuild + add_gap_power + classify_races
  - StepB + dashboard for UPDATE/FULL only (INITIAL defers to job 2)
  - Safety upload with `if: always()` — preserves weather cache + Master on timeout
  - Persec cache upload for INITIAL mode
  - FROM_STEPB/DASHBOARD modes skip this job entirely

- **Job 2 (stepb_deploy):** Auto-chains after INITIAL/CLASSIFY_RACES succeeds, or runs directly for FROM_STEPB/DASHBOARD. Fresh 6h clock.
  - Downloads from Dropbox, runs StepB
  - Re-classifies races (INITIAL/CLASSIFY_RACES) — second pass with predictions available
  - StepB second pass to apply newly-discovered race flags
  - Dashboard generation + GitHub Pages deploy

**CLASSIFY_RACES mode now included** in both intervals.icu and FIT-only trigger options.

**Schedule commented out** for new athletes — uncomment when ready for automated runs.

### 2. `onboard_athlete.py` — HR zones written to athlete.yml

**`generate_athlete_yml()`** now builds a `zones:` block:
- Reads `hr_zones` from config JSON (form sends `{z12, z23, z34, z45}` dict)
- Falls back to auto-calculation from LTHR (same formula as `generate_dashboard.py`: 0.81, 0.90, 0.955, 1.0 × LTHR)
- Also accepts list format `[144, 160, 169, 178]`

### 3. `onboard_athlete.py` — race HR thresholds uncommented

Default thresholds (3K: 0.98 → Marathon: 0.88) are now **active by default** in generated athlete.yml, not commented out. `classify_races.py` already uses these values — having them explicit in the YAML makes it clear what's being used.

### 4. `fetch_fit_files.py` — create fits.zip from scratch

The zip append section (`if downloaded and os.path.exists(args.zip)`) now handles the case where fits.zip doesn't exist yet:
- Existing zip → append as before (mode `'a'`)
- No zip → create new (mode `'w'`, with `os.makedirs` for parent dir)

This fixes the first-run problem for new athletes where intervals.icu fetch downloads FIT files but can't zip them because fits.zip doesn't exist on Dropbox yet.

### 5. `athlete_template.yml` — updated

- Race HR thresholds uncommented (active by default)
- `zones:` block added with example values from LTHR=165
- Comments updated

### 6. New workflow files for Ian, Steve, Nadi

Generated from the updated `generate_workflow_yml()`:

| Athlete | Old file | New file | Source | Intervals? |
|---------|----------|----------|--------|------------|
| Ian | `ian_pipeline.yml` (332 lines, single job) | `ian_lilley_pipeline.yml` (574 lines, two jobs) | intervals.icu | Yes |
| Steve | `steve_pipeline.yml` (332 lines, single job) | `steve_davies_pipeline.yml` (574 lines, two jobs) | intervals.icu | Yes |
| Nadi | `nadi_pipeline.yml` (332 lines, single job) | `nadi_jahangiri_pipeline.yml` (465 lines, two jobs) | FIT folder | No |

All three have: two-job split, classify_races step, safety uploads, CLASSIFY_RACES mode, schedule commented out.

**Slug change:** workflows now use full name slug (`ian_lilley` not `ian`) to avoid collisions — matches what `onboard_athlete.py` generates for new athletes.

---

## Deployment steps

### Files to add/replace in repo

1. `onboard_athlete.py` → replace
2. `fetch_fit_files.py` → replace
3. `athlete_template.yml` → replace
4. `.github/workflows/ian_lilley_pipeline.yml` → add (new)
5. `.github/workflows/steve_davies_pipeline.yml` → add (new)
6. `.github/workflows/nadi_jahangiri_pipeline.yml` → add (new)

### Files to delete from repo

7. `.github/workflows/ian_pipeline.yml` → delete (replaced by ian_lilley_pipeline.yml)
8. `.github/workflows/steve_pipeline.yml` → delete (replaced by steve_davies_pipeline.yml)
9. `.github/workflows/nadi_pipeline.yml` → delete (replaced by nadi_jahangiri_pipeline.yml)

### Secrets — no changes needed

Workflows edited to use existing secret names (`IAN_INTERVALS_*`, `STEVE_INTERVALS_*`). No new secrets to create.

### Pages paths — no changes needed

Workflows deploy to existing paths (`docs/ian/`, `docs/steve/`, `docs/nadi/`). No link updates needed.

---

## Files NOT changed

- `generate_dashboard.py` — no changes
- `StepB_PostProcess.py` — no changes
- `classify_races.py` — no changes
- `config.py` — no changes
- `onboard.html` — no changes
- Athlete YAML files — no changes (existing athletes keep their current configs)

---

## Known issues / next steps

1. **Athlete folder refactor** — still using name-based folders (IanLilley, SteveDavies, NadiJahangiri). The TODO to refactor to A001-A004 numeric IDs remains.
2. **Onboard.html form HR zones** — form already sends `hr_zones` in JSON, and `onboard_athlete.py` now writes them. End-to-end flow works.

---

## For next Claude

"Onboard_athlete.py now generates two-job workflows matching A005 template (rebuild + stepb_deploy, classify_races, safety uploads, CLASSIFY_RACES mode). HR zones from form are written to athlete.yml. Race HR thresholds uncommented by default. fetch_fit_files.py can create fits.zip from scratch. New workflows generated for Ian/Steve/Nadi using existing secret names and pages paths — drop-in replacement, delete old single-job workflows. See HANDOVER_ONBOARDING_WORKFLOW_FIXES.md."
