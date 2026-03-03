# Handover: Onboarding System Build Session
## Date: 2026-03-02

---

## Summary

Built the end-to-end athlete onboarding system: from a self-service web form through to automated pipeline setup and first CI run. Tested with PaulTest as a live dry run — INITIAL pipeline triggered and running at end of session.

---

## What was built

### 1. Unified onboard.html (replaces old onboard.html + pb_corrections.html)

Single-page athlete profile form with dark theme matching dashboard. Serves both initial onboarding and return visits (localStorage persistence).

**Sections:**
- About You — name, email, DOB, gender, weight, timezone
- Heart Rate — max HR, LTHR with "estimate from data" option
- Your Data — Dropbox file request link for uploads + intervals.icu connection
- Race Times — preset distance buttons (Mile/3K/5K/10K/HM/Marathon/Custom), chip time entry with pace calculation, generates official_time_s overrides
- Upcoming Races — dropdown distance selector with custom option, A/B/C priority
- Submit → mailto to paul.collyer@gmail.com with human-readable summary + JSON config block

**Key design decisions:**
- No file attachment (mailto can't do it, iOS problematic) — JSON encoded in email body between `--- CONFIG JSON START ---` / `--- CONFIG JSON END ---` markers (also supports old `CONFIG JSON` + `===` format)
- Strava/Garmin zip uploaded separately via Dropbox file request: https://www.dropbox.com/request/UWGoW1kajiL1H9FacteU
- localStorage remembers all entries for return visits

### 2. onboard_athlete.py updates

**New features:**
- Accepts pasted email body as `--config` input (extracts JSON between markers, strips trailing spaces from email)
- Auto-assigns athlete ID (A005, A006...) by scanning existing `athletes/` folders
- Folder uses ID (e.g. `athletes/A005/`), slug uses full name (e.g. `paul_test`) for workflow/pages/secrets
- Writes PB race_times into activity_overrides.xlsx as official_time_s overrides
- Generated workflows include INITIAL mode and call `ci/initial_data_ingest.py`
- Timeout set to 600 minutes
- Post-creation summary shows exact steps with git commands and secret names

**Slug changed:** `make_slug("Ian Lilley")` now returns `ian_lilley` (was `ian`) to avoid collisions.

### 3. ci/initial_data_ingest.py (NEW)

Standalone script called during INITIAL pipeline mode. Lists the athlete's `data/` folder on Dropbox, downloads any zip found (regardless of filename), auto-detects Strava vs Garmin export, and produces `fits.zip` + `activities.csv`.

**Detection logic:**
- Zip contains `activities/` folder → Strava → runs strava_ingest.py
- Zip contains .fit files → Garmin → extracts FITs into fits.zip
- No zip found → falls back to intervals.icu sync

### 4. onboard_new_athlete.bat

One-click batch file. Expects `new_athlete.txt` in DataPipeline root (paste email body), runs onboard_athlete.py.

### 5. Supporting changes

- **pipeline.yml** — deploys onboard.html + pb_corrections.html to gh-pages
- **push_dashboard.py** — also copies/pushes onboard pages locally
- **.gitignore** — appended `new_athlete.txt` and `checkpoint_v52_*.zip` to existing
- **make_checkpoint.py** — added onboard_new_athlete.bat, .gitignore

---

## Admin flow for a new athlete

1. Athlete fills in form at `https://dobssi.github.io/Fitness-Data/onboard.html`
2. Athlete clicks Submit → email arrives with JSON config in body
3. Athlete uploads Strava/Garmin zip via Dropbox file request (any filename)
4. Admin pastes email into `DataPipeline/new_athlete.txt`
5. Admin double-clicks `onboard_new_athlete.bat`
6. Script creates `athletes/A00X/` + `.github/workflows/name_pipeline.yml`
7. Admin copies `athletes/A00X/` to Dropbox, moves athlete's zip into `data/`
8. Admin adds GitHub secrets if intervals.icu provided (script prints exact names)
9. Admin runs Git_Push
10. Admin triggers workflow with mode = INITIAL

---

## Current test run status

PaulTest (A005) INITIAL pipeline triggered and running. Uses:
- Secrets: PAUL_TEST_INTERVALS_API_KEY, PAUL_TEST_INTERVALS_ATHLETE_ID
- Strava export uploaded to Dropbox as custom filename (not strava_export.zip)
- ci/initial_data_ingest.py handles the auto-detection

Previous failures resolved:
- Missing ci/initial_data_ingest.py → pushed
- File in repo root instead of ci/ → git mv'd
- Import name mismatch (dropbox_download_file → dropbox_download) → fixed

---

## Files changed this session

| File | Status | Notes |
|---|---|---|
| onboard.html | **REWRITTEN** | Unified profile page, replaces old version |
| onboard_athlete.py | **UPDATED** | Email parsing, auto-ID, PBs, INITIAL mode, slug change |
| ci/initial_data_ingest.py | **NEW** | Dropbox zip finder + Strava/Garmin auto-detect |
| onboard_new_athlete.bat | **NEW** | One-click onboarding |
| .github/workflows/paul_test_pipeline.yml | **REGENERATED** | Uses initial_data_ingest.py |
| .github/workflows/pipeline.yml | **UPDATED** | Deploys onboard pages to gh-pages |
| push_dashboard.py | **UPDATED** | Also pushes onboard pages |
| .gitignore | **UPDATED** | Added new_athlete.txt, checkpoint_v52_*.zip |
| make_checkpoint.py | **UPDATED** | Added new files to manifest |
| pb_corrections.html | KEPT | Legacy, functionality merged into onboard.html |

---

## Known issues / TODOs

1. **INITIAL run result pending** — check PaulTest pipeline outcome
2. **Old athlete folders** (IanLilley, NadiJahangiri, SteveDavies, PaulTest) still use name-based folders. Refactoring to A00X IDs is a TODO for later.
3. **Old paultest_pipeline.yml deleted** — only paul_test_pipeline.yml remains
4. **Onboard summary text** still mentions "strava_export.zip OR garmin_export.zip" in the printed instructions — could simplify since initial_data_ingest.py now handles any filename
5. **Formspree/Google Forms** — future upgrade path to eliminate mailto if scaling beyond ~10 athletes

---

## For next Claude

"Onboarding system is built and live-testing. The form at onboard.html submits athlete config via email (JSON in body). Admin pastes email into new_athlete.txt, runs onboard_new_athlete.bat, which creates athletes/A00X/ with auto-assigned ID. ci/initial_data_ingest.py handles any-name zip from Dropbox during INITIAL mode. PaulTest (A005) INITIAL run was triggered — check outcome. See HANDOVER_ONBOARDING.md for full context."
