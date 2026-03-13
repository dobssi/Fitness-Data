# Handover: Onboarding Workflow Refactor
## Date: 2026-03-13

---

## Summary

Replaced the broken inline f-string workflow generator in `onboard_athlete.py` with a template-based approach using `ci/workflow_template.yml`. The template is based on the proven Paul Test (A005) two-job workflow and fixes all 20 known gaps in the old generator.

---

## Files Changed

| File | Status | Changes |
|------|--------|---------|
| `ci/workflow_template.yml` | **NEW** | 635-line workflow template with `{{PLACEHOLDER}}` substitution. Based on A005 two-job workflow + Johan's gpx_tcx_summaries additions. Schedule commented out by default. |
| `onboard_athlete.py` | **Modified** | `generate_workflow_yml()` now loads template + does string replacement. Old generator preserved as `_generate_workflow_yml_inline()` fallback. Added `--athlete-id` and `--slug` CLI args for re-onboarding. |
| `TODO.md` | **Modified** | Onboard refactor moved from active to completed. |

---

## What the template fixes (all 20 gaps vs old generator)

1. Two-job split (rebuild + stepb_deploy) with fresh 6h clock for INITIAL
2. `--full` fetch on INITIAL to get all intervals.icu FITs
3. `--cache-full` on Dropbox download
4. `--weight-oldest` on sync_athlete_data
5. Weather cache dir mkdir
6. classify_races.py step with `--skip-if-classified`
7. Safety upload (always runs, weather cache + Master)
8. Correct bash `${EXIT_CODE:-0}` (not double-braced `${{EXIT_CODE:-0}}`)
9. `gpx_tcx_summaries.csv` in download + rebuild + upload
10. Proper operator precedence in all `if` conditions
11. `PYTHONUNBUFFERED: 1`
12. Mass extracted from YAML at runtime (not hardcoded)
13. Mode output exposed for job2 conditions
14. INITIAL skips persec cache restore
15. INITIAL does full persec cache upload
16. Re-classify races with predictions (stepb_deploy job2)
17. StepB second pass after re-classify
18. Strava flag conditional on file existence
19. Reset sync state on manual dispatch
20. Upload fits.zip on INITIAL/FULL too

## Template placeholders

| Placeholder | Example | Source |
|-------------|---------|--------|
| `{{ATHLETE_NAME}}` | Johan | `cfg["name"]` |
| `{{ATHLETE_SLUG}}` | johan | `make_slug(name)` or `--slug` |
| `{{ATHLETE_ID}}` | A007 | `next_athlete_id()` or `--athlete-id` |
| `{{ATHLETE_FOLDER}}` | A007 | Same as athlete_id |
| `{{PAGES_SLUG}}` | johan | Same as slug |
| `{{DOB}}` | 1975-07-14 | `cfg["dob"]` |
| `{{GENDER}}` | male | `cfg["gender"]` |
| `{{TIMEZONE}}` | Europe/Stockholm | `cfg["timezone"]` |
| `{{SECRET_PREFIX}}` | JOHAN | `slug.upper()` |
| `{{CRON_SCHEDULE}}` | 0 9,11,...,23 * * * | Hardcoded (schedule is commented out anyway) |

## Validation

- Generated A005 workflow matches reference exactly (modulo gpx_tcx additions)
- Generated Johan workflow diff shows only improvements (UPDATE rebuild gains extra-summaries)
- Generated test athlete (Mo Farah) produces clean 635-line workflow, 0 unresolved placeholders
- End-to-end `--config` + `--dry-run` + full run all tested
- Syntax check passes

## CLI usage

```bash
# New athlete — auto-assigns next ID (A008, A009, etc.)
python onboard_athlete.py --config new_athlete.json --output-dir .

# Re-onboard existing athlete
python onboard_athlete.py --config athletes/A007/onboard_config.json --athlete-id A007 --slug johan --output-dir .

# Dry run
python onboard_athlete.py --config config.json --dry-run
```

## Design decisions

- **Schedule commented out by default** — uncomment after INITIAL is validated and working
- **Template not inline fallback** — if `ci/workflow_template.yml` is missing, falls back to old generator with deprecation warning
- **No conditional blocks in template** — all athletes get gpx_tcx_summaries support (conditional on file existence in bash), classify_races, etc. Simpler than Mustache-style conditionals.
- **Timezone not validated** — onboard.html form should send IANA names. Could add alias map later (UK→Europe/London etc.)

---

## Pending / not changed

- Johan A007 INITIAL still running (separate from this work)
- `user_data` folder pattern not implemented this session
- Existing hand-built workflows unchanged — template only affects new athletes going forward

---

## For Next Claude

"Onboarding refactor complete. `onboard_athlete.py` now uses `ci/workflow_template.yml` (template-based, 10 placeholders, simple string replacement) instead of 520-line inline f-strings. Fixes all 20 gaps vs old generator. Schedule commented out by default. New CLI args `--athlete-id` and `--slug` for re-onboarding. Validated against A005, Johan, and test athlete. See HANDOVER_ONBOARDING_REFACTOR.md."
