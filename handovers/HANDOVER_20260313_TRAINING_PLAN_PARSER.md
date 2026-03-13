# Handover: Training Plan PDF Parser + Claude Code Setup
## Date: 2026-03-13

---

## Summary

First session on Claude Code (macOS). Major feature: training plan PDF/text parser that lets any athlete drop a schedule into `user_data/` and get Banister CTL/ATL projection + dashboard card. Also: codebase cleanup, CLAUDE.md refresh, permissions setup.

---

## Files Changed

| File | Status | Changes |
|------|--------|---------|
| `ci/parse_training_plan.py` | **NEW** | PDF/text training plan parser. Swedish + English. Session classification, duration extraction, TSS estimation (calibrated from A001 history). Outputs date-anchored `planned_sessions.yml`. |
| `ci/merge_user_data.py` | **Modified** | Downloads PDF and TXT files from `user_data/` on Dropbox. |
| `ci/workflow_template.yml` | **Modified** | Parse training plan step (after merge_user_data, both jobs). `--planned-sessions` flag on all StepB calls. Cleanup: removes `planned_sessions.yml` when no PDF/TXT present. Detects both `.pdf` and `.txt`. |
| `StepB_PostProcess.py` | **Modified** | `--planned-sessions` arg. `calc_ctl_atl()` reads date-anchored `planned_sessions.yml` with auto-discovery fallback via `ATHLETE_CONFIG_PATH`. Injects race from plan into `PLANNED_RACES`. |
| `generate_dashboard.py` | **Modified** | `get_upcoming_sessions()` extracts planned sessions from Daily sheet. `_build_upcoming_sessions_html()` renders card with weekly TSS totals, rest/race styling, taper summary. Card positioned after Training Load chart. |
| `requirements.txt` | **Modified** | Added `pdfplumber`. |
| `.github/workflows/*.yml` (x7) | **Modified** | All 7 athlete workflows: parse training plan step (both jobs), `--planned-sessions` on all StepB calls, `.pdf` + `.txt` detection. |
| `CLAUDE.md` | **Modified** | Refreshed TODOs from latest handovers, updated file structure (folder renames done, new files added), A001 noted as bespoke workflow. |
| `ATHLETE_GUIDE_GAP.md` | **Modified** | Added `user_data/` section: training plan format (PDF/text), FIT uploads, activities.csv, weight.csv. |
| `.claude/settings.json` | **NEW** | Project-level Claude Code permissions. |
| `~/.claude/settings.json` | **NEW** | User-level Claude Code permissions (auto-allow Bash, Read, Write, Glob, Grep). |

### Deleted
| File | Reason |
|------|--------|
| 15 `.bat` files | Replaced by GitHub Actions + `run_pipeline.py` |
| `CLAUDE_RUNNING_PROJECT_OVERVIEW.md` | Superseded by `CLAUDE.md` |
| `MIGRATION_PLAN.md` | Migration completed |
| `dir.txt` | Stale snapshot |
| `gitignore` (lowercase duplicate) | `.gitignore` exists |

### Moved
| From | To |
|------|-----|
| 21 `HANDOVER_*.md` files | `handovers/` directory |

---

## Validation Results

| Athlete | Mode | Result |
|---------|------|--------|
| A005 PaulTest | UPDATE | PDF parsed, 30 planned sessions in projection. Dashboard deployed. |
| A005 PaulTest | DASHBOARD | Upcoming Sessions card rendered with 23 sessions. |
| A005 PaulTest | FROM_STEPB | `--planned-sessions` flag working. "Loaded planned_sessions.yml: source=Specifik". "Date-anchored plan: 23 sessions, ending 2026-04-05". Card shows 4 weeks with weekly TSS. |
| A005 PaulTest | UPDATE (pre-feature) | Cleanup commit passed — no regressions. |

---

## Key Decisions

- **PDF is source of truth** for planned sessions. Remove PDF = clear plan on next run.
- **Plain text supported** alongside PDF — athletes can write simple schedules.
- **TSS rates calibrated from A001** actual data: F sessions ~2.3/min, E sessions ~2.3/min, easy ~1.1/min, long ~1.5/min. `--master` flag enables per-athlete calibration from history.
- **Date-anchored YAML** (`planned_sessions.yml`) rather than extending `athlete.yml` week templates. Separate concern, any number of weeks, overwrite-friendly.
- **`planned_sessions.yml` auto-discovery** via `ATHLETE_CONFIG_PATH` env var as fallback when `--planned-sessions` not passed explicitly.
- **Athlete guide** distributed to all 7 athletes' `user_data/` folders on Dropbox.
- **Upcoming Sessions card** positioned between Training Load and Volume charts.

---

## Known Issues

- **Card styling** needs polish to match dark theme (rest rows more subtle, better truncation, weekly totals more visual separation). Next session.
- **Partial first week** in upcoming sessions shows lower TSS than expected (only includes future days of current week).
- **Race date from PDF** (April 12 Hannover) causes CTL/ATL projection to extend with zero-TSS decay after last planned session. Misleading if athlete doesn't have that race. Could cap projection at last planned session + 1 day instead of race day.
- **TSS estimation** still approximate without `--master` calibration. F8 fartlek at 83 TSS (actual ~120-140). Calibration from master improves this.
- **Python 3.9 on macOS** — `generate_dashboard.py` doesn't compile locally (f-string syntax needs 3.12+). Works on CI (3.12). Not a blocker.

---

## Outstanding / Next Session

- **Tidy Upcoming Sessions card** — styling to match dark theme
- **Prediction chart trend line for GAP athletes** — JS fallback needed (high priority bug)
- **Prediction tuning** — Paul and Ian predictions feel too slow
- **`power_adjuster_to_S4` rename** — cross-codebase rename
- **Johan A007** — test with FK Studenterna PDF in his user_data

---

## For Next Claude

"First Claude Code session. Built training plan PDF/text parser (`ci/parse_training_plan.py`) — athletes drop a schedule into `user_data/` on Dropbox, pipeline parses it (Swedish + English), estimates TSS, generates `planned_sessions.yml`. StepB reads date-anchored sessions for Banister CTL/ATL projection. Dashboard shows dashed projection lines + new Upcoming Sessions card with weekly TSS totals. All 7 workflows updated. Also: root cleanup (15 .bat files, stale docs deleted, handovers archived to `handovers/`), CLAUDE.md refreshed, athlete guide updated and distributed. Validated on A005 PaulTest (UPDATE + FROM_STEPB + DASHBOARD). Next: tidy card styling, GAP prediction chart fix, prediction tuning. See handovers/HANDOVER_20260313_TRAINING_PLAN_PARSER.md."
