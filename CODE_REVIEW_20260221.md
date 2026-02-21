# Code Review: v51 Checkpoint (2026-02-21)

## Overview

38 Python files, 20,894 total lines of Python. 10 batch files, 9 markdown docs. The codebase is well-structured for a pipeline that has grown organically from single-athlete to multi-athlete. Phase 1 (YAML config) is fully wired in and working.

---

## Issues Found

### 1. Duplicated section header in `config.py` (lines 85-90)
**Severity: Cosmetic**

```python
# POWER MODE CONFIGURATION (v60)
# =============================================================================

# =============================================================================
# POWER MODE CONFIGURATION (v60)
# =============================================================================
```

The header block is duplicated verbatim. Remove lines 85-88.

---

### 2. `PIPELINE_VER = 50` in `run_pipeline.py` (line 42)
**Severity: Cosmetic / misleading**

The docstring says "v50", the constant is 50, but the pipeline is functionally v51+. The athlete.yml, config.py, and TODO.md all reference v51. Should bump to 51.

---

### 3. `export_athlete_data.py` is dead code
**Severity: Cleanup**

This reads from the "BFW" (Big Fitness Workbook) which is a legacy Excel workbook. It was replaced by `sync_athlete_data.py` which reads from intervals.icu. No Python file imports or calls it. However, `make_checkpoint.py` still lists it in `PIPELINE_FILES` (line 39). Safe to remove from repo and checkpoint list.

---

### 4. `run_multi_mode_pipeline.py` has unimplemented TODO stubs
**Severity: Low — file is not in active use**

Lines 126 and 169 contain `# TODO: Implement update mode` and `# TODO: Implement StepA_SimulatePower integration`. This file was an early Phase 2 prototype. The actual multi-athlete pipeline is now orchestrated by the GitHub Actions workflows (`pipeline.yml`, `ian_pipeline.yml`) which call `run_pipeline.py` directly. Consider whether this file still serves a purpose or is dead code.

---

### 5. `run_ian.bat` hardcodes mass, DOB, timezone (lines 55, 69, 88)
**Severity: Low — bat file is a convenience script**

`--mass-kg 87.0`, `--runner-dob "1971-11-27"`, `--tz "Europe/London"` are hardcoded rather than reading from Ian's `athlete.yml`. This works but means two sources of truth. The CI workflow reads from YAML correctly; the bat file is only for Paul's local use, so it's a low-priority cleanup.

---

### 6. `activity_overrides.yml` is empty (overrides: [])
**Severity: No action needed**

The YAML override file exists as a template but all real overrides live in `activity_overrides.xlsx`. The pipeline uses the xlsx format. This yml file is just a reference/template — harmless but could confuse. Consider adding a comment "# This file is a template. The pipeline uses activity_overrides.xlsx" or removing it.

---

### 7. `make_checkpoint.py` includes `test_phase_1_2.py` (line 53) which doesn't exist
**Severity: Cosmetic**

Listed in PIPELINE_FILES but wasn't in the checkpoint. The script handles missing files gracefully (prints "Missing files (skipped)"), so no functional impact. Remove from list.

---

### 8. Default arg fallbacks still hardcode Paul's values
**Severity: Acceptable — by design**

`StepA_SimulatePower.py`, `StepB_PostProcess.py`, `rebuild_from_fit_zip.py` all default `--tz` to "Europe/Stockholm" and `--mass-kg` to 76.0. These defaults are overridden by `run_pipeline.py` which reads from config.py → athlete.yml, so they only matter if scripts are called directly without arguments. The defaults serve as documentation of the primary athlete's values. This is fine; just worth knowing.

---

### 9. `gap_power.py` line 296 hardcodes `mass_kg = 76.0`
**Severity: Low**

This is inside the `if __name__ == "__main__"` test block, not in any importable function. Harmless for production use but should ideally be a CLI argument.

---

### 10. `README.md` is minimal (30 lines)
**Severity: Low — documentation debt**

Contains basic setup instructions but doesn't reflect the current multi-athlete architecture, the three pipeline modes, the CI/CD setup, or the dashboard features. Not blocking anything.

---

## Structural Observations (No Action Needed)

### What's working well

- **Single source of truth**: All athlete-specific values flow through `athlete.yml` → `athlete_config.py` → `config.py` → all scripts. Clean chain.
- **Mode awareness**: POWER_MODE propagates correctly through StepB, dashboard, and config. GAP/Stryd/Sim all handled.
- **CI separation**: Paul and Ian have separate workflow files with separate concurrency groups. Pages deploy to separate paths.
- **Incremental UPDATE**: The append-master-in fix for Ian is solid. Exit code 2 for "no new runs" is clean.
- **Error handling**: run_pipeline.py has good exit code handling, graceful fallbacks for missing model files, and clear failure messages.
- **Checkpoint system**: `make_checkpoint.py` is a well-thought-out tool for session continuity.

### Architecture notes

- The big three files (StepB at 5,205 lines, rebuild at 4,082 lines, dashboard at 4,397 lines) are large but well-structured with clear function boundaries. StepB has 56 functions, rebuild has 74 — these are doing a lot of work but the complexity is inherent to the domain.
- No leftover debug prints found.
- No dangerous patterns (e.g., no `eval()`, no SQL injection vectors, no credentials in code).
- `requirements.txt` correctly pins `pandas>=2.0,<3.0` (the lesson learned from the pandas 3.0 CI incident).

---

## Recommended Cleanup Actions

**Quick wins (5 min each):**
1. Remove duplicate section header in `config.py` (lines 85-88)
2. Bump `PIPELINE_VER` to 51 in `run_pipeline.py`
3. Remove `test_phase_1_2.py` from `make_checkpoint.py` PIPELINE_FILES list
4. Remove `export_athlete_data.py` from `make_checkpoint.py` PIPELINE_FILES list

**Low priority (next session):**
5. Delete `export_athlete_data.py` from repo
6. Decide fate of `run_multi_mode_pipeline.py` — delete or finish implementing
7. Add a note to `activity_overrides.yml` that it's a template
8. Update README.md with current architecture

**No action / by design:**
- Default arg fallbacks (Paul's values) in script argparsers
- Hardcoded mass in gap_power.py test block
- run_ian.bat hardcoded values (only used locally by Paul)
