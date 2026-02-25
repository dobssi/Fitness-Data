# Handover: Dashboard Crash — Sheet Naming Fix
## Date: 2026-02-25

---

## Summary

Dashboard generation (`generate_dashboard.py`) crashes for all 4 athletes after the athlete ID refactoring changed Excel sheet names from `Master` to `Master (A00x)`. Root cause identified, fix defined but **not yet applied** — the file wasn't uploaded to this session.

---

## The Bug

### Symptom
All 4 athlete pipelines fail at the dashboard generation step:
```
File "generate_dashboard.py", line 57, in load_and_process_data
    df = pd.read_excel(MASTER_FILE, sheet_name='Master')
ValueError: Worksheet named 'Master' not found
```

### Root Cause
The athlete ID refactoring (done earlier today in the session logged at `2026-02-25-10-47-43-steve-onboarding-race-classifier-athlete-ids.txt`) changed `StepB_PostProcess.py` to write Excel sheets with athlete ID suffixes:

**Old**: `Master`, `Daily`, `Weekly`, `Monthly`, `Yearly`
**New**: `Master (A001)`, `Daily (A001)`, `Weekly (A001)`, etc.

`generate_dashboard.py` still hardcodes `sheet_name='Master'` at line 57 (and likely other hardcoded sheet names elsewhere in the file for Daily/Weekly/Monthly/Yearly).

### Fix
For every `pd.read_excel()` call with a hardcoded sheet name, change to read by position (sheet index) instead of by name:

```python
# Line 57 — Master sheet
# OLD:
df = pd.read_excel(MASTER_FILE, sheet_name='Master')
# NEW:
df = pd.read_excel(MASTER_FILE, sheet_name=0)  # First sheet = Master

# Similarly for other sheets:
# Daily  → sheet_name=1
# Weekly → sheet_name=2
# Monthly → sheet_name=3
# Yearly → sheet_name=4
```

Alternatively, `sheet_name=0` (or omitting `sheet_name` entirely, which defaults to first sheet) works for the Master read. For the others, grep for `sheet_name='Daily'`, `sheet_name='Weekly'`, `sheet_name='Monthly'`, `sheet_name='Yearly'` and replace with the corresponding index.

**Full audit needed**: Search `generate_dashboard.py` for ALL occurrences of `sheet_name=` to catch every instance.

---

## Context: What Else Happened This Session

### Steve Davies (A004) Prediction Accuracy Analysis
Compared parkrun vs non-parkrun 5K prediction accuracy. **No meaningful difference**:

| Category | Count | MAE | Mean Error | Within ±3% | Within ±5% |
|---|---|---|---|---|---|
| All 5K | 146 | 3.9% | -1.7% | 47% | 73% |
| Parkruns | 75 | 3.8% | -1.8% | 45% | 72% |
| Non-parkrun | 71 | 3.9% | -1.5% | 49% | 75% |

Conclusion: parkrun vs non-parkrun distinction is irrelevant. The 3.9% MAE is data-quality driven (training runs misclassified as races), not model error. Top outliers are clearly sub-maximal efforts (relay legs, warm-up runs, charity fun runs).

### Steve's Overrides Still Pending
The `activity_overrides.xlsx` has ~50 yellow REVIEW rows that need manual inspection to determine which should have `race_flag` set to 0 (training runs incorrectly flagged as races). This cleanup will improve Steve's prediction accuracy metrics but doesn't affect the pipeline itself.

---

## Files Needed for Fix Session

Upload to Claude:
1. **`generate_dashboard.py`** — the file that needs the fix
2. **`Master_FULL_GPSQ_ID_post.xlsx`** (any athlete) — for testing sheet reads
3. This handover document

---

## Current State of All Athletes

| Athlete | ID | Pipeline | Dashboard | Status |
|---|---|---|---|---|
| Paul | A001 | ✅ runs | ❌ crashes | Sheet name mismatch |
| Ian | A002 | ✅ runs | ❌ crashes | Sheet name mismatch |
| Nadi | A003 | ✅ runs | ❌ crashes | Sheet name mismatch |
| Steve | A004 | ✅ runs | ❌ crashes | Sheet name mismatch |

StepB completes successfully for all athletes. Only the dashboard generation step fails.

---

## For Next Claude

"Dashboard generation crashes for all 4 athletes because `generate_dashboard.py` hardcodes `sheet_name='Master'` but StepB now writes sheets as `Master (A00x)` after the athlete ID refactoring. Fix: change all `pd.read_excel()` calls to use sheet index (0,1,2,3,4) instead of hardcoded names. Need `generate_dashboard.py` uploaded to apply the fix. See HANDOVER_SHEET_NAMING_FIX.md."
