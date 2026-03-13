# Handover: Planned Sessions, Bug Fixes, Fitness Analysis
## Date: 2026-03-07/08

---

## Session Summary

Multi-topic session: fitness analysis, planned training sessions feature, several bug fixes, and scoping for the athlete page.

---

## 1. Fitness Analysis (Paul)

### Finding: Alert is fatigue, not fitness loss
- RFL_Trend (Stryd) declined 0.911 → 0.878 over 8 weeks (-3.5%)
- Alert fires in Stryd mode only (RFL -2.5%, CTL +5 over 28d window)
- GAP mode at -1.9% — just under the 2.0% threshold
- Stryd captures fatigue-related efficiency loss more sensitively than GAP
- **Root cause:** 3 weeks of sustained negative TSB (-11 to -15), not ramp rate
- Weekly CTL ramp is gentle (+1.3-1.4/week) — well within safe bounds
- LFOTM 5K (Feb 27, 19:42, 350W NP) confirms race fitness is intact
- Non-race RF dropped from ~1.85 (Jan) to ~1.75 (Mar) — classic overreaching signature

### Ramp rate context (from literature)
- Friel: 5-8 CTL points/week is normal range
- Couzens: more conservative, ~10 CTL/month for age-group athletes
- Paul's actual: ~1.4/week — very safe
- **TODO:** Add ramp rate as a pipeline metric and use in alerts

---

## 2. Planned Sessions Feature (NEW)

### Files modified:
- **`athlete_config.py`** — `PlannedSession` dataclass, `planned_sessions` field on `AthleteConfig`
- **`config.py`** — `PLANNED_SESSIONS` export
- **`athlete.yml`** — `planned_sessions.week_1` block (Paul's training week)
- **`StepB_PostProcess.py`** — Daily sheet projection uses planned sessions instead of zero-TSS decay
- **`generate_dashboard.py`** — Extended projection on CTL/ATL chart, projected arrival card on Race Readiness

### Design decisions:
- Up to 2 weekly templates (week_1, week_2), played **once** (not repeating)
- 1 week template = 7 days projection, 2 weeks = 14 days
- Planned races within the template window get auto-estimated TSS from historical race median
- Template does NOT repeat to taper start — that's for the athlete page (JS-side, interactive)
- Race entries excluded from taper solver injection (prevents post-race TSB being shown as morning TSB)

### Current state:
- StepB projection working — Daily sheet has correct planned session TSS
- Dashboard projection lines working — dashed CTL/ATL/TSB extend through planned days
- Projected arrival card on Race Readiness — needs verification (may have failed with earlier `_daily_lookup` error)
- **Known issue:** projection tail extends 30 days of zero-TSS decay after template ends. Need to cap `future_end` in `get_ctl_atl_trend()` to last non-null planned day only

### Template TSS warning:
Paul's template (440 TSS/week) will show CTL drifting down from 72 to ~63. His actual weeks are 470-580 TSS. May want to add a 5th session or increase TSS values.

---

## 3. Bug Fixes

### a) `_daily_lookup` scoping error (generate_dashboard.py)
- **Symptom:** `UnboundLocalError: cannot access local variable '_daily_lookup'` in `get_zone_data()`
- **Cause:** Planned sessions injection block referenced `_daily_lookup` before it was defined
- **Fix:** Removed dependency, reads Daily sheet directly via `pd.read_excel()`

### b) Ian's race week plan showing post-race TSB (generate_dashboard.py)
- **Symptom:** Paddock Wood Half showing TSB -18.7 "Race morning" — actually post-race
- **Cause:** Planned race TSS from Daily sheet was injected into `recent_tss`, solver treated it as "done"
- **Fix:** Skip entries with `Planned_Source == 'race'` during injection — solver handles race day correctly (TSS=0)
- Ian's actual race morning TSB: ~+12.8 (+22% of CTL) — in the zone

### c) Race tag colour (generate_dashboard.py)
- Changed `.dt-r` CSS from green (#4ade80) to yellow (#fbbf24) matching Recent Runs race badge

### d) DD/MM/YYYY pending activities not matching (StepB + apply_run_metadata)
- **Symptom:** Activity names from Feb 10-20 showing as generic "Morning run"/"Afternoon run"
- **Fix (StepB):** Added `_date_dmy_pat` fallback regex for DD/MM/YYYY
- **Fix (apply_run_metadata):** Normalises incoming `--date` to YYYY-MM-DD

### e) Projection extending to September (StepB_PostProcess.py)
- Template now plays once (7 or 14 days), races only overlaid within template window

---

## 4. Athlete Page Scoping (NOT built)

- `athlete.html` — client-side Banister projection in JS, instant updates
- Planned sessions editor + live CTL/ATL/TSB chart
- localStorage for edits, Dropbox JSON + clipboard for persistence
- Dedicated session to build

---

## 5. PaulTest Classify Races

Not assessed — upload PaulTest Master to next session for review.

---

## Files to Deploy

| File | Location | Changes |
|------|----------|---------|
| `athlete_config.py` | root | PlannedSession dataclass |
| `config.py` | root | PLANNED_SESSIONS export |
| `athlete.yml` | root | planned_sessions block |
| `StepB_PostProcess.py` | root | Planned projection + DD/MM/YYYY fix |
| `generate_dashboard.py` | root | All dashboard fixes |
| `apply_run_metadata.py` | ci/ | Date normalisation |

---

## Known Issues for Next Session

1. **Projection tail** — 30 days of zero-TSS decay after template. Cap in `get_ctl_atl_trend()`
2. **`nan` markers** — rest days show `{"desc": "nan"}` instead of null in planned array
3. **Arrival card** — verify rendering after `_daily_lookup` fix deployed
4. **Ramp rate** — scoped, not implemented
5. **Athlete page** — scoped, ready to build
