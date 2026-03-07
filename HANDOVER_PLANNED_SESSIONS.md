# Handover: Planned Sessions & CTL/ATL Projection
## Date: 2026-03-07

---

## Summary

Added planned training sessions feature to the pipeline. Athletes can define a weekly training template in `athlete.yml`, and the pipeline projects CTL/ATL/TSB forward using those planned sessions instead of zero-TSS decay. Race Readiness cards now show projected arrival CTL/TSB for each future race.

---

## What changed

### 5 files modified:

**1. `athlete_config.py`** — Added `PlannedSession` dataclass and `planned_sessions` field on `AthleteConfig`
- `PlannedSession`: day (mon-sun), description, tss
- `AthleteConfig.planned_sessions`: List of up to 2 week templates (`[week_1_sessions, week_2_sessions]`)
- Parsed from YAML `planned_sessions.week_1` / `planned_sessions.week_2`

**2. `config.py`** — Added `PLANNED_SESSIONS` export (list of week templates, each a list of `{day, description, tss}`)

**3. `athlete.yml`** — Added `planned_sessions` block with Paul's current training week:
```yaml
planned_sessions:
  week_1:
    - day: mon
      description: "Easy 50'"
      tss: 55
    - day: tue
      description: "FKS intervals"
      tss: 120
    - day: thu
      description: "Easy 60'"
      tss: 65
    - day: sat
      description: "Long run + parkrun sandwich 120'"
      tss: 200
```

**4. `StepB_PostProcess.py`** — Modified Daily sheet projection logic:
- Builds `_planned_tss` dict mapping future dates → TSS from weekly templates
- Templates repeat from tomorrow until the first planned race's taper start
- Taper window respects priority: A=14d, B=7d, C=0d
- Planned races auto-overlay with TSS estimated from historical race median at similar distance (±20%)
- Future dates with planned TSS get injected into `daily_df` before Banister calculation
- Daily sheet now includes `Planned_Description` and `Planned_Source` columns
- Projection extends to last planned day + 1 (was fixed at 30 days)

**5. `generate_dashboard.py`** — Three changes:
- **CTL/ATL chart projection** now extends to last planned session (was capped at 14 days). Return data includes TSB projection and planned session markers
- **Projected arrival card** on each Race Readiness card: shows projected race-morning CTL, TSB, and TSB as % of CTL, based on Daily sheet projection with planned sessions
- **Taper solver awareness**: planned sessions from the Daily sheet are injected into the recent_tss data so the taper solver respects user-specified sessions instead of its template when races are ≤7d away
- `zone_data` dict now includes `daily_lookup` for the race card to access projected CTL/ATL values
- `get_ctl_atl_trend()` returns 7 elements (was 5): added `tsb_proj_line` and `planned_markers`
- `unpack_ctl_atl()` updated to handle 5, 7, or 3-element returns for backward compatibility

---

## How it works

### Template to projection flow:

1. `athlete.yml` → `planned_sessions.week_1` (+ optional `week_2`)
2. `athlete_config.py` parses into `List[List[PlannedSession]]`
3. `config.py` exports as `PLANNED_SESSIONS`
4. `StepB_PostProcess.py` at Banister calculation time:
   - For each future day, checks if weekday matches a template session
   - Alternates week_1/week_2 if both defined
   - Stops repeating at taper start for the nearest A/B race
   - Overlays planned races with auto-estimated TSS
   - Injects planned TSS into `daily_df['tss_running']`
   - Banister loop sees realistic TSS values → projected CTL/ATL/TSB
5. `generate_dashboard.py` reads Daily sheet projection, extends dashed lines, shows arrival card

### Race TSS auto-estimation:

For planned races in the projection window, TSS is estimated as the median TSS from past races within ±20% of the distance. Examples from Paul's history:
- 5K: ~115 TSS (median of 15+ parkrun/5K races)
- HM: ~340 TSS (Örebro HM)
- 30K: ~414 TSS (Lidingöloppet)

Fallback: 25 × distance_km if no historical races at that distance.

---

## Important note: template TSS vs CTL maintenance

Paul's current template (440 TSS/week) will NOT maintain his current CTL of 72. At 62.9 avg daily TSS, the steady-state CTL is ~63. His recent actual weeks have been 470-580 TSS. He may want to:
- Add a 5th session (e.g. Wed easy, ~50 TSS) → 490 TSS/week → sustains ~70 CTL
- Or increase long run TSS to match actual behaviour
- The projection will make this visible immediately via the dashed CTL line trending downward

---

## Testing

### Syntax: all 4 Python files pass `ast.parse()` ✓
### YAML: `athlete.yml` parses correctly, `PlannedSession.from_dict()` produces expected output ✓
### Simulation: Banister forward projection from CTL=72 with template produces sensible results ✓

### To test on real pipeline:
1. Replace `athlete_config.py`, `config.py`, `athlete.yml`, `StepB_PostProcess.py`, `generate_dashboard.py`
2. Run StepB (FROM_STEPB mode)
3. Check Daily sheet — future rows should show planned TSS instead of 0
4. Run generate_dashboard.py
5. Check Training Load chart — dashed projection should extend ~5-6 weeks with planned-session-aware CTL/ATL
6. Check Race Readiness cards — Premiärhalvan should show projected arrival CTL/TSB

---

## Not yet done (follow-up items)

- **Planned session markers on CTL/ATL chart** — `planned_markers` data is injected into the JS but Chart.js tooltip/annotation not yet wired to show session descriptions on hover. The data is there, just needs JS tooltip formatting.
- **Ramp rate metric** — discussed as future alert enhancement. Would compute weekly CTL change and flag when >8 sustained. Not implemented in this session.
- **Onboarding form** — `planned_sessions` not yet exposed in `onboard.html`. Low priority — hand-edit YAML for now.
- **week_2 alternation** — implemented but not tested (Paul only has week_1). The logic alternates by week number since the template start date.

---

## Files for deployment

- `athlete_config.py` — dataclass changes
- `config.py` — PLANNED_SESSIONS export
- `athlete.yml` — planned_sessions block (Paul's)
- `StepB_PostProcess.py` — projection logic
- `generate_dashboard.py` — projection display + arrival card
