# Handover: Race History Comparison Cards
## Date: 2026-03-04

---

## What happened this session

Added **Race History** section to `generate_dashboard.py` — a side-by-side race comparison tool positioned between Top Race Performances and All-Time Milestones.

### Feature overview

Two stacked card slots, each with:
1. **Distance filter dropdown** — All distances / 3K / 5K / 10K / 10M / HM / 30K / Marathon (auto-populated from actual race data)
2. **Race select dropdown** — searchable list showing `date · name · distance · time`
3. **Race card** — matching the Race Readiness card design

### Card metrics (3 rows)

**Row 1 — Performance:**
- Time (with pace/km below)
- Avg HR
- nAG% (normalised AG, with raw AG as subtitle)
- TSS

**Row 2 — Training state:**
- TSB (colour-coded: green if positive, amber if negative)
- CTL
- RFL (switches to RFL_gap in GAP mode via `currentMode`)
- ATL

**Row 3 — Preparation + conditions:**
- 14d effort mins (HR race zones)
- 42d effort mins (HR race zones)
- Temperature + terrain indicator

### Delta comparison panel

When both slots have a race selected, a comparison panel appears below showing the difference (B minus A) for: time (formatted M:SS, green=faster, red=slower), nAG, CTL, TSB, 14d/42d effort, avg HR.

Time comparison only shown when both races are the same distance category (otherwise meaningless).

---

## Implementation details

### Data function: `get_race_history_data(df, ctl_atl_lookup, zone_data)`

- Extracts all races from master (race_flag=1)
- Computes normalised AG on-the-fly using `_calc_normalised_ag()` (same as get_top_races)
- Looks up CTL/ATL/TSB from daily lookup dict
- Computes 14d/42d effort mins using HR race zone data from zone_runs
- Returns list of race dicts sorted by date descending

### HTML generator: `_generate_race_history_html(race_history_data)`

- Generates the section HTML + all JavaScript inline
- CSS classes prefixed with `rh-` to avoid collision with existing `rc`/`rh` race readiness classes
- Mobile responsive: stacks cards vertically below 599px

### Wiring

- `main()` calls `get_race_history_data()` after milestones, passes to `generate_html()`
- `generate_html()` accepts `race_history_data=None` parameter
- HTML template calls `_generate_race_history_html()` between Top Races and Milestones

---

## Files changed

| File | Change |
|---|---|
| `generate_dashboard.py` | +360 lines: `get_race_history_data()`, `_generate_race_history_html()`, CSS, wiring |
| `TODO.md` | Race History moved to completed |

---

## Not yet implemented (Phase 2 candidates)

- **Predicted-at-the-time** — show what the pipeline predicted for this race on race day. Requires storing historical predictions in the master (not currently retained).
- **Long run tail metrics per race** — time beyond 80% of race duration in qualifying long runs. Currently only on Race Readiness cards.
- **Surface-specific effort** — count only trail runs for trail race cards, only road for road.
- **Sparkline CTL curve** — mini chart showing 42d CTL trajectory into race day.

---

## PaulTest INITIAL rebuild

Not yet run. Plan from prior session:
1. Delete from Dropbox A005: output/, persec_cache/, fit_sync_state.json, pending_activities.csv, athlete_data.csv, activity_overrides.xlsx, data/fits.zip, data/activities.csv
2. Keep: athlete.yml, data/PaulTest_export_*.zip, onboard_config.json
3. Push updated generate_dashboard.py + TODO.md
4. Trigger INITIAL — two-job workflow handles 6h limit

---

## For next Claude

"Race History comparison section added to generate_dashboard.py. Two card slots with distance filtering, race selection, full training context, and delta comparison. File compiles clean, ready to deploy. PaulTest INITIAL rebuild hasn't run yet — needs Dropbox cleanup + INITIAL trigger. See HANDOVER_RACE_HISTORY.md."
