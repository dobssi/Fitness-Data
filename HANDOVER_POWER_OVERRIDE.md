# Handover: Power Override + Stryd S5 Investigation
## Date: 2026-02-22

---

## What happened this session

### 1. Stryd S5 power anomaly investigation (continued from yesterday)

Yesterday's session identified that today's Battersea parkrun (21 Feb) had inflated Stryd power: 368W nPower vs 350W on Jan 17 at essentially the same pace/effort. Form power stable at 76W; leg power inflated from ~274W to 290W.

This session performed:

**S5 era-wide divergence analysis** — compared Stryd RFL vs GAP RFL across all 58 S5 runs:
- No gradual drift in Power/Speed ratio (r=-0.028 with time). NOT a progressive degradation.
- Divergence is intermittent and biased to easy runs (+0.020 mean for easy, +0.001 for hard).
- Today's parkrun was the worst single run: -3.2σ on speed-adjusted RE z-score.
- Other outlier dates: Dec 20, Dec 29, Jan 1, Jan 16-17 — several are travel/London days.

**FIT file analysis** — parsed both watch file (Battersea_from_Pod.fit) and pod native file:
- Confirmed form power stable at 76W across all Battersea races ✓
- Leg power inflated: 290W vs ~274W expected (+5.7%)
- **Stryd distance: 5.564km for official 5.000km course (+11.3%)** — smoking gun
- The stride length / IMU calibration was systematically too long, inflating both distance and leg power
- Power-speed correlation was actually 0.611 (better than yesterday's pipeline figure of 0.063, which used a broader window)

**Post soft-reset** — Paul's next run (23 Feb, flat Thames path) showed RE 0.910 — normal. Soft reset fixed the issue.

### 2. Power override feature implemented in StepB

Added `power_override_w` column support to `activity_overrides.xlsx`:

**Changes to StepB_PostProcess.py:**

a) **`load_override_file()`** — reads `power_override_w` column, converts to float, includes in summary print count.

b) **Override application block** (~line 3589) — stores `power_override_w` on dfm row during override processing pass.

c) **NPZ loop** (~line 3905) — after npower_w is computed from per-second data but BEFORE `_rf_metrics()`:
   - Computes ratio: `_pwr_ratio = override / original_npower`
   - Replaces `npower_w`, scales `avg_power_w` proportionally
   - Recalculates `nPower_HR` and `RE_avg`
   - **Scales the per-second `p` array**: `p = p * _pwr_ratio` — this is critical, ensures `_rf_metrics()`, `_calc_full_run_drift()`, and all downstream per-second calculations use corrected power
   - Prints: `Power override: <file>: 368W -> 340W (0.924x)`

d) **Post-loop fallback** (~line 4123) — for runs without NPZ cache (summary-level only), applies the same correction to dfm columns. Checks `abs(npower - override) > 1.0` to avoid double-applying.

e) **Override hash** — includes `power_override_w` in change detection hash so UPDATE mode picks up new overrides.

**First version had a bug:** power/RE were corrected on the dashboard but RFL wasn't — because `_rf_metrics()` reads the per-second `p` array directly, not `dfm['npower_w']`. Fixed by adding `p = p * _pwr_ratio` before the RF calculation.

### 3. Overrides to apply

Paul needs to add `power_override_w` column to `activity_overrides.xlsx`:

| file | power_override_w | notes |
|---|---|---|
| 2026-02-21_10-06-29.FIT | 340 | S5 leg power inflated, pre soft-reset |
| 2026-02-21_09-42-01.FIT | 240 | Preamble, same issue |

### 4. Stryd support exchange

Paul contacted Stryd support with the data. Key points:
- They refused to review "AI-generated content" (the report from yesterday's session)
- Dismissed RE comparisons because GPS was pace source (technically narrow point but misses that power itself is the issue)
- "Your baseless reports are being treated as such" — hostile response
- Eventually asked for offline sync data and PowerCenter links, which Paul provided
- The soft reset fixing the problem (RE 0.910 next day) strengthens Paul's case if needed

---

## Files modified

- **StepB_PostProcess.py** — power_override_w support (5 insertion points described above)

## Files NOT modified

- generate_dashboard.py — no changes this session (yesterday's nAG distance correction changes are still in place)
- config.py, all other scripts — unchanged

---

## Current state

- StepB with power override is ready to deploy
- Paul needs to add `power_override_w` column to activity_overrides.xlsx and set values for the two Feb 21 runs
- Run FROM_STEPB to apply
- Soft reset appears to have fixed the S5 issue — monitor going forward

---

## TODOs from this session

1. **Auto-detect Stryd power outliers** (added to memory): Fit `RE ~ speed` within era, compute z-score per run. If z < -2.5σ, auto-set factor=0 and fall back to GAP RF values. Today's parkrun at -3.2σ would be caught; legitimate trail/hilly runs wouldn't because speed regression accounts for naturally lower RE at slow pace.

2. **Monitor S5 post soft-reset** — if power inflation recurs, the `power_override_w` mechanism is in place. Consider whether auto-detection should apply the override automatically rather than requiring manual entry.

---

## For next Claude

"Power override feature added to StepB — `power_override_w` column in activity_overrides.xlsx replaces inflated Stryd nPower and scales the per-second power array before RF calculation. Two runs on 21 Feb need overrides (340W parkrun, 240W preamble). S5 pod had IMU calibration drift causing +11% distance and +5% power; soft reset fixed it. See HANDOVER_POWER_OVERRIDE.md."
