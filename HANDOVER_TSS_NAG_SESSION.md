# Handover: TSS Z5 Boost, HR Normalisation, Dashboard Fixes, nAG Distance Scaling
## Date: 2026-03-02

---

## Changes this session

### 1. StepB_PostProcess.py — TSS Z5 intensity boost
- **z5_frac** column: computed from per-second HR in NPZ loop — `(hr >= LTHR).sum() / total_valid_hr`
- **TSS multiplier**: `TSS *= (1 + z5_effective)` where `z5_effective = min(z5_seconds, 1200) / moving_time_s`
- **20-min cap** (z5_cap_s=1200): prevents long hard races (HM at Z5) from exceeding marathon TSS
- Added to RF_CONSTANTS, column ensure list, column order
- Not race-flag dependent — any run with Z5 time gets boosted (intervals included)
- Z5 threshold = ATHLETE_LTHR (178 for Paul)

**Impact**: 5K race 64→123 TSS, controlled parkrun 60→68, easy runs unchanged. Marathons untouched (z5=0). HM with cap: Copenhagen 521→378, Örebro 442→342. Hierarchy preserved: marathons > hard HMs > hard 10Ks > hard 5Ks.

### 2. StepB_PostProcess.py — TSS HR normalisation across athletes
- **tss_hr_scale**: `(192 - 90) / (ATHLETE_MAX_HR - 90)` — normalises HR range so equivalent relative effort produces equivalent TSS
- Paul (max 192) = scale 1.000 (unchanged), Ian (~175) ≈ 1.200, Steve (~170) ≈ 1.275
- Applied as `scaled_hr_diff = (avg_hr - 90) * hr_scale` in calc_tss, squared term
- Z5 boost multiplies the normalised TSS, so it scales proportionately

### 3. generate_dashboard.py — Milestone bug fix (mode-aware RFL)
- `get_recent_achievements()` and `get_milestones_data()` now use mode-aware RFL column
- Steve's GAP mode dashboard was showing 90.8% from hardcoded `RFL_Trend` instead of `RFL_gap_Trend`

### 4. generate_dashboard.py — Race readiness card: same-day run aggregation
- `recent_tss` now groups multiple runs on same date (warmup + race + cooldown)
- TSS summed, race activity name preferred, race flag OR'd
- Fixes Ian's parkrun Saturday showing only the cooldown in the race week plan

### 5. generate_dashboard.py — Race week plan lookback = 6 days before race
- Changed from `min(6, days_to_race)` to `max(0, 6 - days_to_race)`
- Old behaviour: 6 days before today (showed 12+ days before race for distant races)
- New behaviour: 6 days before race day, regardless of when you view it
- As race day approaches, completed days fill in from the left
- Both the table and TSB chart use the same `_solve_taper`, so both fixed

### 6. generate_dashboard.py — nAG% distance scaling: option D
- **Old**: `dist_adj = 1 + 0.04 × ln(dist/5) / ln(42.2/5)` — only boosted above 5K
- **New**: `dist_adj = 1 + 0.02 × ln(dist/5)` — penalises sub-5K AND boosts longer distances
- Adjustments: Mile -2.3pp, 3K -1.0pp, 5K ±0pp, 10K +1.4pp, HM +2.9pp, Marathon +4.3pp
- Addresses known WMA table bias towards short distances for club-level runners
- Top 10 race rankings now show good diversity across 5K, 10K, and HM

---

## Analysis performed (no code changes needed)

- **SIM vs Stryd divergence**: Paul's 2026-03-01 long run SIM RFL 88.5% vs Stryd 85.1%. Root cause: RE model speed-dependency limitation at slow paces. Not a bug — inherent model limitation, washes out at trend level.
- **Sunday long run RF suppression after parkrun**: Confirmed -1.4% RF from 101 parkrun→Sunday pairs. Strongest in 90-130min range. Today's extra suppression from 3 hard efforts in 4 days (Wed tuneup, Fri LFOTM, Sat parkrun).
- **Terrain and RF**: Hills slightly HELP RF with Stryd (captures uphill power, HR doesn't rise proportionally). Downhills suppress RF (coasting but HR elevated from preceding climb).
- **Solar radiation in Temp_Adj**: Confirmed solar IS correctly incorporated. VLM 2018 at 20°C + 698 W/m² = effective 23.5°C → Temp_Adj 1.053 matches calculation exactly. Duration-scaled to 7.9% for marathon tooltip.

---

## Files changed
- `StepB_PostProcess.py` — z5_frac column, Z5 TSS boost with 20-min cap, HR normalisation
- `generate_dashboard.py` — milestone mode-aware RFL, same-day aggregation, race week lookback, nAG distance scaling option D

## Requires
- FROM_STEPB run to populate z5_frac from NPZ data (for Z5 boost)
- `ATHLETE_MAX_HR` set in athlete.yml (already present for all athletes)
- Dashboard refresh to see nAG and race readiness changes
