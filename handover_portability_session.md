# Handover: Multi-Athlete Portability Analysis Session
## Date: 2026-02-09

---

## What happened this session

No code changes to v51. This was a pure analysis and planning session exploring portability.

### Analysis performed

1. **Portability audit** of v51 codebase — identified all Paul-specific hardcoding (era dates, PEAK_CP, mass corrections, RE reference era). Output: `v51_portability_analysis.md`

2. **No-power feasibility study** — mathematically decomposed RF = Power/HR into RF = speed×mass/(RE×HR), showed RE varies only 3.4% CV, meaning speed/HR captures ~90% of fitness signal.

3. **Prototype: Approach C** (simulate power for all runs using generic Minetti cost model). Ran full RF→trend→RFL→prediction pipeline on all 3,097 runs. Output: `no_power_prototype.py`, `no_power_prototype_output.xlsx`, `comparison_dashboard.html`

4. **Head-to-head: A vs B vs C** — discovered B (GAP/HR) and C (sim power, fixed RE) are **mathematically identical** at run-average level. C2 (speed-dependent RE) is marginally better. Output: `abc_comparison.html`

5. **Planning document** for multi-athlete architecture (v60). Output: `multi_athlete_planning.md`

### Key findings

| Approach | What | Trend corr | Trend MAE | Hilly MAE |
|---|---|---|---|---|
| A | Speed / HR | 0.890 | 2.2% | 7.4% |
| B | GAP / HR | 0.904 | 2.1% | 6.2% |
| B=C | Sim power (fixed RE) | 0.904 | 2.1% | 6.2% |
| C2 | Sim power (speed-dep RE) | 0.909 | 1.8% | 5.8% |

- B and C are identical because mass/RE is a constant that cancels in scaling
- 5K race predictions from no-power pipeline: median error +1.0%, 76% within ±5%
- HM predictions +8% too slow (Riegel scaling, not power issue — fixable)
- Era system completely eliminated in no-power mode
- Generic RE=0.92 matches S4/S5 Stryd nearly perfectly (scale 0.998/1.008)

### Architecture decisions made

- **Pipeline forks on power_mode**: Stryd → real power + simulation for gaps; no Stryd → GAP/HR
- **Garmin power excluded**: different scale, too problematic across watch generations
- **Strava API deferred**: bulk export (FIT files) sufficient; API OAuth is over-engineering for now
- **athlete.yml**: single config file per athlete holds all athlete-specific values
- **Paul is first test of GAP mode**: can run his own data through both modes to validate

### Phased implementation plan

1. Extract athlete config to YAML (no functional change)
2. Add GAP/HR mode alongside power mode
3. Test with second athlete
4. Strava bulk export support (GPX/TCX parsing)
5. Onboarding polish

### Files produced this session

All in `/mnt/user-data/outputs/`:
- `v51_portability_analysis.md` — audit of what's portable vs hardcoded
- `no_power_analysis.md` — analysis of power vs no-power approaches
- `no_power_prototype.py` — Approach C prototype script
- `no_power_prototype_output.xlsx` — full pipeline output for all 3,097 runs
- `comparison_dashboard.html` — visual overlay of power vs no-power trends
- `abc_comparison.html` — A vs B vs C head-to-head dashboard
- `multi_athlete_planning.md` — architecture and planning document

### No changes to v51 pipeline

v51 is unchanged. No checkpoint needed. All outputs are analysis artefacts.
