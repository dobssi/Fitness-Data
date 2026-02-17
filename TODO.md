# Pipeline TODO — v51
## Updated: 2026-02-17

---

## Current State Summary

v51 is a mature three-mode pipeline (Stryd/GAP/SIM) processing 3,100+ runs from 2013–2026. Phase 1 (YAML athlete config) is wired in. Dashboard has dark theme, training zones, race readiness cards, and full mode switching. CI runs daily via GitHub Actions with Dropbox sync.

### What's working
- Three parallel RF models (Stryd power, GAP physics, SIM athlete-specific) correlating >0.989
- 4 validated alerts: CTL↑/RFL↓, taper ineffective, deep fatigue, easy run outlier
- Race predictions with surface-adjusted continuous power-duration curve
- Recovery jog HR-lag filter for interval sessions
- Cold + heat + extreme heat temperature adjustments (U-shaped, cold NOT duration-scaled)
- Per-second NPZ cache with COT spike exclusion in rebuild
- GAP race zones from per-second pace data (rpz_gap)
- Alert bitmask (0-15) per mode with enriched text
- Weight integration from intervals.icu
- NPZ cache sync: CI → Dropbox → local

---

## Active TODOs

### 1. Surface-specific effort specificity for trail race cards
**Priority: Medium** — Affects LL30 and trail race readiness cards only

Race card "14d/28d at effort" currently counts flat road runs at equivalent power toward trail race prep. A 10K tempo on flat road at HM power is not meaningful LL30 preparation.

**Needed**: Use NPZ per-second grade data to classify runs as hilly (undulation_score > threshold) and only count hilly runs for trail race specificity. Flat runs still count for road race cards.

### 2. PS Floor bias
**Priority: Low** — Partially addressed

PS Floor still ~0.7% generous after divisor raised from 180→184. 64% → ~40% of races trigger the floor. Not urgent — the remaining bias is small and conservative (slightly flatters bad-day races).

### 3. Bannister mile surface override
**Priority: Low** — Manual fix

Tagged TRACK in overrides but was actually a road race. Fix in `activity_overrides.yml`.

### 4. Zones for SIM mode
**Priority: Very low**

Zone table uses Stryd power zones for both Stryd and SIM. SIM CP is 338W vs 337W — 1W difference. Only matters if the models diverge further.

---

## Future: Multi-Athlete (v60)

Architecture planned in `multi_athlete_planning.md`. Key decisions made:

- Pipeline forks on `power_mode`: Stryd → real power; no Stryd → GAP/HR
- Garmin power excluded (scale inconsistency across watch generations)
- `athlete.yml` per athlete holds all athlete-specific values
- Phase 1 (YAML config) is done and wired in
- GAP/HR mode validated: 0.904 correlation, 2.1% MAE vs power

### Remaining phases
- **Phase 2**: Add GAP/HR mode alongside power mode (~2 sessions)
- **Phase 3**: Test with second athlete (~2 sessions)
- **Phase 4**: Strava bulk export support — GPX/TCX parsing (~1 session)
- **Phase 5**: Onboarding polish — template folder, setup script (~1-2 sessions)

---

## Backlog (no urgency)

- **Simulation power 3-4% too low** in pre-Stryd/v1_late eras. Mitigated by GAP era overrides. Would need model retraining to fix properly.
- **Strava elevation corrections** — 30 runs with Strava elevation = 0. Manual Strava corrections, 9 high priority. Affects terrain_adj gating for those runs.
- **v1_late simulation approach** — 37 runs using sim power due to dying v1 pod. Working adequately.
- **Pipeline guide** — `pipeline_guide.md` was deleted (referenced v47). Could write a fresh one if needed.

---

## Recently Completed (for reference)

### Feb 16
- Alert 5 retired (false positives at peak fitness)
- Cold temperature adjustment (U-shaped, NOT duration-scaled)
- Specificity "at or above" for 5K races (Sub-5K + 5K minutes)
- Easy run mask: HR≤148 OR nPower<mid-Z2
- Training zones anchored via RF×HR (5-zone LT model)

### Feb 15
- Planned races: A/B/C priority, chart markers, taper alerts
- Surface-adjusted predictions (continuous P-D curve × surface multipliers)
- LL30 validated: 277W/2:39 predicted vs 271W/2:41 actual

### Feb 14
- GAP-calibrated era overrides (Stryd-GAP gap → 0.1%)
- GAP terrain adjustment
- PS ceiling for short race HR-lag
- PS_sim, Easy RF for GAP/SIM

### Feb 13
- Recovery jog HR-lag filter (gate + filter, grade-aware descent)
- Extreme heat quadratic (k=0.0015 above 25°C)
- PS Floor divisor 180→184
- Dead_frac gate raised to 0.20 (hill repeats)
- Mode toggle fully functional (rpz_gap for GAP race zones)

### Feb 11
- COT spike exclusion in dashboard (later moved to rebuild)
- Dark theme dashboard, DM Sans font
- Training zones section (HR/Power/Race Effort views)

### Feb 10
- Dashboard restyle complete
- NPZ cache sync: CI → Dropbox

### Feb 9
- Phase 1 YAML config built and verified safe
- Multi-athlete portability analysis
- No-power feasibility study (B=C mathematical proof)
