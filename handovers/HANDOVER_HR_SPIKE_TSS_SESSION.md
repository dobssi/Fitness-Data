# Handover: HR Spike Filter + TSS Investigation
## Date: 2026-03-14

---

## Session Summary

Two-part session: (1) Extended HR spike filter to handle mid-session pauses, with overshoot correction attempted and reverted. (2) Investigated TSS discrepancy across A001/A005/A006 for Haga parkrun.

---

## HR Spike Filter — Mid-Session Pauses

### Problem
Mar 10 run had a 686s watch pause mid-run. After resuming, HR spiked to 191 (vs equilibrium 175) before settling. The existing Swiss cheese model only handled session-start spikes — settling time was capped at `min(pause, 60) + HR_LAG_S = 75s`, far too short for a 10+ minute pause.

### Fix landed: Settling time scaling
New formula in StepB_PostProcess.py (~line 2539 and ~2559):
```python
settling_time = max(min(pause_duration, 60.0),
                    min(pause_duration * 0.3, 180.0)) + HR_LAG_S
```
- Short pauses (<=200s): unchanged behaviour (min(pause, 60) + 15 = up to 75s)
- Long pauses: 30% of pause + 15s, capped at 195s
- `max()` ensures long pauses never get LESS settling than short ones
- Blast radius: safe across all 3,127 NPZ files

### Overshoot filter: attempted and reverted
After the ramp-up phase, HR overshoots equilibrium by ~10-15 bpm then settles. Tried ratio-based filtering (flag seconds where power:HR ratio is >5% below reference). Result: 121/3,127 runs triggered, 66 lost >50% valid RF data. The threshold can't distinguish overshoot from normal hill/effort variation. Reverted entirely. Comment left in StepB noting this is deferred to own session.

---

## TSS Discrepancy — A001 vs A005/A006

### Problem
Mar 14 Haga parkrun: A001 TSS=103.5, A005 TSS=168.0, A006 TSS=169.1. Same runner, same race, RFL ~87-88% across all three.

### Root cause: A001 missing z5_frac (NPZ cache miss)
Base TSS is nearly identical across all three (~71.7-72.2). The gap is the Z5 intensity boost:
- A005/A006: z5_frac=0.90 (90% time above LTHR) → TSS *= 1.9 → ~137
- A001: z5_frac=NaN (NPZ cache not found during StepB) → no boost → 71.7

When StepB can't find the NPZ cache for a run, it `continue`s past all per-second computations including z5_frac. The Z5 boost condition `pd.notna(z5_frac) and z5_frac > 0` is False, so no multiplier applied.

### Resolution
A001 UPDATE triggered. The NPZ cache should exist from the rebuild job, so next StepB pass will find it and compute z5_frac correctly. TSS should match A005/A006 (~137 for the parkrun).

### CHECK FIRST TASK NEXT SESSION
Verify A001's Mar 14 parkrun TSS is now ~137, not 103.5.

---

## A001 Training Plan Not Picked Up

### Problem
A001 bespoke workflow (`paul_collyer_pipeline.yml`) doesn't have the `merge_user_data.py` step. Training plan file copied to `user_data/` on Dropbox is never downloaded or processed.

### Status
Root cause identified but **fix not yet applied**. Need to add the `merge_user_data.py` step to A001's workflow (matching what template-generated workflows have). This is in the stepb_deploy job's Dropbox download section.

---

## Commits This Session

- HR spike settling time fix in StepB_PostProcess.py
- Overshoot filter added then reverted (comment left noting deferral)
- TODO.md updated

---

## Next Session Priorities

1. **Check A001 TSS** — verify parkrun TSS corrected after UPDATE
2. **Fix A001 workflow** — add `merge_user_data.py` step so training plan is picked up
3. **Johan A007 INITIAL** — deferred from previous sessions, ready to go
4. **HR overshoot filter** — own session, needs surgical approach (not ratio threshold)
