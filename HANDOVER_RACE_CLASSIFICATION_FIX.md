# Handover: Race Classification False Positive Fix
## Date: 2026-03-09

---

## Problem

Ian's INITIAL rebuild flagged 175 races from 3,089 runs. ~108 were false positives — training runs at club sessions, tempo runs, and interval workouts flagged by HR alone.

**Root cause:** Ian's LTHR (157) is correct (median 5K race HR is 162 = 103% LTHR), but his training HR frequently exceeds race-classification thresholds. With 95% LTHR = 149 bpm, almost every non-easy run becomes a candidate. HR simply cannot distinguish race from training for this athlete profile.

---

## Approach: No anti-keyword changes

Anti-keywords ("Afternoon Run", "Evening Run") were considered but rejected — athletes who don't name their races on Strava would have genuine unnamed parkruns blocked. Instead, three data-driven fixes that don't rely on activity names:

## Changes to classify_races.py

### Fix A: Bespoke candidates require race keyword
Non-standard distances now require a race keyword or name override in addition to HR. Eliminates 88 false positives with zero risk (bespoke races always have recognisable names — relay legs, XC, non-standard events).

### Fix B: Step 5 (unnamed runs) — higher evidence bar
When a run has NO keywords either way (no race keyword, no anti-keyword), two guards apply:

1. **HR uplift +4% LTHR:** Unnamed runs need HR >= threshold + 4% (e.g. 5K needs 102% instead of 98%). This catches training runs at standard distances while still detecting genuine unnamed race efforts with convincing HR.

2. **Pace check (ratio > 1.25):** If pace is more than 25% slower than predicted race pace for the distance, classified as training regardless of HR. Catches slow high-HR training (e.g. 35-min 5K by a 19:30 runner).

**Calibrated on Ian's data:** +4% uplift keeps 11/14 true unnamed races, eliminates 86/119 unnamed false positives. Pace check catches 9 more.

### Fix C: Race name overrides
Added: Paddock Wood, Chichester, Maidenhead, Beckenham Assembly, Middlesex 10k, Kent XC, Surrey league, Titsey.

---

## Impact

| | Before | After |
|---|---|---|
| Total races | 175 | ~62 |
| False positives | ~108 | ~11 |
| True races | ~47 | ~51 (gained from name overrides) |
| True races lost | 0 | ~3 (borderline 97-98% LTHR unnamed) |

Remaining ~11 false positives need manual overrides in activity_overrides.xlsx:
- "HM training run", "Sunday LSR with DRs", "LSR with Ebe" — long runs at HM distance
- "1600 @ hmp", "Track 5000 - 400 on 200 off" — sessions at standard distances
- "2nd half Tim's long marathon run" — training
- A few "Evening Run" entries at 5K with 102-108% LTHR

35 runs have no prediction data (N/A) — early history before the model has enough data. These rely entirely on HR and keywords.

---

## Analysis details

### Pace ratio distribution (140 races with predictions)
- Ratio <= 1.10: mix of true races and false positives — not separable
- Ratio 1.10-1.25: mostly false positives but some true XC/trail races
- Ratio > 1.25: all false positives (33 caught)
- Threshold 1.25 chosen to avoid any true race loss

### HR uplift calibration
- True unnamed races: mean 103% LTHR, range 97-108%
- False positive unnamed runs: mean 98% LTHR, range 95-108%
- +4% uplift: loses 3 true races at 97-98% (early data, low confidence)
- +3% uplift: only loses 2 but keeps 47 more FPs
- +5% uplift: loses 5 true races — too aggressive

### LTHR validation
- Ian's LTHR 157 confirmed correct from race data
- Median 5K race avg HR: 162 (103% LTHR)
- Recent (2023+) 5K avg: 159 (101% LTHR)
- Raising LTHR would be incorrect — the problem is HR overlap, not miscalibration

---

## Files delivered

| File | Action |
|---|---|
| classify_races.py | Replace in repo root |

## Next steps

1. Deploy classify_races.py
2. Run CLASSIFY_RACES mode for Ian
3. Review ~11 remaining false positives in overrides, set race_flag=0
4. Run CLASSIFY_RACES for Paul — verify no regression
5. Consider adding more UK race names to RACE_NAME_OVERRIDES as athletes are onboarded
