# Handover: Onboarding Script, Snow Analysis, Alert Rework
## Date: 2026-02-24

---

## Files modified

- **StepB_PostProcess.py** — Alert 1b rewritten (TSB projection replaces RFL-vs-peak)
- **generate_dashboard.py** — Past race filtering on race cards, alert name/detail updated
- **onboard_athlete.py** — NEW file, athlete onboarding automation
- **TODO.md** — Updated with new items

## Files NOT modified

- config.py, athlete.yml, athlete_config.py, rebuild_from_fit_zip.py, all workflows — unchanged

---

## 1. Onboarding script (onboard_athlete.py)

New script to automate new athlete setup. Generates all four files needed:

- `athletes/<Name>/athlete.yml` — full config
- `.github/workflows/<slug>_pipeline.yml` — complete CI workflow (intervals.icu or FIT-only variant)
- `athletes/<Name>/activity_overrides.xlsx` — with auto-detected races if FIT scan used
- `athletes/<Name>/athlete_data.csv` — empty with headers

**Three usage modes:**
- `python onboard_athlete.py` — interactive prompts
- `python onboard_athlete.py --config athlete.json` — from JSON
- `python onboard_athlete.py --scan-fits path/to/fits.zip` — scan FITs for HR zones, power meter detection, race auto-detection

**FIT scanning features:**
- Estimates max HR (95th percentile) and LTHR (from tempo efforts or Karvonen fallback)
- Detects Stryd vs Garmin power (CV analysis) → recommends GAP or Stryd mode
- Auto-detects races by pace/HR/distance heuristics with confidence levels
- Pre-populates override file with detected races

Tested with both FIT-folder and intervals.icu variants. YAML validates cleanly.

---

## 2. Snow surface adjustment analysis

Analysed 10 snow easy runs (Stockholm, Jan 25–Feb 17) vs 8 clear easy runs (London + Stockholm).

**Key findings:**

| Method | Snow penalty | Notes |
|---|---|---|
| Stryd/GAP ratio (energy cost) | ~0.7% | Stryd measures barely any extra energy |
| HR-matched pace comparison | ~1.0% median | Excluding post-flu run |
| RF_gap at matched HR | implied adj 0.999 | Essentially no adjustment needed for easy runs |
| Haga HM sandwich (moderate effort) | implied adj 1.078 | But confounded by cardiac drift + trail terrain |

**Conclusion:** The blanket 1.05 SNOW adjustment is too generous for packed/ploughed paths. Most snow runs were easy effort where the penalty is minimal. Paul will fine-tune individual overrides manually — he knows the conditions better than any formula. The studded Adizeros also narrow the gap in recent weeks.

**No code changes** — this was analysis only. Surface adjustment values remain in activity_overrides.xlsx for Paul to edit.

---

## 3. Alert 1b rewritten — TSB projection

### Old logic (removed)
- Compared current RFL to 90-day peak
- If gap > 2% within taper window → "Taper not working"
- Only fired for A/B races (C had taper_days=0)
- Problem: compared to irrelevant historical peaks, fired when taper was actually working fine

### New logic
- Projects TSB to race day using exponential decay of CTL (τ=42d) and ATL (τ=7d)
- Assumes light training (TSS ~20/day) during remaining taper
- Compares projected TSB to target range based on **race distance** and **priority**
- Fires for A, B, AND C races (with different target ranges and windows)

### TSB target ranges

| Priority | 5K/10K | HM | Marathon |
|---|---|---|---|
| A | +5 to +15 | +10 to +25 | +20 to +35 |
| B | 0 to +15 | +5 to +20 | +15 to +30 |
| C | -5 to +10 | 0 to +15 | +10 to +25 |

### Taper windows (check period)
- A: 14 days + 3 day early warning = 17 days before race
- B: 7 + 3 = 10 days
- C: 3 + 3 = 6 days

### Validation against current data

| Scenario | Projected TSB | Target | Alert? |
|---|---|---|---|
| Paul: LFOTM 5K (B, 3d) | +6.1 | 0 to +15 | No ✓ |
| Ian: Paddock Wood HM (A, 12d) | +14.0 | +10 to +25 | No ✓ |
| Ian: Burgess parkrun (C, 4d) | +2.8 | -5 to +10 | No ✓ |
| Hypothetical: A mara, TSB -50, 5d | -4.6 | +20 to +35 | Yes ✓ |
| Hypothetical: A HM, TSB -25, 3d | -6.5 | +10 to +25 | Yes ✓ |

### Dashboard changes
- Alert name: "Taper not working" → "Pre-race TSB concern"
- Alert level: concern (red) → watch (yellow ⏳)
- Detail text: shows projection, e.g. "TSB projected -6 for Paddock Wood HM in 3d (target +10 to +25)"
- Race readiness cards: now filter out past races (date < today)

### Config constants no longer used
- `ALERT1B_RFL_GAP` and `ALERT1B_PEAK_WINDOW_DAYS` are still in config/athlete.yml but are no longer referenced by the new alert logic. Can be removed in a future cleanup.

---

## 4. TODO updates

Added to TODO.md:
- **#5** Auto-detect Stryd power outliers (RE z-score < -2.5σ → fall back to GAP RF)
- **#6** Nadi weight history — Paul has the data, needs importing to athlete_data.csv
- **#7** Apply Feb 21 power overrides (340W parkrun, 240W preamble) — quick task

---

## For next Claude

"Session added onboard_athlete.py for automated new athlete setup (generates athlete.yml, workflow YAML, overrides xlsx, athlete_data.csv from FIT scan or JSON config). Alert 1b rewritten: now projects TSB to race day using CTL/ATL decay instead of comparing RFL to 90-day peak. Fires for A/B/C races with distance- and priority-specific TSB target ranges. Dashboard race readiness cards now filter out past races. See HANDOVER_ONBOARD_ALERTS.md."
