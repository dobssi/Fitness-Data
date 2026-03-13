# Handover: Johan A007 INITIAL — Polar FIT Debugging
## Date: 2026-03-11

---

## Summary

Johan (A007) INITIAL pipeline run from Polar JSON export. Three bugs fixed across four CI iterations before the rebuild started processing successfully. As of handover, the rebuild job is running through 1,439 FIT files.

---

## Bugs fixed this session

### 1. YAML parse error in `athletes/A007/athlete.yml`

**Symptom:** `yaml.scanner.ScannerError: could not find expected ':'` at line 36.

**Root cause:** `onboard_athlete.py` generated invalid YAML for empty planned_races:
```yaml
planned_races:
[]          # ← [] at column 1 looks like a new key, not a value
```

**Fix (two files):**
- `athletes/A007/athlete.yml` — changed to `planned_races: []` (inline). Also added Johan's Hannover Half Marathon (2026-04-12, priority A).
- `onboard_athlete.py` — two changes:
  - Template line: `planned_races:\n{races_yaml}` → `planned_races: {races_yaml}` (value on same line as key)
  - Populated case: added leading `\n` to `races_yaml` so block list items start on the next line after the key
  - Empty case: `races_yaml = "[]"` (unchanged, now works because it's inline)

**Other athletes:** Only A007 was affected. A005/A006 have populated planned_races, Nadi had `planned_races: []` inline already, Ian/Steve have races populated.

### 2. FIT base type constants wrong in `ci/polar_ingest.py`

**Symptom:** All 1,439 FIT files failed with `TypeError("'>=' not supported between instances of 'tuple' and 'int'")`.

**Root cause:** FIT base type constants were completely wrong:
```python
# WRONG (original)
FIT_UINT8 = 0    # actually enum
FIT_UINT16 = 2   # actually uint8
FIT_UINT32 = 4   # actually uint16
```
Every field in the binary FIT output was tagged with the wrong type. `fitparse` couldn't decode multi-byte values and returned raw bytes as tuples.

**Fix:** Corrected to FIT SDK values: `FIT_UINT8=2`, `FIT_UINT16=4`, `FIT_UINT32=6`, etc.

**This fix alone was insufficient** — see bug #3.

### 3. FIT base types need endianness flag for multi-byte fields

**Symptom:** Same `tuple >= int` error even after correcting base type numbers.

**Root cause (confirmed via full traceback):** `fitparse` 1.2.0's `process_type_date_time` processor received a tuple instead of an int for the timestamp field. The traceback:
```
fitparse/processors.py line 72, in process_type_date_time
    if value is not None and value >= 0x10000000:
TypeError: '>=' not supported between instances of 'tuple' and 'int'
```

In the FIT binary protocol, multi-byte field definitions encode an **endianness flag** (bit 7, 0x80) in the base type byte. Without this flag, `fitparse` doesn't recognise the field as multi-byte and returns raw bytes as a tuple instead of unpacking to an integer. Real Garmin/Polar FIT files use the flagged variants.

**Fix:** All multi-byte base types now use 0x80-flagged values:
```python
FIT_SINT16 = 0x83    # was 3
FIT_UINT16 = 0x84    # was 4
FIT_SINT32 = 0x85    # was 5
FIT_UINT32 = 0x86    # was 6
FIT_UINT32Z = 0x8C   # was 12
```
Single-byte types (enum=0x00, sint8=0x01, uint8=0x02, string=0x07) unchanged.

### Additional change: fixed record definition

During debugging (before the endian flag was identified as root cause), `add_record()` in `polar_ingest.py` was also refactored from dynamic per-record field redefinition to a **fixed definition written once** with FIT invalid sentinels for missing values. This is cleaner and avoids potential `fitparse` issues with 1000+ definition redefinitions per file. The change is kept — it produces slightly larger files but more compatible ones.

---

## Diagnostic change (can revert)

`rebuild_from_fit_zip.py` — added `traceback.print_exc()` for the first 3 failures in the main processing loop (line ~3563). This was essential for diagnosing the fitparse error. Harmless to leave in, or revert after confirming Johan's INITIAL completes.

---

## Current state

- **Johan INITIAL:** Rebuild job running. Polar ingest converts 1,439 running sessions (from 2,446 total, 947 non-running, 60 no per-sec data) to FIT files. Processing confirmed started after the endian flag fix.
- **Polar data profile:** 16 years (2010–2026). Products: Polar V800 (253), Vantage V (133), INW4A (186), INW5P (425), Flow app (18), unknown (424). 736 with power, 997 with GPS.
- **Power timeline:** Pre-2021 no power, 2021–2024 Polar wrist power (not Stryd), 2025-01+ Stryd. GAP mode is correct — ignores all power.
- **Activity names:** Sparse. Polar `name` field often empty. No Strava `activities.csv` for backfill yet.

---

## Files changed this session

| File | Change |
|------|--------|
| `athletes/A007/athlete.yml` | Fixed `planned_races` YAML syntax + added Hannover HM |
| `onboard_athlete.py` | Fixed `planned_races` template for empty case |
| `ci/polar_ingest.py` | Fixed FIT base types (endian flag) + fixed record definition |
| `rebuild_from_fit_zip.py` | Added traceback for first 3 failures (diagnostic, can revert) |

---

## What to expect when INITIAL completes

- Job 1 (rebuild): Processing 1,439 FIT files, weather fetch, persec NPZ cache build. Should produce `Master_A007_FULL.xlsx` and upload to Dropbox.
- Job 2 (stepb_deploy): StepB post-processing, classify_races, dashboard generation, GitHub Pages deploy.
- PEAK_CP will bootstrap from race data (seeded at 441W from weight-based estimate of 4.85 W/kg × 91kg).
- First dashboard at `dobssi.github.io/Fitness-Data` (A007 path).

---

## Post-INITIAL data quality: rogue Polar speed/distance data

### Problem

The INITIAL completed successfully (1,439 runs in master), but **RFL_gap_Trend is only 17%** — should be ~80-90%. The dashboard fitness curve is crushed.

### Root cause chain

1. **442 runs have no GPS** (mostly 2010-2014, pre-GPS Polar watch era). Of these, **226 have implausible session-level speed** (>20 km/h avg, some >60 km/h). The Polar JSON `distance` field for non-GPS sessions doesn't match elapsed time — likely accumulated stride sensor distance at a different scale.

2. **PS_gap uses session-level distance/time for speed** (StepB line ~4944): `avg_speed_mps = (distance_km * 1000) / moving_time_s`. For rogue runs, this gives 10-20 m/s → enormous GAP power → PS_gap values of 2000-5000 (normal: 200-400).

3. **PS floor overrides RF_gap_adj** (StepB line ~5275): `rf_gap_floor = PS_gap / ps_rf_divisor`. With PS_gap=4960 and divisor=184, the floor is ~27.0, overriding the sane RF_gap_median of 1.86.

4. **RF_gap_Trend peaks at 11.15** from the floor-inflated RF_gap_adj values in 2011. Current RFL = 1.9/11.15 = 17%.

### Key fact: per-second RF data is clean

The `RF_gap_median` values are sane even for rogue runs (median 1.77 vs 1.81 for GPS runs). The per-second Polar speed data appears correct — it's only the **session summary distance** that's wrong. The contamination flows through PS_gap (which uses session distance) → PS floor → RF_gap_adj → RF_gap_Trend peak.

### Fix options (for next session)

**Option A (quick, in StepB):** Cap `avg_speed_mps` for PS_gap calculation at a reasonable max (e.g., 6.5 m/s = 2:34/km). This would prevent rogue distances from inflating PS_gap. Affects all athletes but is a sensible sanity check.

**Option B (targeted, in StepB):** Use per-second median speed instead of `distance_km/moving_time_s` for PS_gap when the computed speed exceeds a threshold. The per-second data is clean.

**Option C (in polar_ingest.py):** Cap or validate speed/distance at ingest time. For non-GPS sessions where stride-sensor distance exceeds what's physically possible given elapsed time, mark distance as unreliable.

**Option D (in rebuild):** The existing v51 filter (line ~5226) skips runs with `avg_pace < 3:00/km AND distance > 5km`, but it uses `avg_pace_min_per_km` which is computed from per-second data (sane). The filter needs to also check session-level speed `distance_km / elapsed_time_s` to catch the Polar rogue cases.

**Recommended:** Option A or B — fix at PS_gap level in StepB. Quick and surgical.

### Other findings

- **Zero races classified** — `is_race=1: 0`. No PEAK_CP bootstrap. Johan has raced (Polar data should contain race efforts) but `classify_races.py` may need the session-level distance to be correct to identify race distances. Check after fixing the rogue data.
- **CTL at 180** — likely inflated by rogue TSS values from impossible speeds/distances. TSS depends on Power_Score which also uses the session-level speed. Fixing PS_gap will also fix TSS.
- **104 runs with no HR** — 18 from 2025 (possibly Stryd-only sessions without HR strap), rest scattered across 2010-2018. These are filtered from RF calculation (no HR = no RF).

---

## Outstanding TODOs (from this + prior sessions)

- **PRIORITY:** Refactor `onboard_athlete.py` workflow generation to use template file instead of f-strings (the planned_races bug is a symptom of the broader f-string fragility)
- Investigate 1.3% systematic RF_gap_median offset between A005/A006 (low priority)
- Add `stryd_mass_kg` to `athlete.yml` as separate field from `mass_kg` (low priority)
- Johan: Get Strava export for activity name backfill
- Johan: Confirm intervals.icu backsync from Strava is complete, then enable scheduled pipeline runs

---

## For next Claude

"Johan A007 INITIAL is running (Polar JSON → FIT conversion working after three rounds of FIT binary format fixes in polar_ingest.py). Check CI logs for completion. If rebuild succeeded, job 2 should auto-trigger for StepB + dashboard. The planned_races YAML bug in onboard_athlete.py is also fixed. See HANDOVER_JOHAN_INITIAL_DEBUGGING.md for full context."
