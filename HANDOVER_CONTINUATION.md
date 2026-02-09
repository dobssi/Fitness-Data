# Handover: Phase 1 Integration in Progress
## Date: 2026-02-09
## Status: Testing Phase 1 (Athlete Config via YAML)

---

## Where We Are

**Completed:**
- ✅ Built Phase 1 + 2 implementation (athlete config + GAP mode)
- ✅ Delivered phase_1_2_delivery.zip with 10 files
- ✅ All 20 tests passing in isolation
- ✅ Paul unzipped files into pipeline root
- ✅ config.py replaced with new version

**Currently Testing:**
- Phase 1 integration (YAML config)
- About to run: `.\StepB_PostProcess.bat FULL`
- Testing backward compatibility with v51

**athlete.yml status:**
- Currently renamed to `athlete.yml.template` (moved aside)
- New config.py should use v51 defaults (no YAML)
- Once StepB works, will restore athlete.yml and test with YAML config

---

## What Phase 1 Does

**Before (v51):**
```python
# config.py - hardcoded constants
ATHLETE_MASS_KG = 76.0
PEAK_CP_WATTS = 372
# etc...
```

**After (Phase 1):**
```python
# config.py - loads from athlete.yml or falls back to v51
if _ATHLETE_CONFIG:
    ATHLETE_MASS_KG = _ATHLETE_CONFIG.mass_kg  # From athlete.yml
else:
    ATHLETE_MASS_KG = 76.0  # v51 default
```

All other scripts unchanged - they just import from config.py as before.

---

## Integration Progress

### Completed Steps:

1. ✅ **Backup config.py:** `cp config.py config.py.v51.backup`
2. ✅ **Replace config.py:** New version from phase_1_2_delivery.zip
3. ✅ **Install PyYAML:** `pip install pyyaml`
4. ✅ **Test backward compatibility:** `python test_phase_1_2.py` (all passed)

### Current Step:

5. ⏳ **Test existing pipeline:** `.\StepB_PostProcess.bat FULL`

   **Expected:** Should work identically to v51 since no athlete.yml exists yet
   
   **What it tests:**
   - New config.py loads v51 defaults correctly
   - All imports work (ATHLETE_MASS_KG, PEAK_CP_WATTS, etc.)
   - StepB_PostProcess.py runs without errors
   - Output matches v51 behavior

### Remaining Steps (After StepB Works):

6. **Restore athlete.yml:** `mv athlete.yml.template athlete.yml`
7. **Edit athlete.yml:** Fill in Paul's actual values
8. **Test with YAML:** Run StepB again, verify identical output
9. **Verify config loading:** `python athlete_config.py athlete.yml`
10. **Complete Phase 1** ✅

---

## File Locations

**Paul's pipeline root:**
```
C:\Users\paul.collyer\Dropbox\Running and Cycling\DataPipeline\
├── athlete.yml.template          # Moved aside for now
├── athlete_config.py             # New: config loader
├── config.py                     # New: YAML-aware version
├── config.py.v51.backup          # Backup of original
├── gap_power.py                  # New: Phase 2 (not using yet)
├── add_gap_power.py              # New: Phase 2 (not using yet)
├── run_multi_mode_pipeline.py   # New: Phase 2 (not using yet)
├── test_phase_1_2.py             # New: test suite
├── StepB_PostProcess.py          # Unchanged (v51)
├── StepB_PostProcess.bat         # Unchanged (v51)
└── ... (all other v51 files)
```

---

## StepB Batch File Details

Paul's `StepB_PostProcess.bat FULL` runs:

```batch
python StepB_PostProcess.py ^
  --master "Master_FULL_GPSQ_ID_simS4.xlsx" ^
  --persec-cache-dir "persec_cache_FULL" ^
  --model-json "re_model_s4_FULL.json" ^
  --out "Master_FULL_GPSQ_ID_post.xlsx" ^
  --mass-kg 76 ^
  --runner-dob 1969-05-27 ^
  --runner-gender male ^
  (... other args)
```

**Key point:** Even though batch file passes `--mass-kg 76`, the new config.py should load this from YAML (or v51 defaults if no YAML). The command-line args may be ignored or used as fallback depending on how StepB is coded.

---

## Expected Outcomes

### If StepB Works:

✅ **Phase 1 backward compatibility confirmed**
- New config.py working with v51 defaults
- All imports functional
- Ready to add athlete.yml

**Next:** Restore athlete.yml, test with YAML config

### If StepB Fails:

Possible issues:
1. **Import error:** Missing module or typo in config.py
2. **Constant missing:** StepB expects a constant that's not in new config.py
3. **Type mismatch:** YAML value has wrong type (string vs int, etc.)

**Solution:** Debug the specific error, fix, re-test

---

## Phase 2 (GAP Mode) - Not Active Yet

Phase 2 files are present but **not being used** yet:
- gap_power.py
- add_gap_power.py  
- run_multi_mode_pipeline.py

These are for GAP/HR mode (power simulation without Stryd). Will test later if Paul wants to explore it.

**Current focus:** Get Phase 1 (YAML config) working in Stryd mode.

---

## athlete.yml Content (When Restored)

```yaml
athlete:
  name: "Paul"
  mass_kg: 76.0
  date_of_birth: "1969-05-27"
  gender: "male"

power:
  mode: "stryd"
  
  stryd:
    peak_cp_watts: 372
    eras:
      pre_stryd: "1900-01-01"
      v1: "2017-05-05"
      repl: "2017-09-12"
      air: "2019-09-07"
      s4: "2023-01-03"
      s5: "2025-12-17"
    
    mass_corrections:
      - start_date: "2017-07-08"
        end_date: "2017-09-10"
        configured_kg: 77.0
      - start_date: "2017-09-11"
        end_date: "2019-09-06"
        configured_kg: 79.0
    
    re_reference_era: "s4"

data:
  source: "intervals"
```

---

## Quick Reference: Config Loading Priority

1. **athlete.yml exists:** Load all values from YAML
2. **No athlete.yml:** Use v51 hardcoded defaults
3. **Environment variable ATHLETE_CONFIG_PATH:** Use specified YAML path

Current state: #2 (no athlete.yml, using v51 defaults)

---

## Rollback Plan (If Needed)

If something breaks and Paul needs to revert:

```powershell
# Restore v51 config.py
cp config.py.v51.backup config.py

# Remove new files
rm athlete_config.py
rm gap_power.py
rm add_gap_power.py
rm run_multi_mode_pipeline.py

# Back to v51 - nothing broken
```

---

## Next Window Action Items

1. **Check StepB result** - Did it run successfully?
2. **If yes:** Proceed with steps 6-10 (restore athlete.yml, test YAML config)
3. **If no:** Debug error, fix, re-test
4. **Once Phase 1 working:** Decide on Phase 2 (GAP mode testing) or Phase 3 (multi-athlete)

---

## Files in Checkpoint

**phase1_integration_checkpoint.zip** contains:
- All Phase 1+2 implementation files (10 files)
- Same as phase_1_2_delivery.zip
- Includes this handover document

---

## Context for Next Session

**Paul is integrating Phase 1 (YAML config) into his v51 pipeline.**

- Goal: Externalize athlete-specific values to athlete.yml
- Method: Replace config.py, maintain backward compatibility
- Status: Testing batch file to verify v51 defaults work
- Next: If successful, enable YAML config and verify identical output

**Phase 2 (GAP mode) is built but not active yet.**

---

## Key Reminders

- **No breaking changes** - v51 pipeline should work identically
- **athlete.yml currently moved aside** - using v51 defaults
- **All other scripts unchanged** - only config.py replaced
- **Test suite passed** - 20/20 tests in isolation
- **Next test:** Real pipeline integration with StepB batch file

---

**Waiting on StepB test result before proceeding to athlete.yml restoration.**
