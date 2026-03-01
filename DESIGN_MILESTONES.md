# Milestone Achievements Feature — Design Spec
## Date: 2026-03-01

---

## Overview

Flag notable performances on the dashboard: hero banner for significant milestones, inline badges in Recent Runs for lighter ones. Computed at dashboard-generation time from the master Excel (no StepB changes needed).

Applies to any run with a valid `age_grade_pct`, including parkruns — not limited to `race_flag=1`.

---

## Placement

**Hero banner**: above Race Predictions section (near the AG chart).

**Recent Runs badges**: new column/inline marker in the existing Recent Runs table.

---

## What counts as a milestone

### Age grade milestones

Compare the run's `age_grade_pct` against all prior runs with valid AG (any distance).

| Tier | Condition | Display |
|------|-----------|---------|
| 🏆 Gold | All-time best AG (and 5+ prior AG runs exist) | Banner + badge |
| ⭐ Silver | Best AG in 3+ years | Banner + badge |
| ✨ Bronze | Best AG in 1+ year | Badge only |

### Distance AG milestones

Compare the run's `age_grade_pct` against prior runs at the same distance bucket AND surface category.

| Tier | Condition | Display |
|------|-----------|---------|
| 🏆 Gold | All-time distance+surface AG PB (and 1+ prior attempt) | Banner + badge |
| ⭐ Silver | Best AG at distance+surface in 3+ years | Banner + badge |
| ✨ Bronze | Best AG at distance+surface in 1+ year | Badge only |

### Distance time milestones

Compare the run's `elapsed_time_s` against prior runs at the same distance bucket AND surface category. A PB is a PB — clock time milestones are meaningful regardless of age.

| Tier | Condition | Display |
|------|-----------|---------|
| 🏆 Gold | All-time distance+surface time PB (and 1+ prior attempt) | Banner + badge |
| ⭐ Silver | Fastest at distance+surface in 3+ years | Banner + badge |
| ✨ Bronze | Fastest at distance+surface in 1+ year | Badge only |

### Condition-adjusted time milestones

Compare the run's condition-adjusted time against prior adjusted times at the same distance bucket AND surface category. Uses the same duration-scaled heat formula as the race prediction chart: `adj_time = elapsed_time_s / (temp_factor × surface_adj)`. This catches performances where the runner was in PB shape but conditions masked it — e.g. Steve's London 2018 marathon was 2:42 on the clock but 2:30 adjusted, far better than any other marathon.

| Tier | Condition | Display |
|------|-----------|---------|
| 🏆 Gold | All-time distance+surface adj-time PB (and 1+ prior attempt) | Banner + badge |
| ⭐ Silver | Best adj time at distance+surface in 3+ years | Banner + badge |
| ✨ Bronze | Best adj time at distance+surface in 1+ year | Badge only |

Only flagged when it differs from the clock-time milestone — if you set both a clock PB and an adjusted PB on the same run, the adjusted PB is redundant and suppressed. The adjusted milestone adds value when conditions prevented a clock PB but the underlying performance was exceptional.

### The 2-attempt rule

PBs require at least one prior attempt at the same distance+surface combo. Your first ever marathon can't be a PB — but your second can. This prevents trivially flagging debut performances while allowing genuinely meaningful early PBs (e.g. Steve's 2nd marathon London 2016 was a legitimate PB over London 2015).

For "best in N years" milestones (silver/bronze), the prior attempt threshold doesn't apply — the comparison is against the most recent better performance, which implies sufficient history.

### Surface categories

Runs are grouped into surface categories so that road PBs, trail PBs, and track PBs are tracked independently.

| Category | Surface values | Notes |
|----------|---------------|-------|
| road | NaN, empty, road (default) | Most parkruns, road races |
| trail | TRAIL, SNOW, HEAVY_SNOW | Off-road surfaces |
| track | TRACK | Outdoor track |
| indoor | INDOOR_TRACK | Indoor track |

When the surface category is not `road`, the milestone label includes it: "trail 5K PB", "indoor 3K AG PB", "track Mile time PB". Road is the default and unlabelled: just "5K PB", "HM AG PB".

### Distance buckets

Runs are grouped by standard distance for time comparisons:

| Bucket | Range |
|--------|-------|
| 1K | 0.8–1.2 km |
| Mile | 1.4–1.8 km |
| 3K | 2.7–3.3 km |
| 5K | 4.5–5.5 km |
| 10K | 9.2–10.8 km |
| HM | 19.6–22.6 km |
| Marathon | 39.7–44.7 km |

Runs outside these buckets get AG milestones only, not distance time milestones.

---

## Detection logic (Python, in generate_dashboard.py)

```python
def detect_milestones(df):
    """
    For each run with valid age_grade_pct, detect milestone achievements.
    Returns a list of dicts with milestone metadata per run index.
    
    Called once during dashboard generation. Iterates chronologically,
    comparing each run against all prior runs.
    """
    ag_runs = df[df['age_grade_pct'].notna() & (df['age_grade_pct'] > 0)].copy()
    ag_runs = ag_runs.sort_values('date')
    
    results = {}  # index → list of milestone dicts
    
    for idx, row in ag_runs.iterrows():
        dt = row['date']
        ag = row['age_grade_pct']
        t = row['elapsed_time_s']
        bucket = dist_bucket(row['distance_km'])
        
        prior = ag_runs[ag_runs['date'] < dt]
        milestones = []
        
        # --- AG milestone (any distance) ---
        prior_better_ag = prior[prior['age_grade_pct'] >= ag]
        if len(prior_better_ag) == 0 and len(prior) >= 5:
            milestones.append({
                'tier': 'gold', 'type': 'ag',
                'text': 'All-time best age grade',
            })
        elif len(prior_better_ag) > 0:
            last = prior_better_ag.sort_values('date').iloc[-1]
            days = (dt - last['date']).days
            if days >= 365:
                milestones.append({
                    'tier': 'gold' if days >= 5*365 else 
                            'silver' if days >= 3*365 else 'bronze',
                    'type': 'ag',
                    'text': f'Best age grade in {fmt_gap(days)}',
                    'prev_date': last['date'],
                    'prev_name': last['activity_name'],
                    'prev_ag': last['age_grade_pct'],
                    'prev_time': last['elapsed_time_s'],
                })
        
        # --- Distance AG milestone (same distance + surface) ---
        if bucket:
            same = prior[(prior['dist_bucket'] == bucket) & (prior['surface_cat'] == scat)]
            label = f'{bucket}' if scat == 'road' else f'{scat} {bucket}'
            
            better_ag_dist = same[same['age_grade_pct'] >= ag]
            if len(better_ag_dist) == 0 and len(same) >= 1:  # 2-attempt rule
                milestones.append({
                    'tier': 'gold', 'type': 'dist_ag',
                    'text': f'{label} AG PB',
                })
            elif len(better_ag_dist) > 0:
                last = better_ag_dist.sort_values('date').iloc[-1]
                days = (dt - last['date']).days
                if days >= 365:
                    milestones.append({
                        'tier': 'silver' if days >= 3*365 else 'bronze',
                        'type': 'dist_ag',
                        'text': f'Best {label} AG in {fmt_gap(days)}',
                        'prev_date': last['date'],
                        'prev_name': last['activity_name'],
                        'prev_time': last['elapsed_time_s'],
                        'prev_ag': last['age_grade_pct'],
                    })
            
            # --- Distance time milestone (same distance + surface) ---
            faster = same[same['elapsed_time_s'] <= t]
            has_time_pb = len(faster) == 0 and len(same) >= 1
            if has_time_pb:
                milestones.append({
                    'tier': 'gold', 'type': 'dist_time',
                    'text': f'{label} time PB',
                })
            elif len(faster) > 0:
                last = faster.sort_values('date').iloc[-1]
                days = (dt - last['date']).days
                if days >= 365:
                    milestones.append({
                        'tier': 'silver' if days >= 3*365 else 'bronze',
                        'type': 'dist_time',
                        'text': f'Fastest {label} in {fmt_gap(days)}',
                        'prev_date': last['date'],
                        'prev_name': last['activity_name'],
                        'prev_time': last['elapsed_time_s'],
                    })
            
            # --- Condition-adjusted time milestone (same distance + surface) ---
            # Only flag when it differs from the clock-time result
            # adj_time = elapsed / (temp_factor × surface_adj)
            adj_t = calc_adj_time(row)
            same_adj = same.copy()
            same_adj['_adj_time'] = same_adj.apply(calc_adj_time, axis=1)
            faster_adj = same_adj[same_adj['_adj_time'] <= adj_t]
            has_adj_pb = len(faster_adj) == 0 and len(same) >= 1
            if has_adj_pb and not has_time_pb:  # suppress if clock PB already flagged
                milestones.append({
                    'tier': 'gold', 'type': 'dist_adj_time',
                    'text': f'{label} adj-time PB',
                })
            elif not has_time_pb and len(faster_adj) > 0:
                last = faster_adj.sort_values('date').iloc[-1]
                days = (dt - last['date']).days
                # Only flag if different from clock-time milestone
                if days >= 365:
                    milestones.append({
                        'tier': 'silver' if days >= 3*365 else 'bronze',
                        'type': 'dist_adj_time',
                        'text': f'Best {label} adj time in {fmt_gap(days)}',
                        'prev_date': last['date'],
                        'prev_name': last['activity_name'],
                        'prev_time': last['elapsed_time_s'],
                    })
        
        if milestones:
            results[idx] = milestones
    
    return results
```

---

## Hero banner design

Shows the most recent banner-worthy milestone (gold or silver tier) from the last 28 days. If multiple milestones exist on the same run, the highest-tier AG milestone takes priority as the headline, with others shown as secondary lines.

### Layout (dark theme)

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  ⭐  Best age grade in 5y 8m                             │
│                                                          │
│  LFOTM Feb 2026                                          │
│  77.9% AG · 19:42 · 27 Feb                               │
│                                                          │
│  Previously: Jun 2020 · BMC 3km TT · 80.3% AG · 10:49   │
│                                                          │
│  ⏱  Also: Best 5K AG in 3y 9m · Fastest 5K in 3y+        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Styling

- Card background: `#1a1d27` with subtle accent border-left (4px)
- Gold border: `#f59e0b` (amber)
- Silver border: `#818cf8` (indigo, matches dashboard accent)
- Headline: 1.1rem, font-weight 600, white
- Race name: 0.95rem, `#e4e7ef`
- Stats line (AG · time · date): 0.85rem, `#8b90a0`
- "Previously" line: 0.8rem, `#6b7280`, italic
- "Also" secondary milestones: 0.8rem, `#8b90a0`

### Rules

- Banner expires after 28 days (no milestone → section hidden entirely, not empty)
- If two runs in the window both have banner-worthy milestones, show the most recent one
- If a run has both AG and time milestones, AG is the headline, time is the "Also" line
- Multiple "Also" items are allowed (e.g. "Best 5K AG in 5y+ · Fastest 5K in 3y+")
- Maximum one "Also" line (items joined with ·)

---

## Recent Runs badges

### Layout

Add a milestone indicator after the existing run row content:

```
27 Feb  LFOTM Feb 2026      5K  19:42  77.9%  🏆 Best AG 5y 8m
21 Feb  Battersea parkrun    5K  19:53  76.8%
17 Jan  Battersea parkrun    5K  19:49  77.4%  ⭐ Best AG 3y 8m
24 Jan  Vinthundsvintern 3K  3K  11:21  71.8%  ✨ indoor 3K AG PB
```

### Badge format

- Gold: `🏆` prefix, amber text `#f59e0b`
- Silver: `⭐` prefix, indigo text `#818cf8`
- Bronze: `✨` prefix, muted text `#8b90a0`
- Text is abbreviated: "Best AG 5y 8m" or "5K AG PB" or "5K time PB" or "Fastest 10K 2y+"
- If a run has both AG and time milestones at the same tier, show the AG one (it's rarer and more meaningful for masters runners)
- Surface prefix when not road: "trail 5K time PB", "indoor 3K AG PB"
- Only the highest-tier milestone shown per row (avoid clutter)
- On mobile: badge wraps to second line if needed

---

## Data flow

```
Master_FULL_post.xlsx
       │
       ▼
generate_dashboard.py
  detect_milestones(df)
       │
       ▼
  milestone_data dict  ──→  Hero banner HTML (if recent gold/silver)
       │
       ▼
  recent_runs list     ──→  Badge column in Recent Runs table
       │
       ▼
  index.html (self-contained, no external data)
```

No changes to StepB, config.py, athlete.yml, or pipeline steps. The milestone computation runs entirely within `generate_dashboard.py` at dashboard generation time. The master Excel is not modified.

---

## Expected frequency (from Paul's data)

| Tier | Occurrences per year | Notes |
|------|---------------------|-------|
| 🏆 Gold | ~0–1 | All-time AG PBs become rare over time |
| ⭐ Silver | ~3–4 | Best AG in 3y+, strong comeback performances |
| ✨ Bronze | ~5 | Best AG in 1y+, seasonal progressions |
| **Banner-worthy** | **~3/year** | Gold + Silver = ~1 every 3–4 months |

These rates validated against Paul's 2024–2026 data (60 AG runs, 19 milestones, 6 banner-worthy over 2 years).

---

## Edge cases

- **Why AG not clock time**: For masters runners, clock-time PBs become impossible with age. AG already accounts for ageing — a 77.9% AG at age 56 running 19:42 is equivalent quality to 81.1% AG at age 49 running 17:52. However, time PBs are also tracked because a PB is a PB: Steve running 2:42 at London was a marathon time PB and that matters regardless of age grading.
- **2-attempt rule**: PBs (gold tier, both time and AG) require at least 1 prior attempt at the same distance+surface combo. Your first ever marathon can't be a PB, but your second can. This prevents trivially flagging debut performances.
- **Surface tracking**: Road, trail, track, and indoor are tracked independently. A road 5K PB is separate from a trail 5K PB. Most runs default to "road" when surface is unset. Athletes can set surface in activity_overrides.xlsx.
- **First few runs**: Overall AG milestones require 5+ prior AG runs. Distance PBs require 1+ prior attempt at same distance+surface.
- **Simultaneous time + AG PB**: Common when improving (Steve's London 2016–2018 marathons). Both are flagged; AG takes the headline in the banner, time PB appears in the "Also" line.
- **Virtual races**: Included normally (they have valid AG). The run name in the "previously" reference makes it clear if the comparison target was virtual.
- **Indoor/treadmill**: If AG is calculated (has valid distance + time), included. No special treatment needed.
- **Multiple runs same day**: Each compared independently against prior runs (not same-day peers).
- **Missing AG**: Runs without `age_grade_pct` are invisible to the system. No milestones, not counted as "prior runs" for others.
- **Distance bucket mismatch**: A 4.8km "5K" still falls in the 5K bucket (4.5–5.5km range). A 6km run gets AG milestones only.

---

## Implementation estimate

One session. Changes are confined to `generate_dashboard.py`:

1. Add `detect_milestones()` function with all 4 milestone types (~80 lines)
2. Add `calc_adj_time()` helper using duration-scaled Temp_Adj + surface_adj (~15 lines)
3. Add hero banner HTML generation (~40 lines)
4. Modify `get_recent_runs()` to include milestone badge data (~10 lines)
5. Add banner CSS styles (~20 lines)
6. Modify Recent Runs table template to show badges (~10 lines)

Total: ~175 lines of new code.
