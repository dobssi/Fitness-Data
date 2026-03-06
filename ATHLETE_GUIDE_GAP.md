# Understanding Your Dashboard
### GAP Mode (no power meter)

A guide to reading the metrics on your running dashboard.

---

## The Stats Grid

The top of your dashboard shows 16 cards in four rows. Hover over any card for a description.

### Row 1: Training Load

**CTL (Chronic Training Load)** — Your rolling 42-day training load, representing your underlying fitness. Built gradually through consistent training over weeks and months. Higher CTL means you've been training more and harder. Responds slowly — a single big week won't move it much, but steady training accumulates.

**ATL (Acute Training Load)** — Your rolling 7-day training load, representing recent fatigue. Spikes quickly after hard training blocks and drops with rest. When ATL is much higher than CTL, you're accumulating fatigue faster than fitness.

**TSB (Training Stress Balance)** — The difference between fitness and fatigue (CTL minus ATL). When positive, you're fresher than your recent training would suggest — good for racing. When negative, you're carrying fatigue. For race day, aim for TSB between 5–25% of your CTL (the exact target depends on race distance and priority — your Race Readiness cards show the specific range).

**Weight** — Your 7-day smoothed weight from the most recent data available. Sourced from intervals.icu (synced from your watch) or manual entries.

### Row 2: Fitness & Performance

**RFL Trend (Relative Fitness Level)** — Your current running fitness as a percentage of your all-time peak. 100% means you're in the best shape you've ever been. Calculated from the ratio of your Grade Adjusted Pace to heart rate, smoothed over 42 days. Grade adjustment means hills don't distort the picture — a hilly run at the same effort as a flat run produces the same fitness reading. This is the single most important number on your dashboard.

**RFL 14d** — How your RFL has changed over the last two weeks. Positive means fitness is trending up. Negative means it's declining. Useful for spotting whether your training is working or whether you need to adjust.

**Easy RF Gap** — How your easy runs compare to the overall fitness trend. If this goes negative, your easy runs feel harder than expected relative to your fitness level. Can be an early warning sign of fatigue, illness, or overtraining — your body is struggling on runs that should feel comfortable.

**Age Grade** — Your estimated 5K performance compared to the world record for your age and sex, using World Masters Athletics tables. As a rough guide: 50% is recreational, 60%+ is good club level, 70%+ is competitive, 80%+ is national standard.

### Row 3: Race Predictions

Predicted finish times for 5K, 10K, half marathon, and marathon. Based on your current RFL trend and a peak performance level bootstrapped from your race results. The more races you run, the better calibrated these become. Predictions update with every run as your fitness changes. They assume reasonably flat road conditions — your Race Readiness cards adjust for specific course surfaces.

### Row 4: Volume

Running distance totals for the last 7 days, 30 days, 12 months, and your entire training history.

---

## RFL Chart

The main chart shows your Relative Fitness Level over time. Toggle between 90-day, 180-day, 1-year, 2-year, 3-year, and 5-year views.

- **Purple line** — RFL Trend (42-day smoothed fitness)
- **Dots** — Individual run RFL values (single-run fitness snapshots, from GAP/HR ratio)
- **Green dashed line** — Easy RF trend (how your easy runs are tracking)
- **Race markers** — Triangles showing where you raced

The RFL trend rising means you're getting fitter. Plateaus are normal during maintenance phases. Dips happen during rest periods or illness — they're expected before races (taper) and usually recover.

---

## Training Load Chart

Shows CTL (fitness, blue), ATL (fatigue, orange), and TSB (form, green area) over time.

The classic pattern for race preparation: build CTL through consistent training, then reduce ATL through a taper, creating positive TSB (freshness) on race day while retaining the fitness you built.

---

## Training Zones

Three views of how you spend your training time:

**HR Zones** — Time in each heart rate zone per week. Most of your running should be in Z1–Z2 (easy/aerobic). Z3+ is tempo and harder work.

**Race Effort** — Time spent at race-specific intensities (Sub-5K, 5K, 10K, HM, Marathon pace). Uses pace zones derived from your predicted race paces. Shows whether your training is targeting the right effort bands for your upcoming race.

---

## Race Readiness

Cards for each of your planned races showing:

- **Target pace** — The pace needed for your predicted time
- **Predicted time** — Based on your current fitness, adjusted for the course surface
- **Effort minutes (14d / 42d)** — Time spent training at race-specific pace recently. More is better for race preparation.
- **Long run minutes (14d / 42d)** — Time spent on runs longer than 60 minutes. Important for half marathon and marathon preparation.
- **Z3+ tail minutes** — Time spent at tempo or harder during long runs. Quality long runs, not just volume.
- **TSB target** — Where your Training Stress Balance should be on race morning, shown as a percentage range of your CTL
- **Taper plan** — Suggested training for race week to hit your TSB target

---

## Race History

Select any two past races to compare side by side. Each card shows:

- **Performance** — Time, pace, heart rate, normalised age grade
- **Training context** — CTL, ATL, TSB, and TSS on the morning of the race
- **Preparation** — RFL coming into the race, effort minutes at race-specific intensity, long run volume
- **Conditions** — Temperature, terrain difficulty, surface

The delta panel highlights differences between the two races, helping you understand what contributed to better or worse performances.

---

## Milestones

**Recent Achievements** — Personal bests and notable performances from the last 60 days. Includes time PBs, age grade PBs, fitness milestones, and surface-specific records.

**All-Time** — Your complete record, organised by PBs, Volume, and Fitness tabs. Shows progressive time PBs per distance and surface, plus progress bars toward your next milestone.

---

## Recent Runs

A table of your latest activities showing date, name, distance, time, pace, heart rate, and training stress. Races are highlighted.

## Top Race Performances

Your best race results by normalised age grade, with times adjusted for conditions. This table shows your strongest performances relative to your age, regardless of distance.

---

## Alerts

The banner at the top of the dashboard shows health check alerts when something needs attention:

- **Overreaching** — RFL dropping while training load increases
- **Fatigue accumulation** — Multiple consecutive days with very negative TSB
- **Easy runs feeling hard** — Easy RF gap significantly below trend
- **Fitness plateau** — RFL stuck below recent peak for an extended period

Green "All clear" means everything looks normal.
