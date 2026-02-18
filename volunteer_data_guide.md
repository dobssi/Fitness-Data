# Getting set up on the running dashboard

Hi! Thanks for volunteering to test the running fitness dashboard. Here's what I need from you to get things running.

---

## What you'll need to provide

### Essential (can't start without these)

1. **Your running data** — a bulk export of your FIT files from Garmin Connect or Strava
   - **Garmin Connect**: go to [garmin.com/account](https://www.garmin.com/account), request a data export, and send me the zip when it arrives (usually takes a few hours)
   - **Strava**: go to [strava.com/athlete/delete_your_account](https://www.strava.com/athlete/delete_your_account) — don't worry, you're not deleting anything! There's a "Request Your Archive" button at the top of that page. Send me the zip when it arrives.

2. **A few personal details** (I'll set these up in your config file):
   - Your **weight** in kg (a rough constant is fine — we're not tracking daily fluctuations)
   - Your **date of birth** (for age-graded performance calculations)
   - Your **gender** (male/female — used for age grading tables)

3. **Your heart rate numbers**:
   - **Maximum heart rate** — if you know it from a test or a hard race, great. If not, I can estimate it from your data.
   - **Lactate threshold heart rate (LTHR)** — if you know it. If not, I'll estimate it as roughly 93% of your max HR, which works well enough to start.

### Helpful but not essential

4. **A recent race result** — any standard distance (5K, 10K, half marathon, marathon) with the date and your finishing time. This helps calibrate predictions much faster. Without it, the system will auto-detect your races from the data — it just takes a few weeks of data to converge.

5. **Upcoming races** — if you have any planned, let me know the date, distance, and how important it is to you (goal race vs just-for-fun). The dashboard can show countdown cards with predicted times and target paces.

### Not needed

- No power meter required (the system works from pace and heart rate)
- No special apps or accounts needed on your end
- No ongoing effort from you — once I have the initial export, I can set up a sync

---

## What you'll get back

A personal fitness dashboard showing:

- **Fitness trend** — how your running fitness is changing over time (think: a personal form curve)
- **Training load** — fitness, fatigue, and freshness balance (CTL/ATL/TSB)
- **Race predictions** — estimated 5K, 10K, half marathon, and marathon times based on your current fitness
- **Age-graded performance** — how your results compare on an age-adjusted scale
- **Training zones** — heart rate and pace zones tailored to your current fitness
- **Race readiness** — for any upcoming races, how prepared you are and what pace to target

The dashboard updates automatically and works on your phone.

---

## Privacy

Your data stays on my system — it's not uploaded to any third-party service. The dashboard is a static HTML file that can be hosted privately or shared however you like. I won't share your data with anyone else.

---

## Questions?

Just ask. The whole point of this is to make something useful for you — if there's something you'd want to see on the dashboard, let me know.
