"""
Milestones Feature for Running Dashboard
=========================================
Computes and displays running milestones/achievements from the master data.

Two distinct sections:
A) RECENT ACHIEVEMENTS — "best X in Y years" for the last 30 days. What matters NOW.
B) ALL-TIME MILESTONES — cumulative distance, run counts, PBs, fitness peaks, consistency.

Milestone categories:
1. VOLUME: Distance totals (1K, 5K, 10K, 15K, 20K, 25K km), run counts (100, 500, 1000, 2000, 3000)
2. RACE PBs: Progressive personal bests per race distance
3. FITNESS: RFL_Trend peaks, first time above thresholds
4. CONSISTENCY: Consecutive day streaks, consecutive weeks with N+ runs
5. YEARLY: First year at 1000km, 1500km, 2000km, 2500km, 3000km
6. AGE GRADE: First 70%, 75%, 80%+ AG performance

Each milestone has: date, category, title, description, icon, and an "importance" score (1-3).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import json


def get_recent_achievements(df, lookback_days=30):
    """
    Scan recent runs/races for "best X in Y" achievements.
    
    Looks at races in the last `lookback_days` and checks if they are the best
    in various lookback windows (1y, 2y, 3y, 5y, all-time) for:
    - Age grade percentage
    - Race time per distance
    - RFL_Trend peak
    
    Returns list of achievement dicts sorted by significance.
    """
    df = df.sort_values('date').reset_index(drop=True)
    if len(df) == 0:
        return []
    
    achievements = []
    cutoff = df['date'].max() - pd.Timedelta(days=lookback_days)
    races = df[df['race_flag'] == 1].copy()
    
    lookback_windows = [
        (1, '1 year'),
        (2, '2 years'),
        (3, '3 years'),
        (5, '5 years'),
        (10, '10 years'),
    ]
    
    # --- AGE GRADE achievements ---
    recent_races = races[races['date'] >= cutoff]
    for _, r in recent_races.iterrows():
        ag = r.get('age_grade_pct', None)
        if pd.isna(ag):
            continue
        
        best_window = None
        for years_back, label in lookback_windows:
            lookback_start = r['date'] - pd.Timedelta(days=365 * years_back)
            prev = races[
                (races['date'] >= lookback_start) & 
                (races['date'] < r['date']) & 
                races['age_grade_pct'].notna()
            ]
            if len(prev) > 0 and ag >= prev['age_grade_pct'].max():
                best_window = (years_back, label)
            else:
                break  # No point checking longer windows
        
        # Also check all-time
        all_prev = races[(races['date'] < r['date']) & races['age_grade_pct'].notna()]
        is_all_time = len(all_prev) == 0 or ag >= all_prev['age_grade_pct'].max()
        
        if is_all_time:
            achievements.append({
                'date': r['date'].strftime('%Y-%m-%d'),
                'type': 'age_grade',
                'title': f'Best Ever Age Grade: {ag:.1f}%',
                'description': r.get('activity_name', ''),
                'icon': '👑',
                'significance': 5,  # All-time best
                'window': 'all-time',
            })
        elif best_window and best_window[0] >= 2:
            achievements.append({
                'date': r['date'].strftime('%Y-%m-%d'),
                'type': 'age_grade',
                'title': f'Best Age Grade in {best_window[1]}: {ag:.1f}%',
                'description': r.get('activity_name', ''),
                'icon': '🎖️',
                'significance': min(best_window[0], 4),
                'window': best_window[1],
            })
    
    # --- RACE TIME achievements per distance ---
    dist_names = {
        5.0: '5K', 10.0: '10K', 21.097: 'Half Marathon', 42.195: 'Marathon',
        3.0: '3K', 1.609: 'Mile', 1.0: 'Mile',
    }
    
    for dist in [5.0, 10.0, 21.097, 42.195, 3.0]:
        dist_races = races[races['official_distance_km'] == dist].sort_values('date')
        recent_dist = dist_races[dist_races['date'] >= cutoff]
        
        for _, r in recent_dist.iterrows():
            t = r['elapsed_time_s']
            
            best_window = None
            for years_back, label in lookback_windows:
                lookback_start = r['date'] - pd.Timedelta(days=365 * years_back)
                prev = dist_races[
                    (dist_races['date'] >= lookback_start) & 
                    (dist_races['date'] < r['date'])
                ]
                if len(prev) > 0 and t <= prev['elapsed_time_s'].min():
                    best_window = (years_back, label)
                else:
                    break
            
            # All-time PB?
            all_prev = dist_races[dist_races['date'] < r['date']]
            is_pb = len(all_prev) == 0 or t <= all_prev['elapsed_time_s'].min()
            
            dist_name = dist_names.get(dist, f'{dist}km')
            hrs = int(t // 3600)
            mins = int((t % 3600) // 60)
            secs = int(t % 60)
            time_str = f'{hrs}:{mins:02d}:{secs:02d}' if hrs > 0 else f'{mins}:{secs:02d}'
            
            if is_pb and len(all_prev) > 0:
                achievements.append({
                    'date': r['date'].strftime('%Y-%m-%d'),
                    'type': 'race_pb',
                    'title': f'{dist_name} PB: {time_str}',
                    'description': r.get('activity_name', ''),
                    'icon': '🏅',
                    'significance': 5,
                    'window': 'all-time',
                })
            elif best_window and best_window[0] >= 2:
                achievements.append({
                    'date': r['date'].strftime('%Y-%m-%d'),
                    'type': 'race_time',
                    'title': f'Fastest {dist_name} in {best_window[1]}: {time_str}',
                    'description': r.get('activity_name', ''),
                    'icon': '⚡',
                    'significance': min(best_window[0], 4),
                    'window': best_window[1],
                })
    
    # --- RFL TREND achievements ---
    recent_all = df[df['date'] >= cutoff]
    if len(recent_all) > 0 and 'RFL_Trend' in df.columns:
        rfl_recent = recent_all[['date', 'RFL_Trend']].dropna()
        if len(rfl_recent) > 0:
            peak_row = rfl_recent.loc[rfl_recent['RFL_Trend'].idxmax()]
            peak_val = peak_row['RFL_Trend']
            peak_date = peak_row['date']
            
            best_window = None
            for months_back, label in [(3, '3 months'), (6, '6 months'), (12, '1 year'), (24, '2 years')]:
                lookback_start = peak_date - pd.Timedelta(days=30 * months_back)
                prev = df[
                    (df['date'] >= lookback_start) & 
                    (df['date'] < peak_date)
                ]['RFL_Trend'].dropna()
                if len(prev) > 0 and peak_val >= prev.max():
                    best_window = (months_back, label)
                else:
                    break
            
            if best_window and best_window[0] >= 6:
                achievements.append({
                    'date': peak_date.strftime('%Y-%m-%d'),
                    'type': 'fitness',
                    'title': f'Highest Fitness in {best_window[1]}: {peak_val:.1%}',
                    'description': 'RFL Trend peak',
                    'icon': '📈',
                    'significance': 2 if best_window[0] >= 12 else 1,
                    'window': best_window[1],
                })
    
    # Sort by significance (highest first), then date (newest first)
    achievements.sort(key=lambda a: (-a['significance'], a['date']), reverse=False)
    achievements.sort(key=lambda a: -a['significance'])
    
    return achievements


def get_milestones_data(df, cutoff_date=None):
    """
    Compute all milestones from master dataframe.
    
    Parameters:
        df: Master dataframe with columns: date, distance_km, race_flag, 
            official_distance_km, elapsed_time_s, RFL_Trend, age_grade_pct, etc.
        cutoff_date: Optional date to filter data up to (for testing). 
                     If None, uses all data.
    
    Returns:
        dict with:
            'milestones': list of milestone dicts sorted by date
            'next_milestones': list of upcoming milestones (not yet achieved)
            'summary': dict with totals
    """
    df = df.sort_values('date').reset_index(drop=True)
    
    if cutoff_date is not None:
        if isinstance(cutoff_date, str):
            cutoff_date = pd.Timestamp(cutoff_date)
        df = df[df['date'] <= cutoff_date].copy()
    
    if len(df) == 0:
        return {'milestones': [], 'next_milestones': [], 'summary': {}}
    
    milestones = []
    
    # ========== 1. CUMULATIVE DISTANCE MILESTONES ==========
    cum_dist = df['distance_km'].cumsum()
    distance_thresholds = [
        (1000,  '1,000 km',  '🌍', 2),
        (2000,  '2,000 km',  '🌍', 1),
        (5000,  '5,000 km',  '🌏', 2),
        (10000, '10,000 km', '🌎', 3),
        (15000, '15,000 km', '🌐', 2),
        (20000, '20,000 km', '🛤️', 3),
        (25000, '25,000 km', '🏔️', 3),
        (30000, '30,000 km', '🚀', 3),
    ]
    for threshold, label, icon, importance in distance_thresholds:
        idx = cum_dist[cum_dist >= threshold].index
        if len(idx) > 0:
            row = df.loc[idx[0]]
            run_num = idx[0] + 1
            milestones.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'category': 'distance',
                'title': f'{label} Total Distance',
                'description': f'Reached {label} on run #{run_num:,}',
                'icon': icon,
                'importance': importance,
                'value': threshold,
            })
    
    # ========== 2. RUN COUNT MILESTONES ==========
    run_thresholds = [
        (100,  '💯', 2),
        (250,  '📊', 1),
        (500,  '📊', 2),
        (1000, '🔥', 3),
        (1500, '🔥', 2),
        (2000, '⚡', 3),
        (2500, '⚡', 2),
        (3000, '💎', 3),
    ]
    for threshold, icon, importance in run_thresholds:
        if threshold <= len(df):
            row = df.iloc[threshold - 1]
            milestones.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'category': 'runs',
                'title': f'Run #{threshold:,}',
                'description': f'{row["distance_km"]:.1f}km — {row.get("activity_name", "")}',
                'icon': icon,
                'importance': importance,
                'value': threshold,
            })
    
    # ========== 3. RACE PBs (Progressive Personal Bests) ==========
    races = df[df['race_flag'] == 1].copy()
    
    # Map distances to display names
    dist_names = {
        1.0: 'Mile', 1.5: '1500m', 1.609: 'Mile', 3.0: '3K',
        5.0: '5K', 8.9: '8.9K', 10.0: '10K', 15.0: '15K',
        16.09: '10M', 21.097: 'Half Marathon', 30.25: '30K',
        42.195: 'Marathon'
    }
    
    # Key distances for PB tracking
    pb_distances = [5.0, 10.0, 21.097, 42.195]
    
    for dist in pb_distances:
        dist_races = races[races['official_distance_km'] == dist].sort_values('date')
        if len(dist_races) == 0:
            continue
        
        running_best = dist_races['elapsed_time_s'].cummin()
        pb_mask = dist_races['elapsed_time_s'] == running_best
        # Also need to check it's actually a new PB (not same as previous)
        pb_runs = dist_races[pb_mask].copy()
        
        for i, (idx, row) in enumerate(pb_runs.iterrows()):
            t = row['elapsed_time_s']
            hrs = int(t // 3600)
            mins = int((t % 3600) // 60)
            secs = int(t % 60)
            if hrs > 0:
                time_str = f'{hrs}:{mins:02d}:{secs:02d}'
            else:
                time_str = f'{mins}:{secs:02d}'
            
            dist_name = dist_names.get(dist, f'{dist}km')
            is_first = (i == 0)
            is_latest = (i == len(pb_runs) - 1)
            
            # Importance: first race = 1, intermediate PBs = 1, current PB = 3
            importance = 3 if is_latest else (2 if not is_first else 1)
            
            # For first race at distance, it's still a PB but less exciting
            if is_first and len(pb_runs) > 1:
                importance = 1
            
            name = row.get('activity_name', '')
            # Truncate long names
            if isinstance(name, str) and len(name) > 50:
                name = name[:47] + '...'
            
            milestones.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'category': 'pb',
                'title': f'{dist_name} PB: {time_str}',
                'description': name if isinstance(name, str) else '',
                'icon': '🏅' if is_latest else '🎯',
                'importance': importance,
                'value': t,
                'distance_km': dist,
                'is_current_pb': is_latest,
            })
    
    # ========== 4. FITNESS MILESTONES (RFL_Trend) ==========
    rfl = df[['date', 'RFL_Trend']].dropna(subset=['RFL_Trend'])
    if len(rfl) > 0:
        # Peak RFL_Trend all-time
        peak_idx = rfl['RFL_Trend'].idxmax()
        peak_row = df.loc[peak_idx]
        milestones.append({
            'date': peak_row['date'].strftime('%Y-%m-%d'),
            'category': 'fitness',
            'title': f'Peak Fitness: {peak_row["RFL_Trend"]:.1%}',
            'description': 'All-time highest Running Fitness Level',
            'icon': '⭐',
            'importance': 3,
            'value': peak_row['RFL_Trend'],
        })
        
        # First time crossing thresholds
        rfl_thresholds = [
            (0.50, '50%', 1),
            (0.60, '60%', 1),
            (0.70, '70%', 2),
            (0.80, '80%', 2),
            (0.90, '90%', 3),
            (0.95, '95%', 3),
        ]
        for threshold, label, importance in rfl_thresholds:
            above = rfl[rfl['RFL_Trend'] >= threshold]
            if len(above) > 0:
                first_row = df.loc[above.index[0]]
                milestones.append({
                    'date': first_row['date'].strftime('%Y-%m-%d'),
                    'category': 'fitness',
                    'title': f'First {label} RFL',
                    'description': f'Running Fitness Level first reached {label}',
                    'icon': '📈',
                    'importance': importance,
                    'value': threshold,
                })
    
    # ========== 5. AGE GRADE MILESTONES ==========
    if 'age_grade_pct' in df.columns:
        ag = df[df['age_grade_pct'].notna() & (df['race_flag'] == 1)].copy()
        if len(ag) > 0:
            # Best AG performance
            best_ag_idx = ag['age_grade_pct'].idxmax()
            best_ag = df.loc[best_ag_idx]
            milestones.append({
                'date': best_ag['date'].strftime('%Y-%m-%d'),
                'category': 'age_grade',
                'title': f'Best Age Grade: {best_ag["age_grade_pct"]:.1f}%',
                'description': best_ag.get('activity_name', ''),
                'icon': '👑',
                'importance': 3,
                'value': best_ag['age_grade_pct'],
            })
            
            # First time crossing AG thresholds
            ag_thresholds = [
                (60, '60%', 'Regional', 1),
                (65, '65%', 'Regional+', 1),
                (70, '70%', 'Local', 2),
                (75, '75%', 'National', 2),
                (80, '80%', 'National+', 3),
            ]
            ag_sorted = ag.sort_values('date')
            ag_running_best = ag_sorted['age_grade_pct'].cummax()
            for threshold, label, level, importance in ag_thresholds:
                above = ag_sorted[ag_running_best >= threshold]
                if len(above) > 0:
                    first_row = df.loc[above.index[0]]
                    milestones.append({
                        'date': first_row['date'].strftime('%Y-%m-%d'),
                        'category': 'age_grade',
                        'title': f'First {label} Age Grade',
                        'description': f'{level} class performance',
                        'icon': '🎖️',
                        'importance': importance,
                        'value': threshold,
                    })
    
    # ========== 6. YEARLY VOLUME MILESTONES ==========
    yearly = df.groupby(df['date'].dt.year).agg(
        total_dist=('distance_km', 'sum'),
        total_runs=('distance_km', 'count'),
        last_date=('date', 'max')
    )
    
    year_dist_thresholds = [
        (1000, '1,000 km Year', 1),
        (1500, '1,500 km Year', 2),
        (2000, '2,000 km Year', 2),
        (2500, '2,500 km Year', 3),
        (3000, '3,000 km Year', 3),
    ]
    
    for threshold, label, importance in year_dist_thresholds:
        # Find the first year that exceeded this threshold
        years_above = yearly[yearly['total_dist'] >= threshold]
        if len(years_above) > 0:
            first_year = years_above.index[0]
            # Find the exact run that crossed the threshold within that year
            year_runs = df[df['date'].dt.year == first_year].copy()
            year_cum = year_runs['distance_km'].cumsum()
            cross_idx = year_cum[year_cum >= threshold].index
            if len(cross_idx) > 0:
                cross_row = df.loc[cross_idx[0]]
                milestones.append({
                    'date': cross_row['date'].strftime('%Y-%m-%d'),
                    'category': 'yearly',
                    'title': f'{label} ({first_year})',
                    'description': f'First year reaching {label}',
                    'icon': '📅',
                    'importance': importance,
                    'value': threshold,
                })
    
    # ========== 7. CONSISTENCY MILESTONES ==========
    # Consecutive day streaks
    run_dates = sorted(df['date'].dt.date.unique())
    if len(run_dates) > 1:
        streak = 1
        max_streak = 1
        max_streak_end = run_dates[0]
        current_streak_start = run_dates[0]
        best_streak_start = run_dates[0]
        
        for i in range(1, len(run_dates)):
            if (run_dates[i] - run_dates[i-1]).days == 1:
                streak += 1
                if streak > max_streak:
                    max_streak = streak
                    max_streak_end = run_dates[i]
                    best_streak_start = current_streak_start
            else:
                streak = 1
                current_streak_start = run_dates[i]
        
        if max_streak >= 7:
            milestones.append({
                'date': max_streak_end.strftime('%Y-%m-%d'),
                'category': 'consistency',
                'title': f'{max_streak}-Day Streak',
                'description': f'{best_streak_start.strftime("%d %b")} to {max_streak_end.strftime("%d %b %Y")}',
                'icon': '🔥',
                'importance': 2 if max_streak >= 14 else 1,
                'value': max_streak,
            })
    
    # Consecutive weeks with 4+ runs
    df_copy = df.copy()
    df_copy['iso_year'] = df_copy['date'].dt.isocalendar().year.astype(int)
    df_copy['iso_week'] = df_copy['date'].dt.isocalendar().week.astype(int)
    weekly_counts = df_copy.groupby(['iso_year', 'iso_week']).size().reset_index(name='runs')
    weekly_counts = weekly_counts.sort_values(['iso_year', 'iso_week'])
    
    for min_runs, label in [(4, '4+ runs/week'), (5, '5+ runs/week')]:
        streak = 0
        max_streak = 0
        max_end_row = None
        for _, wrow in weekly_counts.iterrows():
            if wrow['runs'] >= min_runs:
                streak += 1
                if streak > max_streak:
                    max_streak = streak
                    max_end_row = wrow
            else:
                streak = 0
        
        if max_streak >= 10:
            # Approximate end date from iso year/week
            end_date = datetime.strptime(f"{int(max_end_row['iso_year'])}-W{int(max_end_row['iso_week']):02d}-7", "%G-W%V-%u").date()
            milestones.append({
                'date': end_date.strftime('%Y-%m-%d'),
                'category': 'consistency',
                'title': f'{max_streak} Weeks of {label}',
                'description': f'Longest streak of consistent training',
                'icon': '📆',
                'importance': 2 if max_streak >= 26 else 1,
                'value': max_streak,
            })
    
    # ========== 8. RACE COUNT MILESTONES ==========
    race_counts = [
        (50,  '50th Race',  '🏁', 1),
        (100, '100th Race', '🏁', 2),
        (200, '200th Race', '🏁', 2),
        (300, '300th Race', '🏁', 3),
    ]
    race_cumcount = (df['race_flag'] == 1).cumsum()
    for threshold, label, icon, importance in race_counts:
        idx = race_cumcount[race_cumcount >= threshold].index
        if len(idx) > 0:
            row = df.loc[idx[0]]
            milestones.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'category': 'races',
                'title': label,
                'description': row.get('activity_name', ''),
                'icon': icon,
                'importance': importance,
                'value': threshold,
            })
    
    # ========== COMPUTE NEXT MILESTONES ==========
    next_milestones = []
    total_dist = df['distance_km'].sum()
    total_runs = len(df)
    total_races = (df['race_flag'] == 1).sum()
    
    # Next distance milestone
    for threshold, label, icon, importance in distance_thresholds:
        if total_dist < threshold:
            remaining = threshold - total_dist
            # Estimate days based on recent pace (last 90 days avg)
            recent = df[df['date'] >= df['date'].max() - pd.Timedelta(days=90)]
            if len(recent) > 0:
                daily_avg = recent['distance_km'].sum() / 90
                est_days = int(remaining / daily_avg) if daily_avg > 0 else None
            else:
                est_days = None
            next_milestones.append({
                'category': 'distance',
                'title': f'{label} Total Distance',
                'remaining': f'{remaining:.0f} km to go',
                'est_days': est_days,
                'icon': icon,
                'pct_complete': total_dist / threshold * 100,
            })
            break
    
    # Next run count
    for threshold, icon, importance in run_thresholds:
        if total_runs < threshold:
            remaining = threshold - total_runs
            next_milestones.append({
                'category': 'runs',
                'title': f'Run #{threshold:,}',
                'remaining': f'{remaining} runs to go',
                'icon': icon,
                'pct_complete': total_runs / threshold * 100,
            })
            break
    
    # Next race count
    for threshold, label, icon, importance in race_counts:
        if total_races < threshold:
            remaining = threshold - total_races
            next_milestones.append({
                'category': 'races',
                'title': label,
                'remaining': f'{remaining} races to go',
                'icon': icon,
                'pct_complete': total_races / threshold * 100,
            })
            break
    
    # Sort milestones by date
    milestones.sort(key=lambda m: m['date'])
    
    # Recent achievements (best X in Y years)
    recent_achievements = get_recent_achievements(df, lookback_days=30)
    
    # Summary
    summary = {
        'total_distance_km': total_dist,
        'total_runs': total_runs,
        'total_races': total_races,
        'years_active': len(df['date'].dt.year.unique()),
        'first_run': df['date'].min().strftime('%Y-%m-%d'),
        'latest_run': df['date'].max().strftime('%Y-%m-%d'),
    }
    
    return {
        'milestones': milestones,
        'next_milestones': next_milestones,
        'recent_achievements': recent_achievements,
        'summary': summary,
    }


def generate_milestones_html(milestone_data):
    """Generate the HTML section for milestones in the dashboard."""
    milestones = milestone_data['milestones']
    next_milestones = milestone_data['next_milestones']
    recent_achievements = milestone_data.get('recent_achievements', [])
    summary = milestone_data['summary']
    
    if not milestones:
        return '<!-- milestones: no data -->'
    
    milestones_json = json.dumps(milestones)
    next_json = json.dumps(next_milestones)
    recent_json = json.dumps(recent_achievements)
    
    return f'''
    <!-- Recent Achievements (if any) -->
    <div class="chart-container" id="recentAchievementsSection" style="display:none;">
        <div class="chart-title">🔥 Recent Achievements</div>
        <div class="chart-desc">Last 30 days</div>
        <div id="recentAchievementsList"></div>
    </div>
    
    <!-- Milestones -->
    <div class="chart-container" id="milestonesSection">
        <div class="chart-title-row">
            <span class="chart-title">🏆 Milestones</span>
            <div class="chart-toggle" id="milestoneToggle">
                <button class="active" data-filter="highlights">Top</button>
                <button data-filter="pbs">PBs</button>
                <button data-filter="volume">Volume</button>
                <button data-filter="fitness">Fitness</button>
                <button data-filter="all">All</button>
            </div>
        </div>
        <div class="chart-desc">{summary['total_runs']:,} runs · {summary['total_distance_km']:,.0f} km · {summary['total_races']} races · {summary['years_active']} years</div>
        
        <!-- Next milestone progress bars -->
        <div id="nextMilestones" style="display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 14px 0;">
        </div>
        
        <!-- Milestone timeline -->
        <div id="milestoneTimeline" style="max-height: 420px; overflow-y: auto; padding-right: 4px;">
        </div>
    </div>
    
    <script>
    (function() {{
        const allMilestones = {milestones_json};
        const nextMilestones = {next_json};
        const recentAchievements = {recent_json};
        
        // --- Recent Achievements ---
        const recentSection = document.getElementById('recentAchievementsSection');
        const recentList = document.getElementById('recentAchievementsList');
        if (recentAchievements.length > 0) {{
            recentSection.style.display = 'block';
            recentList.innerHTML = recentAchievements.map(a => {{
                const sigStars = a.significance >= 5 ? '★★★' : (a.significance >= 3 ? '★★' : '★');
                const sigColor = a.significance >= 5 ? '#f59e0b' : (a.significance >= 3 ? '#818cf8' : '#94a3b8');
                const windowBadge = a.window === 'all-time' 
                    ? '<span style="font-size:9px;padding:1px 5px;border-radius:3px;background:#f59e0b22;color:#f59e0b;font-weight:600;">ALL-TIME</span>'
                    : `<span style="font-size:9px;padding:1px 5px;border-radius:3px;background:#818cf822;color:#818cf8;font-weight:500;">${{a.window}}</span>`;
                
                const name = a.description && a.description.length > 60 ? a.description.slice(0, 57) + '...' : (a.description || '');
                
                return `<div style="padding:8px 10px;border-bottom:1px solid var(--border,#2e3340);display:flex;align-items:center;gap:8px;">
                    <span style="font-size:22px;">${{a.icon}}</span>
                    <div style="flex:1;min-width:0;">
                        <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
                            <span style="font-weight:600;color:var(--text,#e4e7ef);font-size:13px;">${{a.title}}</span>
                            ${{windowBadge}}
                        </div>
                        <div style="font-size:11px;color:var(--text-dim,#8b8fa3);margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${{name}}</div>
                    </div>
                    <div style="font-size:10px;color:${{sigColor}};white-space:nowrap;">${{sigStars}}</div>
                </div>`;
            }}).join('');
        }}
        
        // --- Next milestones (progress bars) ---
        const nextContainer = document.getElementById('nextMilestones');
        if (nextMilestones.length > 0) {{
            nextContainer.innerHTML = nextMilestones.map(m => {{
                const pct = Math.min(m.pct_complete, 100).toFixed(1);
                const estText = m.est_days ? ` · ~${{m.est_days}}d` : '';
                return `<div style="flex:1; min-width:140px; background:var(--card-bg, #1a1d27); border:1px solid var(--border, #2e3340); border-radius:8px; padding:8px 10px;">
                    <div style="font-size:11px; color:var(--text-dim, #8b8fa3); margin-bottom:4px;">${{m.icon}} ${{m.title}}</div>
                    <div style="background:var(--bg, #0f1117); border-radius:4px; height:6px; overflow:hidden;">
                        <div style="width:${{pct}}%; height:100%; background:var(--accent, #818cf8); border-radius:4px; transition:width 0.5s;"></div>
                    </div>
                    <div style="font-size:10px; color:var(--text-dim, #8b8fa3); margin-top:3px;">${{m.remaining}}${{estText}}</div>
                </div>`;
            }}).join('');
        }} else {{
            nextContainer.style.display = 'none';
        }}
        
        // Category → colour mapping
        const catColors = {{
            distance: '#818cf8',
            runs:     '#60a5fa',
            pb:       '#f59e0b',
            fitness:  '#4ade80',
            age_grade:'#f472b6',
            yearly:   '#38bdf8',
            consistency:'#fb923c',
            races:    '#a78bfa',
        }};
        
        const catLabels = {{
            distance: 'Distance', runs: 'Runs', pb: 'PB', fitness: 'Fitness',
            age_grade: 'Age Grade', yearly: 'Yearly', consistency: 'Streak', races: 'Races'
        }};
        
        function formatDate(dateStr) {{
            const d = new Date(dateStr);
            const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
            return `${{d.getDate()}} ${{months[d.getMonth()]}} ${{d.getFullYear()}}`;
        }}
        
        function renderMilestones(filter) {{
            let filtered;
            if (filter === 'highlights') {{
                // Show importance 3 first, then 2, newest first within each tier
                filtered = [...allMilestones]
                    .filter(m => m.importance >= 2)
                    .sort((a, b) => {{
                        if (b.importance !== a.importance) return b.importance - a.importance;
                        return new Date(b.date) - new Date(a.date);
                    }});
            }} else if (filter === 'pbs') {{
                filtered = allMilestones.filter(m => m.category === 'pb')
                    .sort((a, b) => new Date(b.date) - new Date(a.date));
            }} else if (filter === 'volume') {{
                filtered = allMilestones.filter(m => ['distance', 'runs', 'yearly', 'races'].includes(m.category))
                    .sort((a, b) => new Date(b.date) - new Date(a.date));
            }} else if (filter === 'fitness') {{
                filtered = allMilestones.filter(m => ['fitness', 'age_grade', 'consistency'].includes(m.category))
                    .sort((a, b) => new Date(b.date) - new Date(a.date));
            }} else {{
                filtered = [...allMilestones].sort((a, b) => new Date(b.date) - new Date(a.date));
            }}
            
            const container = document.getElementById('milestoneTimeline');
            if (filtered.length === 0) {{
                container.innerHTML = '<div style="color:var(--text-dim,#8b8fa3);padding:20px;text-align:center;">No milestones in this category yet</div>';
                return;
            }}
            
            container.innerHTML = filtered.map(m => {{
                const color = catColors[m.category] || '#818cf8';
                const stars = m.importance >= 3 ? '★★★' : (m.importance >= 2 ? '★★' : '★');
                const starColor = m.importance >= 3 ? '#f59e0b' : (m.importance >= 2 ? '#94a3b8' : '#475569');
                const isPB = m.is_current_pb ? ' style="border-left:3px solid #f59e0b;"' : '';
                const badge = catLabels[m.category] || m.category;
                
                return `<div class="milestone-item"${{isPB}}>
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="font-size:20px;">${{m.icon}}</span>
                        <div style="flex:1; min-width:0;">
                            <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
                                <span style="font-weight:600; color:var(--text,#e4e7ef); font-size:13px;">${{m.title}}</span>
                                <span style="font-size:9px; padding:1px 5px; border-radius:3px; background:${{color}}22; color:${{color}}; font-weight:500;">${{badge}}</span>
                                <span style="font-size:10px; color:${{starColor}};">${{stars}}</span>
                            </div>
                            <div style="font-size:11px; color:var(--text-dim,#8b8fa3); margin-top:1px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${{m.description}}</div>
                        </div>
                        <div style="font-size:11px; color:var(--text-dim,#8b8fa3); white-space:nowrap;">${{formatDate(m.date)}}</div>
                    </div>
                </div>`;
            }}).join('');
        }}
        
        // Toggle handlers
        document.querySelectorAll('#milestoneToggle button').forEach(btn => {{
            btn.addEventListener('click', function() {{
                document.querySelectorAll('#milestoneToggle button').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                renderMilestones(this.dataset.filter);
            }});
        }});
        
        // Initial render
        renderMilestones('highlights');
    }})();
    </script>
    '''


def generate_test_dashboard(df, cutoff_date=None, title=None):
    """
    Generate a standalone test dashboard showing milestones at a given cutoff date.
    Returns complete HTML string.
    """
    milestone_data = get_milestones_data(df, cutoff_date=cutoff_date)
    
    if cutoff_date:
        if isinstance(cutoff_date, str):
            cutoff_dt = pd.Timestamp(cutoff_date)
        else:
            cutoff_dt = cutoff_date
    else:
        cutoff_dt = df['date'].max()
    
    if title is None:
        title = f'Milestones as of {cutoff_dt.strftime("%d %b %Y")}'
    
    milestones_section = generate_milestones_html(milestone_data)
    summary = milestone_data['summary']
    recent_count = len(milestone_data.get('recent_achievements', []))
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    :root {{
        --bg: #0f1117;
        --card-bg: #1a1d27;
        --border: #2e3340;
        --text: #e4e7ef;
        --text-dim: #8b8fa3;
        --accent: #818cf8;
        --grid: rgba(255,255,255,0.04);
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'DM Sans', sans-serif;
        background: var(--bg);
        color: var(--text);
        padding: 12px;
        max-width: 700px;
        margin: 0 auto;
    }}
    .header {{
        text-align: center;
        padding: 20px 0 16px 0;
    }}
    .header h1 {{
        font-size: 20px;
        font-weight: 600;
        color: var(--text);
    }}
    .header .subtitle {{
        font-size: 12px;
        color: var(--text-dim);
        margin-top: 4px;
    }}
    .stats-row {{
        display: flex;
        gap: 8px;
        margin-bottom: 14px;
        flex-wrap: wrap;
    }}
    .stat-card {{
        flex: 1;
        min-width: 80px;
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 12px;
        text-align: center;
    }}
    .stat-card .value {{
        font-size: 20px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent);
    }}
    .stat-card .label {{
        font-size: 10px;
        color: var(--text-dim);
        margin-top: 2px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .chart-container {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 14px;
    }}
    .chart-title-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
        flex-wrap: wrap;
        gap: 6px;
    }}
    .chart-title {{
        font-size: 15px;
        font-weight: 600;
    }}
    .chart-desc {{
        font-size: 11px;
        color: var(--text-dim);
        margin-bottom: 10px;
    }}
    .chart-toggle {{
        display: flex;
        gap: 3px;
        background: var(--bg);
        border-radius: 6px;
        padding: 2px;
    }}
    .chart-toggle button {{
        background: transparent;
        border: none;
        color: var(--text-dim);
        font-family: 'DM Sans', sans-serif;
        font-size: 11px;
        padding: 4px 8px;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.15s;
    }}
    .chart-toggle button.active {{
        background: var(--accent);
        color: #fff;
    }}
    .chart-toggle button:hover:not(.active) {{
        color: var(--text);
    }}
    .milestone-item {{
        padding: 8px 10px;
        border-bottom: 1px solid var(--border);
        transition: background 0.15s;
    }}
    .milestone-item:hover {{
        background: rgba(129, 140, 248, 0.04);
    }}
    .milestone-item:last-child {{
        border-bottom: none;
    }}
    #milestoneTimeline::-webkit-scrollbar {{ width: 4px; }}
    #milestoneTimeline::-webkit-scrollbar-track {{ background: transparent; }}
    #milestoneTimeline::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}
    .cutoff-badge {{
        display: inline-block;
        background: #f59e0b22;
        color: #f59e0b;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 500;
    }}
</style>
</head>
<body>
    <div class="header">
        <h1>🏆 Running Milestones</h1>
        <div class="subtitle">
            <span class="cutoff-badge">Data through {cutoff_dt.strftime('%d %b %Y')}</span>
        </div>
    </div>
    
    <div class="stats-row">
        <div class="stat-card">
            <div class="value">{summary['total_runs']:,}</div>
            <div class="label">Runs</div>
        </div>
        <div class="stat-card">
            <div class="value">{summary['total_distance_km']:,.0f}</div>
            <div class="label">km</div>
        </div>
        <div class="stat-card">
            <div class="value">{summary['total_races']}</div>
            <div class="label">Races</div>
        </div>
        <div class="stat-card">
            <div class="value">{summary['years_active']}</div>
            <div class="label">Years</div>
        </div>
    </div>
    
    {milestones_section}
    
</body>
</html>'''
    return html


# ============================================================
# TEST: Generate dashboards at key cutoff dates
# ============================================================
if __name__ == '__main__':
    import sys
    import os
    
    MASTER_FILE = '/mnt/user-data/uploads/Master_FULL_GPSQ_ID_post.xlsx'
    OUTPUT_DIR = '/mnt/user-data/outputs'
    
    print("Loading master data...")
    df = pd.read_excel(MASTER_FILE, sheet_name='Master (A001)')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  {len(df)} runs, {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Generate current (full) milestones dashboard
    print("\n=== Generating full milestones dashboard ===")
    html = generate_test_dashboard(df)
    path = os.path.join(OUTPUT_DIR, 'milestones_current.html')
    with open(path, 'w') as f:
        f.write(html)
    print(f"  -> {path}")
    
    # Generate at key cutoff dates
    cutoff_dates = [
        ('2014-01-01', 'After ~1 year'),
        ('2016-01-01', 'Start of big volume year'),
        ('2017-06-01', 'Entering peak fitness era'),
        ('2019-06-01', 'Post-PB period (all PBs set)'),
        ('2020-06-01', 'Peak RFL = 1.000'),
        ('2022-01-01', 'Post-knee recovery'),
        ('2024-01-01', 'Recent era'),
    ]
    
    for cutoff, desc in cutoff_dates:
        print(f"\n=== {cutoff}: {desc} ===")
        milestone_data = get_milestones_data(df, cutoff_date=cutoff)
        n = len(milestone_data['milestones'])
        next_n = len(milestone_data['next_milestones'])
        recent_n = len(milestone_data.get('recent_achievements', []))
        summary = milestone_data['summary']
        print(f"  {summary['total_runs']:,} runs, {summary['total_distance_km']:,.0f} km, {n} milestones, {next_n} upcoming, {recent_n} recent achievements")
        
        # Show recent achievements
        for a in milestone_data.get('recent_achievements', []):
            print(f"  🔥 {a['icon']} {a['title']} [{a['window']}]")
        
        # Show top milestones
        top = [m for m in milestone_data['milestones'] if m['importance'] >= 2]
        for m in top[-5:]:
            print(f"  {m['icon']} {m['title']} ({m['date']})")
        
        # Show next milestones
        for nm in milestone_data['next_milestones']:
            print(f"  ⏳ {nm['title']}: {nm['remaining']} ({nm['pct_complete']:.0f}%)")
        
        html = generate_test_dashboard(df, cutoff_date=cutoff, title=f'Milestones: {desc} ({cutoff})')
        safe_name = cutoff.replace('-', '')
        path = os.path.join(OUTPUT_DIR, f'milestones_{safe_name}.html')
        with open(path, 'w') as f:
            f.write(html)
        print(f"  -> {path}")
    
    # Also generate an index page linking all dashboards
    print("\n=== Generating index page ===")
    index_links = [f'<a href="milestones_current.html" style="color:#818cf8;">Current (all data)</a><br>']
    for cutoff, desc in cutoff_dates:
        safe_name = cutoff.replace('-', '')
        index_links.append(f'<a href="milestones_{safe_name}.html" style="color:#818cf8;">{cutoff}: {desc}</a><br>')
    
    index_html = f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Milestone Dashboards</title>
<style>
body {{ font-family: 'DM Sans', sans-serif; background: #0f1117; color: #e4e7ef; padding: 40px; max-width: 600px; margin: 0 auto; }}
h1 {{ font-size: 24px; margin-bottom: 20px; }}
a {{ display: block; padding: 8px 0; font-size: 15px; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style></head>
<body>
<h1>🏆 Milestone Test Dashboards</h1>
<p style="color:#8b8fa3; margin-bottom:20px;">Dashboards generated at different date cutoffs to show milestones being reached over time.</p>
{''.join(index_links)}
</body></html>'''
    
    index_path = os.path.join(OUTPUT_DIR, 'milestones_index.html')
    with open(index_path, 'w') as f:
        f.write(index_html)
    print(f"  -> {index_path}")
    
    print(f"\nDone! {len(cutoff_dates) + 1} dashboards generated.")
