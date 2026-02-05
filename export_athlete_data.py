"""
Export Athlete Data from BFW to athlete_data.csv

This script reads weight, Garmin Training Readiness, and non-running TSS
from the BFW (Big Fitness Workbook) and writes them to athlete_data.csv
for use by the v46+ pipeline.

Usage:
    python export_athlete_data.py [--bfw BFW_FILE] [--out OUTPUT_FILE]

BFW Sheet Structure:
    - Daily Metrics: Column A (Date), Column C (Weight), Column Z (Garmin TR)
    - Biking history: Column P (Date), Column G (TSS)
    - Cardio History: Column A (Date), Column G (TSS)

Output:
    athlete_data.csv with columns: date, weight_kg, garmin_tr, non_running_tss
"""

import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
BFW_FILE = r"Running_and_Fitness_analysis.xlsx"
OUTPUT_FILE = r"athlete_data.csv"


def parse_weight(val):
    """Parse weight value, handling 'kg' suffix."""
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower().replace('kg', '').strip()
    try:
        return float(val_str)
    except ValueError:
        return np.nan


def export_from_bfw(bfw_path: str, output_path: str):
    """
    Export athlete data from BFW to CSV.
    
    Args:
        bfw_path: Path to BFW Excel file
        output_path: Path to output CSV file
    """
    print(f"Reading BFW file: {bfw_path}")
    
    if not os.path.exists(bfw_path):
        print(f"ERROR: BFW file not found: {bfw_path}")
        return False
    
    # =========================================================================
    # Read Daily Metrics sheet (Weight and Garmin TR)
    # Column A = Date, Column C = Weight, Column Z = Garmin TR
    # =========================================================================
    print("\
--- Daily Metrics ---")
    try:
        df_dm = pd.read_excel(bfw_path, sheet_name='Daily Metrics', usecols='A,C,Z')
        df_dm.columns = ['date', 'weight', 'garmin_tr']
        df_dm['date'] = pd.to_datetime(df_dm['date'], errors='coerce')
        df_dm = df_dm[df_dm['date'].notna()].copy()
        df_dm = df_dm.sort_values('date')
        print(f"  Loaded {len(df_dm)} rows")
        print(f"  Date range: {df_dm['date'].min().strftime('%Y-%m-%d')} to {df_dm['date'].max().strftime('%Y-%m-%d')}")
        
        # Parse weight
        df_dm['weight_kg'] = df_dm['weight'].apply(parse_weight)
        valid_weights = df_dm['weight_kg'].notna().sum()
        print(f"  Weight values: {valid_weights}")
        
        # Parse Garmin TR
        df_dm['garmin_tr'] = pd.to_numeric(df_dm['garmin_tr'], errors='coerce')
        valid_tr = df_dm['garmin_tr'].notna().sum()
        print(f"  Garmin TR values: {valid_tr}")
        
    except Exception as e:
        print(f"ERROR: Could not read Daily Metrics sheet: {e}")
        return False
    
    # =========================================================================
    # Read Biking history sheet (TSS)
    # Column P = Date, Column G = TSS
    # =========================================================================
    print("\
--- Biking history ---")
    biking_tss = {}
    try:
        df_bike = pd.read_excel(bfw_path, sheet_name='Biking history', usecols='G,P')
        df_bike.columns = ['tss', 'date']
        df_bike['date'] = pd.to_datetime(df_bike['date'], errors='coerce')
        df_bike = df_bike[df_bike['date'].notna()].copy()
        df_bike['tss'] = pd.to_numeric(df_bike['tss'], errors='coerce').fillna(0)
        
        # Aggregate by date
        bike_agg = df_bike.groupby(df_bike['date'].dt.strftime('%Y-%m-%d'))['tss'].sum()
        biking_tss = bike_agg.to_dict()
        print(f"  Loaded {len(df_bike)} activities, {len(biking_tss)} unique days")
        print(f"  Total TSS: {sum(biking_tss.values()):.0f}")
    except Exception as e:
        print(f"  WARNING: Could not read Biking history sheet: {e}")
    
    # =========================================================================
    # Read Cardio History sheet (TSS)
    # Column A = Date, Column G = TSS
    # =========================================================================
    print("\
--- Cardio History ---")
    cardio_tss = {}
    try:
        df_cardio = pd.read_excel(bfw_path, sheet_name='Cardio History', usecols='A,G')
        df_cardio.columns = ['date', 'tss']
        df_cardio['date'] = pd.to_datetime(df_cardio['date'], errors='coerce')
        df_cardio = df_cardio[df_cardio['date'].notna()].copy()
        df_cardio['tss'] = pd.to_numeric(df_cardio['tss'], errors='coerce').fillna(0)
        
        # Aggregate by date
        cardio_agg = df_cardio.groupby(df_cardio['date'].dt.strftime('%Y-%m-%d'))['tss'].sum()
        cardio_tss = cardio_agg.to_dict()
        print(f"  Loaded {len(df_cardio)} activities, {len(cardio_tss)} unique days")
        print(f"  Total TSS: {sum(cardio_tss.values()):.0f}")
    except Exception as e:
        print(f"  WARNING: Could not read Cardio History sheet: {e}")
    
    # =========================================================================
    # Combine all data
    # =========================================================================
    print("\
--- Combining data ---")
    
    # Start with Daily Metrics dates
    out_df = pd.DataFrame()
    out_df['date'] = df_dm['date'].dt.strftime('%Y-%m-%d')
    out_df['weight_kg'] = df_dm['weight_kg'].values
    out_df['garmin_tr'] = df_dm['garmin_tr'].values
    
    # Add non-running TSS (sum of biking + cardio)
    def get_nr_tss(date_str):
        bike = biking_tss.get(date_str, 0)
        cardio = cardio_tss.get(date_str, 0)
        total = bike + cardio
        return total if total > 0 else np.nan
    
    out_df['non_running_tss'] = out_df['date'].apply(get_nr_tss)
    
    # Also add any dates that have TSS but weren't in Daily Metrics
    all_tss_dates = set(biking_tss.keys()) | set(cardio_tss.keys())
    existing_dates = set(out_df['date'])
    missing_dates = all_tss_dates - existing_dates
    
    if missing_dates:
        print(f"  Adding {len(missing_dates)} dates with TSS not in Daily Metrics")
        missing_rows = []
        for d in sorted(missing_dates):
            missing_rows.append({
                'date': d,
                'weight_kg': np.nan,
                'garmin_tr': np.nan,
                'non_running_tss': get_nr_tss(d)
            })
        out_df = pd.concat([out_df, pd.DataFrame(missing_rows)], ignore_index=True)
        out_df = out_df.sort_values('date').reset_index(drop=True)
    
    # Filter to rows that have at least one value
    has_data = (out_df['weight_kg'].notna() | 
                out_df['garmin_tr'].notna() | 
                out_df['non_running_tss'].notna())
    out_df = out_df[has_data].copy()
    
    print(f"  Total rows with data: {len(out_df)}")
    
    # =========================================================================
    # Write output CSV
    # =========================================================================
    with open(output_path, 'w', newline='') as f:
        f.write(f"# Athlete data exported from BFW on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"# Source: {os.path.basename(bfw_path)}\n")
    
    out_df.to_csv(output_path, mode='a', index=False)
    
    print(f"\
\u2713 Exported to: {output_path}")
    
    # Summary
    print(f"\
Summary:")
    print(f"  Rows with weight:    {out_df['weight_kg'].notna().sum()}")
    print(f"  Rows with Garmin TR: {out_df['garmin_tr'].notna().sum()}")
    print(f"  Rows with NR TSS:    {out_df['non_running_tss'].notna().sum()}")
    
    if len(out_df) > 0:
        # Find latest of each
        weight_rows = out_df[out_df['weight_kg'].notna()]
        tr_rows = out_df[out_df['garmin_tr'].notna()]
        tss_rows = out_df[out_df['non_running_tss'].notna()]
        
        print(f"\
Latest values:")
        if len(weight_rows) > 0:
            latest = weight_rows.iloc[-1]
            print(f"  Weight:    {latest['weight_kg']:.1f} kg ({latest['date']})")
        if len(tr_rows) > 0:
            latest = tr_rows.iloc[-1]
            print(f"  Garmin TR: {int(latest['garmin_tr'])}% ({latest['date']})")
        if len(tss_rows) > 0:
            latest = tss_rows.iloc[-1]
            print(f"  NR TSS:    {latest['non_running_tss']:.0f} ({latest['date']})")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export athlete data from BFW to athlete_data.csv"
    )
    parser.add_argument(
        "--bfw", 
        default=BFW_FILE, 
        help=f"Path to BFW Excel file (default: {BFW_FILE})"
    )
    parser.add_argument(
        "--out", 
        default=OUTPUT_FILE, 
        help=f"Output CSV file (default: {OUTPUT_FILE})"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Export Athlete Data from BFW")
    print("=" * 60)
    
    success = export_from_bfw(args.bfw, args.out)
    
    if not success:
        print("\
Export failed. Check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())