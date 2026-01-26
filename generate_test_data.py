import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def random_date(start, end):
    """Generate a random datetime between `start` and `end`"""
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)

def generate_data():
    """Generate a test CSV file with 10 series of random daily returns."""
    # Define bounds
    start_range_begin = datetime(2000, 1, 1)
    start_range_end = datetime(2005, 1, 1)
    end_range_begin = datetime(2020, 12, 31)
    end_range_end = datetime(2025, 12, 31)

    full_start = start_range_begin
    full_end = end_range_end
    master_index = pd.date_range(start=full_start, end=full_end, freq='D')
    
    df = pd.DataFrame(index=master_index)
    df.index.name = "Date"

    print("Generating 10 series with random start/end dates...")
    
    for i in range(1, 11):
        series_name = f"Series_{i}"
        
        # Random start and end
        s_date = random_date(start_range_begin, start_range_end)
        e_date = random_date(end_range_begin, end_range_end)
        
        # Round to day to match index frequency roughly
        s_date = s_date.replace(hour=0, minute=0, second=0, microsecond=0)
        e_date = e_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Create mask
        mask = (df.index >= s_date) & (df.index <= e_date)
        n_days = mask.sum()
        
        if n_days > 0:
            # Generate returns: Mean 0.0004 (~10% annual), Std 0.015
            returns = np.random.normal(loc=0.0004, scale=0.015, size=n_days)
            
            # Initialize column
            df[series_name] = np.nan
            df.loc[mask, series_name] = returns
            
            print(f"  {series_name}: {s_date.date()} to {e_date.date()} ({n_days} days)")

    # Drop rows where all columns are NaN
    df.dropna(how='all', inplace=True)
    
    output_file = "test_data.csv"
    df.to_csv(output_file)
    print(f"\nSaved to {output_file}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    generate_data()
