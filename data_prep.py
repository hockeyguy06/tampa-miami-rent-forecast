import pandas as pd
import os

# Your exact file paths
ZORI_ZORDI_PATH = '/Users/tylerbrenner/Documents/Projects/Housing/ZORI_ZORDI.csv'
ZORF_PATH = '/Users/tylerbrenner/Documents/Projects/Housing/ZORF.csv'

def melt_wide_to_long(df, value_name='rent_index'):
    """
    Convert Zillow wide-format CSV (regions as rows, dates as columns) to long format.
    """
    # Find non-date columns (identifiers) and date columns
    date_cols = [col for col in df.columns if '-' in col and len(col.split('-')) == 3]
    id_cols = [col for col in df.columns if col not in date_cols]
    
    if not date_cols:
        raise ValueError("No date columns found (expected format like '2015-01-31')")
    
    print(f"Melting {len(date_cols)} date columns into long format...")
    
    melted = pd.melt(
        df,
        id_vars=id_cols,
        value_vars=date_cols,
        var_name='date',
        value_name=value_name
    )
    melted['date'] = pd.to_datetime(melted['date'])
    melted = melted.sort_values(['RegionName', 'date']).reset_index(drop=True)
    return melted

# === Main processing ===
print("=== Starting data preparation ===")
print(f"Loading: {ZORI_ZORDI_PATH}")

raw = pd.read_csv(ZORI_ZORDI_PATH)

print("Columns in file:", raw.columns.tolist())
print("Sample RegionName values (first 5):", raw['RegionName'].head().tolist() if 'RegionName' in raw.columns else "No RegionName column")

# Melt the data (assuming the values are rent index)
data = melt_wide_to_long(raw, value_name='rent_index')

# Filter to Tampa Bay (and optionally Miami) metros
# Adjust patterns if your exact metro names differ (run once to see printed RegionName)
metro_patterns = ['Tampa', 'St. Petersburg', 'Clearwater', 'Miami', 'Fort Lauderdale', 'West Palm']
pattern = '|'.join(metro_patterns)

if 'RegionName' in data.columns:
    mask = data['RegionName'].astype(str).str.contains(pattern, case=False, na=False)
    data = data[mask].copy()
    print(f"\nFiltered to {len(data)} rows matching Tampa/Miami metros")
    if len(data) > 0:
        print("Unique metros kept:", data['RegionName'].unique().tolist())
    else:
        print("WARNING: No matches found. Check printed RegionName values above and adjust patterns.")
        print("Showing first 10 RegionName for debugging:")
        print(data['RegionName'].unique()[:10].tolist())
else:
    print("No 'RegionName' column — cannot filter by metro. Using all data.")
    # Optional fallback: if 'StateName' exists and is 'FL'
    if 'StateName' in data.columns:
        data = data[data['StateName'] == 'FL'].copy()
        print(f"Fallback filter: {len(data)} rows where StateName == 'FL'")

# Keep only recent data
data = data[data['date'] >= '2015-01-01'].copy()

# Merge ZORF national forecast (joins on date)
if os.path.exists(ZORF_PATH):
    print(f"Merging ZORF from: {ZORF_PATH}")
    zorf_raw = pd.read_csv(ZORF_PATH)
    zorf = melt_wide_to_long(zorf_raw, value_name='zorf_growth_pct')
    data = pd.merge(data, zorf[['date', 'zorf_growth_pct']], on='date', how='left')
else:
    print("ZORF.csv not found — skipping merge")

# Add lag features and seasonality
if 'rent_index' in data.columns and len(data) > 0:
    data['rent_lag1'] = data.groupby('RegionName')['rent_index'].shift(1)
    data['rent_lag3'] = data.groupby('RegionName')['rent_index'].shift(3)
    data['month'] = data['date'].dt.month
    data.dropna(subset=['rent_index'], inplace=True)
    print(f"Added lags and month feature. Final rows: {len(data)}")
else:
    print("No rent_index column or empty data after filtering — check file contents")

# Save the result
output_path = '/Users/tylerbrenner/Documents/Projects/Housing/tampa_miami_rent_prepared.csv'
data.to_csv(output_path, index=False)
print(f"\n=== Done ===\nPrepared file saved to:\n{output_path}")
print(f"Final shape: {data.shape}")
if len(data) > 0:
    print("Sample data (first 5 rows):\n")
    print(data.head().to_string(index=False))
else:
    print("No data after processing. Please check the RegionName values printed above and adjust metro_patterns.")