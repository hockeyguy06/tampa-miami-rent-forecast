import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import joblib
import warnings
import numpy as np
import json
from pandas_datareader import data as pdr
import datetime

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load prepared data
file_path = '/Users/tylerbrenner/Documents/Projects/Housing/tampa_miami_rent_prepared.csv'
df = pd.read_csv(file_path)
df['ds'] = pd.to_datetime(df['date'])
df['y'] = df['rent_index']

print(f"Loaded {len(df)} rows. Unique metros: {df['RegionName'].unique().tolist()}")

# FRED unemployment series
UNEMP_SERIES = {
    "Miami, FL": "MIAM112UR",      # Miami-Fort Lauderdale-West Palm Beach
    "Tampa, FL": "TAMU336UR"       # Tampa-St. Petersburg-Clearwater
}

cv_results = {}

def train_and_forecast_metro(metro_df, metro_name):
    grouped = metro_df.groupby('ds').agg({
        'y': 'mean',
        'zorf_growth_pct': 'mean',
        'rent_lag1': 'mean',
        'rent_lag3': 'mean',
        'month': 'first'
    }).reset_index()

    # === Add unemployment regressor ===
    series_id = UNEMP_SERIES.get(metro_name)
    if series_id:
        try:
            start = '2014-01-01'
            end = datetime.datetime.now().strftime('%Y-%m-%d')
            unemp = pdr.DataReader(series_id, 'fred', start, end)
            unemp = unemp.resample('ME').last().reset_index()
            unemp.columns = ['ds', 'unemployment_rate']
            unemp['ds'] = pd.to_datetime(unemp['ds']) + pd.offsets.MonthEnd(0)
            grouped = pd.merge(grouped, unemp, on='ds', how='left')
            grouped['unemployment_rate'] = grouped['unemployment_rate'].ffill().bfill().fillna(4.0)
            print(f"Added unemployment rate for {metro_name} (latest: {grouped['unemployment_rate'].iloc[-1]:.1f}%)")
        except Exception as e:
            print(f"Could not fetch unemployment for {metro_name}: {e}")
            grouped['unemployment_rate'] = 4.0
    else:
        grouped['unemployment_rate'] = 4.0

    # Handle other NaNs
    grouped['zorf_growth_pct'] = grouped['zorf_growth_pct'].ffill().bfill().fillna(0)
    grouped['rent_lag1'] = grouped['rent_lag1'].ffill().bfill().fillna(grouped['y'])
    grouped['rent_lag3'] = grouped['rent_lag3'].ffill().bfill().fillna(grouped['y'])
    grouped = grouped.dropna(subset=['y', 'ds'])

    print(f"\n=== {metro_name} === Training on {len(grouped)} months")

    # Historical growth rate (last ~5 years)
    cutoff = grouped['ds'].max() - pd.Timedelta(1825, 'D')
    recent = grouped[grouped['ds'] >= cutoff]
    if len(recent) >= 12:
        months = (recent['ds'].iloc[-1] - recent['ds'].iloc[0]).days / 30.44
        growth_monthly = (recent['y'].iloc[-1] / recent['y'].iloc[0]) ** (1 / months) - 1
        print(f"Historical monthly growth: {growth_monthly:.4f} (~{growth_monthly*1200:.1f}% annual)")
    else:
        growth_monthly = 0.002

    # Prophet model
    model = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.05)
    model.add_regressor('zorf_growth_pct', standardize=False)
    model.add_regressor('rent_lag1', standardize=False)
    model.add_regressor('rent_lag3', standardize=False)
    model.add_regressor('unemployment_rate', standardize=False)

    model.fit(grouped)

    # Cross-validation
    print("Running cross-validation...")
    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
    df_p = performance_metrics(df_cv)
    mape_1yr = df_p[df_p['horizon'] == pd.Timedelta(365, 'D')]['mape'].iloc[0] if not df_p.empty else None

    if mape_1yr:
        cv_results[metro_name] = {
            "mape_1yr": float(mape_1yr),
            "mae_1yr": float(df_p[df_p['horizon'] == pd.Timedelta(365, 'D')]['mae'].iloc[0]),
            "coverage_1yr": float(df_p[df_p['horizon'] == pd.Timedelta(365, 'D')]['coverage'].iloc[0])
        }

    # Future + projection
    future = model.make_future_dataframe(periods=24, freq='ME')
    last_y = grouped['y'].iloc[-1]
    future['zorf_growth_pct'] = grouped['zorf_growth_pct'].iloc[-1]
    future['unemployment_rate'] = grouped['unemployment_rate'].iloc[-1]  # simple hold

    steps = np.arange(len(future)) - len(grouped)
    growth = (1 + growth_monthly) ** steps
    future['rent_lag1'] = last_y * growth
    lag3 = np.ones(len(future))
    lag3[2:] = growth[:-2]
    future['rent_lag3'] = last_y * lag3

    forecast = model.predict(future)

    # Save
    safe = metro_name.replace(", ", "_").replace(" ", "_")
    joblib.dump(model, f'/Users/tylerbrenner/Documents/Projects/Housing/rent_model_{safe}.pkl')
    forecast.to_csv(f'/Users/tylerbrenner/Documents/Projects/Housing/forecast_{safe}.csv', index=False)

    print(f"Saved {metro_name} model & forecast")
    return forecast

# Run
for metro in df['RegionName'].unique():
    train_and_forecast_metro(df[df['RegionName'] == metro].copy(), metro)

# Save CV metrics for app.py
with open('/Users/tylerbrenner/Documents/Projects/Housing/cv_metrics.json', 'w') as f:
    json.dump(cv_results, f, indent=4)

print("\nAll done! Re-run `streamlit run app.py` to see the new accuracy numbers.")