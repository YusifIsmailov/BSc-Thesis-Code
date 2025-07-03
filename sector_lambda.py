# Calculate and plot tail risk estimates.
#
# Required files: 
#   -'cleaned_sector_stock_data_ff3residuals.csv' file created by 'sector_data_prep.py'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Initialize DataFrames
ff5_lambda_df = pd.DataFrame()
ff5_lambda = pd.DataFrame()

# Load sector data
value_column = 'residual'
dtypes = {
    'PERMNO': 'int32',
    'date': 'str',     
    'RET': 'object',    
    'ff5_sector': 'str',
    'residual': 'float32'
}
df = pd.read_csv('cleaned_sector_stock_data_ff3residuals.csv', usecols=list(dtypes.keys()), dtype=dtypes)
df['date'] = pd.to_datetime(df['date'])
df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
required_cols = [value_column, 'ff5_sector', 'date']
df.dropna(subset=required_cols, inplace=True)
# Create year-month column for grouping
df['year_month'] = df['date'].dt.to_period('M')

# Tail risk estimation using the Hill estimator
def calculate_sector_tail_risk(series_data):
    daily_values = series_data['residual'].values
    ut = np.percentile(daily_values, 5)
    exceedances = daily_values[daily_values < ut]
    ratio = exceedances / ut
    lambda_ = (1 / len(ratio)) * np.sum(np.log(ratio))
    return {'lambda': lambda_}

def process_sector_lambdas():
    sector_lambda_list = []
    for sector_name in df['ff5_sector'].unique():
        sector_data = df[df['ff5_sector'] == sector_name]
        
        # Calculate tail risk for this sector
        monthly_series = sector_data.groupby('year_month').apply(
            lambda x: calculate_sector_tail_risk(x), include_groups=False)
            
        # Turn it into a dataframe    
        monthly_df = monthly_series.apply(pd.Series).reset_index()
        monthly_df['sector_name'] = sector_name
        sector_lambda_list.append(monthly_df)
    
    # Combine all sector tail risk metrics together in rows for each month
    final_df_long = pd.concat(sector_lambda_list, ignore_index=True)
    final_df_long['date'] = final_df_long['year_month'].dt.to_timestamp()
    
    final_pivot_df = final_df_long.pivot(index='date', columns='sector_name', values='lambda')
    final_pivot_df.dropna(axis=1, how='all', inplace=True)
    final_pivot_df.dropna(axis=0, how='any', inplace=True)
    
    return final_pivot_df

ff5_lambda = process_sector_lambdas()

# Plots
print("\n--- Generating Plot for Sector Tail Risk ---")

# Full sector names for legend
legend_name_map = {
    'Cnsmr': 'Consumer',
    'HiTec': 'High-Tech',
    'Hlth': 'Healthcare',
    'Manuf': 'Manufacturing',
    'Other': 'Other'
}

# Periods where US was in recession
recession_periods = [
    ('1960-04-01', '1961-02-01'),
    ('1969-12-01', '1970-11-01'),
    ('1973-11-01', '1975-03-01'),
    ('1980-01-01', '1980-07-01'),
    ('1981-07-01', '1982-11-01'),
    ('1990-07-01', '1991-03-01'),
    ('2001-03-01', '2001-11-01'),
    ('2007-12-01', '2009-06-01'),
    ('2020-02-01', '2020-04-01')
]

fig, ax = plt.subplots(figsize=(14, 8))

for sector in ff5_lambda.columns:
    ax.plot(ff5_lambda.index, ff5_lambda[sector], label=legend_name_map.get(sector))

# Add recession shading
for start, end in recession_periods:
    ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='lightsteelblue', alpha=0.5, zorder=0)

# Formatting the plot
ax.set_title('Estimated Sector Tail Risk Measures')
ax.set_xlabel('Date')
ax.set_ylabel('Tail Risk')
ax.legend(title='Sectors', loc='upper left')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Makes axis spacing look prettier
num_months_total = len(ff5_lambda.index)
month_interval = max(1, int(num_months_total / 12)) 
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("ff5_sector_lambda_ff3.png")
plt.close()
print(f"Plot saved to ff5_sector_lambda_ff3.png")

# Save data
ff5_lambda.index.name = 'date'
ff5_lambda.to_csv("sector_lambda_estimates.csv")
