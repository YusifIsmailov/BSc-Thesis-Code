# Monthly return calculation per sector. Also saves price and value of stocks at the end of the month. 
# Note that this takes a long time and a lot of RAM to run.
#
# Required files:
#   -'cleaned_sector_stock_data_returns.csv' file created by 'sector_data_prep.py' 

import pandas as pd
import numpy as np

# Load data
daily_stock_data = pd.read_csv("cleaned_sector_stock_data_returns.csv")
daily_stock_data['date'] = pd.to_datetime(daily_stock_data['date'])
daily_stock_data['year_month'] = daily_stock_data['date'].dt.to_period('M')

# Calculate monthly returns for each stock
monthly_ret = daily_stock_data.groupby(['PERMNO', 'year_month'])['RET'].apply(
    lambda x: (1 + x).prod() - 1
).reset_index(name='RET_monthly')

# Get end-of-month stock characteristics  (sector, price, value)
monthly_chars = daily_stock_data.sort_values('date').groupby(['PERMNO', 'year_month']).last().reset_index()

# Merge monthly returns with characteristics
monthly_data = pd.merge(
    monthly_ret,
    monthly_chars[['PERMNO', 'year_month', 'ff5_sector', 'value']],
    on=['PERMNO', 'year_month'],
    how='inner'
)

# Function to calculate value-weighted returns for a group
def vw_return(group):
    return np.average(group['RET_monthly'], weights=group['value'])

# Calculate value-weighted returns for each sector for each month
sector_returns = monthly_data.groupby(['year_month', 'ff5_sector']).apply(vw_return, include_groups=False).unstack()

# Convert index back to date
sector_returns.index = sector_returns.index.to_timestamp()
sector_returns.index.name = 'date'

# Save results
sector_returns.to_csv("monthly_sector_returns.csv")
