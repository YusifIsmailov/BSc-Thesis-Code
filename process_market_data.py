# Market data processing.
#
# Required files: 
#   -'market_return_daily.csv' contains value-weighted daily market returns from CRSP with required column: vwretd 
#   -'ff3_monthly.csv' contains monthly data for FF3 factors (mktrf, smb, hml), the momentum factor (umd) and the risk-free rate (rf). 

import pandas as pd
import numpy as np

# Load market data
df = pd.read_csv('market_return_daily.csv')
df['caldt'] = pd.to_datetime(df['caldt'], format='%d/%m/%Y')
df.set_index('caldt', inplace=True)
df['vwretd'] = pd.to_numeric(df['vwretd'], errors='coerce')
df.dropna(subset=['vwretd'], inplace=True)

# Calculate monthly total market return
df['vwretd_plus_1'] = df['vwretd'] + 1
monthly_market_returns = df['vwretd_plus_1'].resample('MS').prod() - 1

# Load ff3 data
ff_df = pd.read_csv('ff3_monthly.csv')
ff_df['date'] = pd.to_datetime(ff_df['dateff'], format='%d/%m/%Y')
# Align date to month start to match
ff_df['date'] = ff_df['date'].dt.to_period('M').dt.to_timestamp()
ff_df.set_index('date', inplace=True)

# Calculate monthly excess return
combined_monthly_df = pd.concat([monthly_market_returns.rename('mkt_ret'), ff_df['rf']], axis=1, join='inner')
monthly_excess_ret = combined_monthly_df['mkt_ret'] - combined_monthly_df['rf']

# The factor to be compounded is (1 + monthly_excess_return)
monthly_returns_plus_1 = monthly_excess_ret + 1

# Function to calculate forward returns for some horizon
def calculate_forward_returns(horizon):
    returns = []

    for date in monthly_returns_plus_1.index:
        start_date = date + pd.DateOffset(months=1)
        end_date = date + pd.DateOffset(months=horizon)
        
        if end_date > monthly_returns_plus_1.index.max():
            total_return = np.nan
        else:
            # Compound monthly returns
            total_return = monthly_returns_plus_1.loc[start_date:end_date].prod() - 1
        
        returns.append({'date': date, f'market_{horizon}m_ret': total_return})
    
    return pd.DataFrame(returns).set_index('date')

# Calculate forward returns
ret_1m_df = calculate_forward_returns(1)
ret_12m_df = calculate_forward_returns(12)
ret_36m_df = calculate_forward_returns(36)
ret_60m_df = calculate_forward_returns(60)

# Calculate realized volatility
df['vwretd_sq'] = df['vwretd']**2
rv = np.sqrt(df['vwretd_sq'].resample('MS').sum())
rv.name = 'market_rv'
market_df = rv.to_frame()

# Join all metrics
market_df = market_df.join(ret_1m_df, how='outer')
market_df = market_df.join(ret_12m_df, how='outer')
market_df = market_df.join(ret_36m_df, how='outer')
market_df = market_df.join(ret_60m_df, how='outer')

# Create date column
market_df.reset_index(inplace=True, names='date')

# Save to csv
market_df.to_csv('market_metrics.csv', index=False)