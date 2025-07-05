# Calculate the RMG factor which goes long on most net receiving sector and short on most net transmitting sector. 
#
# Required files:
#   -'monthly_sector_returns.csv' file created by 'sector_monthly_returns.py' 
#   -'net_spillovers.csv' file created by 'sector_spillovers.py'

import pandas as pd

# Load monthly sector returns
sector_returns = pd.read_csv("monthly_sector_returns.csv", index_col='date', parse_dates=True)
# Mapping sectors to full names
sector_name_map = {
    'Cnsmr': 'Consumer',
    'Manuf': 'Manufacturing',
    'HiTec': 'High-Tech',
    'Hlth': 'Healthcare',
    'Other': 'Other'
}
sector_returns.rename(columns=sector_name_map, inplace=True)

# Load spillover data
net_spillovers = pd.read_csv('net_spillovers.csv', index_col='date', parse_dates=True)

ls_returns = []
# Iterate through formation dates
for form_date in net_spillovers.index:
    # We use next months return for this month's spillover metrics
    return_date = form_date + pd.DateOffset(months=1)

    # Skip if there's no data. Only happens on the last formation date
    if return_date not in sector_returns.index:
        continue

    # Rank sectors based on their net spillover
    ranked_sectors = net_spillovers.loc[form_date].sort_values(ascending=False)

    # Long net receiver. Short net transmitter. 
    long_portfolio = ranked_sectors.head(1).index
    short_portfolio = ranked_sectors.tail(1).index

    long_ret = sector_returns.loc[return_date, long_portfolio].mean()
    short_ret = sector_returns.loc[return_date, short_portfolio].mean()
    # Save Receiver-Minus-Giver return
    ls_returns.append({'date': return_date, 'ls_return': long_ret - short_ret})

# Save results
ls_df = pd.DataFrame(ls_returns).set_index('date')
ls_df.dropna(inplace=True)
ls_df.rename(columns={'ls_return': 'gmr_factor'}, inplace=True)
ls_df.to_csv("gmr_factor.csv")