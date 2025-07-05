# Monthly return calculation per stock. Also saves price and value of the stock at the end of the month. 
# Note that this takes a long time and a lot of RAM to run.
#
# Required files:
#   -'cleaned_stock_data_returns.csv' file created by 'data_prep.py' 

import pandas as pd

# Specifying types saves a lot of RAM
dtypes = {
    'PERMNO': 'int32',
    'date': 'str',
    'PRC': 'float32',
    'RET': 'float32',
    'value': 'float32'
}
# Load cleaned stock data
df = pd.read_csv('cleaned_stock_data_returns.csv', usecols=list(dtypes.keys()), dtype=dtypes)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Calculates return for a stock in a given month and takes the last price and value in that month
def process_monthly_group(group):
    # Sort data 
    group = group.sort_values('date')

    # Calculate return that month
    monthly_ret = (1 + group['RET']).prod() - 1

    # Get final price and value of the stock that month
    last_day = group.iloc[-1]
    price = last_day['PRC']
    value = last_day['value']

    return pd.Series({
        'RET_monthly': monthly_ret,
        'last_price': price,
        'last_value': value
    })

# Group data by stock and by month and then and apply the function to each group
monthly_data = df.groupby(['PERMNO', df['date'].dt.to_period('M')]).apply(process_monthly_group)
monthly_data.reset_index(inplace=True)

# Bring date to the correct format
monthly_data['date'] = monthly_data['date'].dt.to_timestamp(how='end')
monthly_data['date'] = monthly_data['date'].dt.strftime('%Y-%m-%d')

# Save results
monthly_data.to_csv('processed_monthly_stock_data.csv', index=False)
