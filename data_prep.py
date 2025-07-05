# Initial data processing file. 
# Note that this takes a long time and a lot of RAM to run.
#
# Required files:
#   -'stock_data.csv' contains daily stock data from CRSP with required columns: PERMNO, date, PRC, RET, SHROUT. 
#   -'ff3_daily.csv' contains daily data for FF3 factors (mktrf, smb, hml), the momentum factor (umd) and the risk-free rate (rf). 

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Specifying types saves a lot of RAM
dtypes = {
    'PERMNO': 'int32',
    'date': 'str',     
    'PRC': 'float32',
    'RET': 'object',    
    'SHROUT': 'float32'
}

# Load stock data
df = pd.read_csv('stock_data.csv', usecols=list(dtypes.keys()), dtype=dtypes)
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df['RET'] = pd.to_numeric(df['RET'], errors='coerce')

# Create value column and then delete shares column
df['value'] = df['PRC'].abs() * df['SHROUT']
df.drop(columns=['SHROUT'], inplace=True)

# Drop nan rows
df.dropna(subset=['RET', 'value', 'PRC'], inplace=True)

# Save returns data
df.to_csv('cleaned_stock_data_returns.csv', index=False)

# Load FF data
dtypes = {
    'date': 'str',
    'mktrf': 'float32',     
    'smb': 'float32',
    'hml': 'float32',    
    'rf': 'float32'
}
ff_data = pd.read_csv('ff3_daily.csv', usecols=list(dtypes.keys()), dtype=dtypes)
ff_data['date'] = pd.to_datetime(ff_data['date'], format='%d/%m/%Y')
ff_data.dropna(subset=['rf', 'mktrf', 'smb', 'hml'], inplace=True)
ff_data.set_index('date', inplace=True)

# Calculate daily FF3 Residuals for each stock using monthly regressions
i = 0
def get_ff3_residuals(monthly_stock_data):
    global i
    i += 1
    if i % 10000 == 1: 
      print(i)
    # Merge this month's stock data with the corresponding FF factors
    merged_month_data = pd.merge(
        monthly_stock_data,
        ff_data,     
        left_on='date',
        right_index=True,
        how='left'
    )
    merged_month_data.dropna(subset=['mktrf', 'smb', 'hml', 'RET', 'rf'], inplace=True)

    Y = merged_month_data['RET'] - merged_month_data['rf']
    X = merged_month_data[['mktrf', 'smb', 'hml']]
    X = sm.add_constant(X)

    model = sm.OLS(Y, X).fit()
    return model.resid

df['residual'] = df.groupby(['PERMNO', df['date'].dt.to_period('M')], group_keys=False).apply(get_ff3_residuals)

# Save ff3 residual data
df.drop(columns=['RET'], inplace=True)
df.to_csv('cleaned_stock_data_ff3residuals.csv', index=False)