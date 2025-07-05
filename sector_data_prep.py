# Initial data processing for sector analysis. 
# Note that this takes a long time and a lot of RAM to run.
#
# Required files:
#   -'stock_data.csv' contains daily stock data from CRSP with required columns: PERMNO, date, PRC, RET, SHROUT. 
#   -'ff3_daily.csv' contains daily data for FF3 factors (mktrf, smb, hml), the momentum factor (umd) and the risk-free rate (rf). 

import pandas as pd
import statsmodels.api as sm

# Specifying types saves a lot of RAM
dtypes = {
    'PERMNO': 'int32',
    'date': 'str',     
    'PRC': 'float32',
    'RET': 'object',    
    'SHROUT': 'float32',
    'SICCD': 'object' 
}

# Load stock data
df = pd.read_csv('stock_data.csv', usecols=list(dtypes.keys()), dtype=dtypes)
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df['RET'] = pd.to_numeric(df['RET'], errors='coerce')
df['SICCD'] = pd.to_numeric(df['SICCD'], errors='coerce')

# Create value column and drop shares column
df['value'] = df['PRC'].abs() * df['SHROUT']
df.drop(columns=['SHROUT'], inplace=True)

# Drop nan rows
df.dropna(subset=['RET', 'SICCD', 'value'], inplace=True)

# Set as integer
df['SICCD'] = df['SICCD'].astype(int)

# Fama-French 5-sector classification function
def classify_ff5_sector(sic_code):
    sic = int(sic_code)
    # Cnsmr
    if (100 <= sic <= 999) or (2000 <= sic <= 2399) or (2700 <= sic <= 2749) or \
       (2770 <= sic <= 2799) or (3100 <= sic <= 3199) or (3940 <= sic <= 3989) or \
       (2500 <= sic <= 2519) or (2590 <= sic <= 2599) or (3630 <= sic <= 3659) or \
       (3710 <= sic <= 3711) or (sic == 3714) or (sic == 3716) or (sic == 3751) or \
       (sic == 3792) or (3900 <= sic <= 3939) or (3990 <= sic <= 3999) or \
       (5000 <= sic <= 5999) or (7200 <= sic <= 7299) or (7600 <= sic <= 7699):
        return 'Cnsmr'
    # Manuf
    if (2520 <= sic <= 2589) or (2600 <= sic <= 2699) or (2750 <= sic <= 2769) or \
       (2800 <= sic <= 2829) or (2840 <= sic <= 2899) or (3000 <= sic <= 3099) or \
       (3200 <= sic <= 3569) or (3580 <= sic <= 3629) or (3700 <= sic <= 3709) or \
       (sic == 3712) or (sic == 3713) or (sic == 3715) or (3717 <= sic <= 3749) or \
       (3752 <= sic <= 3791) or (3793 <= sic <= 3799) or (3830 <= sic <= 3839) or \
       (3860 <= sic <= 3899) or (1200 <= sic <= 1399) or (2900 <= sic <= 2999) or \
       (4900 <= sic <= 4949):
        return 'Manuf'
    # HiTec
    if (3570 <= sic <= 3579) or (3660 <= sic <= 3692) or (3694 <= sic <= 3699) or \
       (3810 <= sic <= 3829) or (7370 <= sic <= 7379) or (sic == 7391) or \
       (8730 <= sic <= 8734) or (4800 <= sic <= 4899):
        return 'HiTec'
    # Hlth
    if (2830 <= sic <= 2839) or (sic == 3693) or (3840 <= sic <= 3859) or \
       (8000 <= sic <= 8099):
        return 'Hlth'
    # Other
    return 'Other'

# Classify sectors
df['ff5_sector'] = df['SICCD'].apply(classify_ff5_sector)


# Load FF data
ff_dtypes = {
    'date': 'str', 
    'mktrf': 'float32', 
    'smb': 'float32', 
    'hml': 'float32', 
    'rf': 'float32'
}
ff_data = pd.read_csv('ff3_daily.csv', usecols=list(ff_dtypes.keys()), dtype=ff_dtypes)
ff_data['date'] = pd.to_datetime(ff_data['date'], format='%d/%m/%Y')
ff_data.dropna(inplace=True)
ff_data.set_index('date', inplace=True)

# Calculate daily FF3 Residuals for each stock
def get_ff3_residuals(stock_data_group):
    merged_data = pd.merge(stock_data_group, ff_data, left_on='date', right_index=True, how='left')
    merged_data.dropna(subset=['RET', 'rf', 'mktrf', 'smb', 'hml'], inplace=True)
    
    Y = merged_data['RET'] - merged_data['rf']
    X = merged_data[['mktrf', 'smb', 'hml']]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return pd.Series(model.resid, index=merged_data.index)

df['residual'] = df.groupby('PERMNO', group_keys=False).apply(get_ff3_residuals)


# Save datasets
print("Saving final datasets...")

# Save returns version
df_returns = df.drop(columns=['residual'], errors='ignore').copy()
df_returns.to_csv('cleaned_sector_stock_data_returns.csv', index=False)

# Save residuals version
df.dropna(subset=['residual'], inplace=True)
df.to_csv('cleaned_sector_stock_data_ff3residuals.csv', index=False)
