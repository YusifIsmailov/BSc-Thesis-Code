# Calculate alpha of the RMG factor for some period of time using Fama-Macbeth regressions.
# Asset pricing models tested: CAPM, FF3, FF4, FF4 + Liquidity, FF4 + Liquidity + Tail Risk Factor
#
# Required files:
#   -'gmr_factor.csv' file created by 'sector_RMG_factor.py'
#   -'tail_risk_factor.csv' file created by 'cross_section_single.py'
#   -'ff3_monthly.csv' contains monthly data for FF3 factors (mktrf, smb, hml), the momentum factor (umd) and the risk-free rate (rf).
#   -'liq_factor.csv' contains monthly data for the liquidity factor (PS_VWF).

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Config
START_DATE = None
END_DATE = '2024-12-31'

# Load data and align dates to month end
gmr_df = pd.read_csv("gmr_factor.csv", index_col='date', parse_dates=True)
gmr_df.index = gmr_df.index + pd.offsets.MonthEnd(0)

# Load FF4 data
dtypes = {'dateff': 'str', 'mktrf': 'float32', 'smb': 'float32', 'hml': 'float32', 'rf': 'float32', 'umd': 'float32' }
ff_df = pd.read_csv("ff3_monthly.csv", usecols=list(dtypes.keys()), dtype=dtypes)
ff_df.rename(columns={'dateff': 'date'}, inplace=True)
ff_df['date'] = pd.to_datetime(ff_df['date'], format='%d/%m/%Y') + pd.offsets.MonthEnd(0)

# Load Liquidity factor data
liq_df = pd.read_csv("liq_factor.csv", usecols=['DATE', 'PS_VWF'], dtype={'PS_VWF': 'float32'})
liq_df.rename(columns={'DATE': 'date', 'PS_VWF': 'liq'}, inplace=True)
liq_df['date'] = pd.to_datetime(liq_df['date'], format='%d/%m/%Y') + pd.offsets.MonthEnd(0)

# Load Tail Risk factor data
trf_df = pd.read_csv("tail_risk_factor.csv", index_col='date', parse_dates=True)
trf_df.index = trf_df.index + pd.offsets.MonthEnd(0)

# Merge all factors
factors_df = ff_df.set_index('date') \
    .join(liq_df.set_index('date'), how='inner') \
    .join(trf_df, how='inner') \
    .join(gmr_df, how='inner') \

# Filter data
if START_DATE:
    factors_df = factors_df[factors_df.index >= pd.to_datetime(START_DATE)]
if END_DATE:
    factors_df = factors_df[factors_df.index <= pd.to_datetime(END_DATE)]

# Dependent variable is monthly excess return 
Y = (factors_df['gmr_factor'] - factors_df['rf']) * 100

# Independent variables are other factors 
benchmark_cols = ['mktrf', 'smb', 'hml', 'umd', 'liq', 'trf']
X_data = factors_df[[col for col in benchmark_cols if col in factors_df.columns]] * 100

# Calculate average return and its t-stat
model_avg = sm.OLS(Y, sm.add_constant(np.ones(len(Y)))).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
print(f"Average Monthly Excess Return: {model_avg.params[0]:.2f}% (t-stat: {model_avg.tvalues[0]:.2f})")

# Models for Fama-Macbeth regressions
models_to_run = {
    'CAPM': ['mktrf'],
    'FF3': ['mktrf', 'smb', 'hml'],
    'FF4': ['mktrf', 'smb', 'hml', 'umd'],
    'FF4 + Liq': ['mktrf', 'smb', 'hml', 'umd', 'liq'],
    'FF4 + Liq + Tail Risk': ['mktrf', 'smb', 'hml', 'umd', 'liq', 'trf'],
}

# Run regression for each model
for name, factors in models_to_run.items():
    X = sm.add_constant(X_data[factors])
    model = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    print(f"{name} Alpha: {model.params['const']:.2f}% (t-stat: {model.tvalues['const']:.2f})")
