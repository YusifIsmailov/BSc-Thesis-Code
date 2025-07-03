# Calculate and plot tail risk estimates. Designed to work for both return based metric and ff3 residual based metric.
#
# Required files: 
#   -'cleaned_stock_data_ff3residuals.csv' OR 'cleaned_stock_data_returns.csv' files created by 'data_prep.py'
#   -'market_metrics.csv' file created by 'process_market_data.py'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm

# Set which tail risk metric you want to use
use_ff3 = True 
if use_ff3:
    input_file = 'cleaned_stock_data_ff3residuals.csv' 
    column = 'residual'
    suffix = "_residuals"
else:
    input_file = 'cleaned_stock_data_returns.csv'
    column = 'RET'
    suffix = "_returns" 

# Load stock data
df = pd.read_csv(input_file)
df['date'] = pd.to_datetime(df['date'])
df[column] = pd.to_numeric(df[column], errors='coerce')
df.dropna(subset=[column], inplace=True)

# Create year-month column for grouping
df['year_month'] = df['date'].dt.to_period('M')

# Tail risk estimation using the Hill estimator
monthly_lambda = []
for month, group_df in df.groupby('year_month'):
    daily_values = group_df[column].values
    ut = np.percentile(daily_values, 5)
    exceedances = daily_values[daily_values < ut]
    Kt = len(exceedances)
    ratio = exceedances / ut
    lambda_ = (1 / Kt) * np.sum(np.log(ratio))

    se_lambda = lambda_ / np.sqrt(Kt)
    z_score = 1.96
    ci_lower = lambda_ - z_score * se_lambda
    ci_upper = lambda_ + z_score * se_lambda

    monthly_lambda.append({
        'year_month': month,
        'lambda': lambda_, 
        'ut': ut,                       
        'ut_abs': np.abs(ut),
        'Kt': Kt,
        'se_lambda': se_lambda,
        'ci_lower_95': ci_lower,
        'ci_upper_95': ci_upper
    })

lambda_df = pd.DataFrame(monthly_lambda)
lambda_df['date'] = lambda_df['year_month'].dt.to_timestamp()

# Standardization
lambda_mean = lambda_df['lambda'].mean()
lambda_std_dev = lambda_df['lambda'].std()

lambda_df['lambda_std'] = (lambda_df['lambda'] - lambda_mean) / lambda_std_dev
lambda_df['ci_lower_95_std'] = (lambda_df['ci_lower_95'] - lambda_mean) / lambda_std_dev
lambda_df['ci_upper_95_std'] = (lambda_df['ci_upper_95'] - lambda_mean) / lambda_std_dev

lambda_df['ut_abs_std'] = (lambda_df['ut_abs'] - lambda_df['ut_abs'].mean()) / lambda_df['ut_abs'].std()

# Load and standardize market data 
market_metrics_df = pd.read_csv('market_metrics.csv')
market_metrics_df['date'] = pd.to_datetime(market_metrics_df['date'])

lambda_df = pd.merge(lambda_df, market_metrics_df, on='date', how='left')

lambda_df['market_36m_ret_std'] = (lambda_df['market_36m_ret'] - lambda_df['market_36m_ret'].mean()) / lambda_df['market_36m_ret'].std()
lambda_df['market_rv_std'] = (lambda_df['market_rv'] - lambda_df['market_rv'].mean()) / lambda_df['market_rv'].std()

# These are the periods where the US was in recession
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

# Makes the spacing of the grid look nicer
num_months_total = len(lambda_df['date'].dt.to_period('M').unique())
month_interval = max(1, int(num_months_total / 12))

# Plot 1: lambda vs market return 
plt.figure(figsize=(14, 7))
# Shade recession periods in blue
for start, end in recession_periods:
    plt.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='lightsteelblue', alpha=0.3, zorder=0)

# Plot the market return line
plt.plot(lambda_df['date'], lambda_df['market_36m_ret_std'], color='black', linestyle='-', label="Forward 3-Year Market Return", zorder=4)

# Plot standardized lambda and confidence interval
plt.plot(lambda_df['date'], lambda_df['lambda_std'], color='grey', linestyle='--', label="Tail Risk", zorder=3)
plt.fill_between(
    lambda_df['date'],
    lambda_df['ci_lower_95_std'],
    lambda_df['ci_upper_95_std'],
    color='grey',
    alpha=0.2,
    label='95% CI for Tail Risk',
    zorder=2
)

plt.xlabel('Date')
plt.ylabel('Standardized Value')
plt.title('Standardized Tail Risk Measure with 95% Confidence Interval and Next 3-Year Market Return')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
plt.xticks(rotation=45)
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig(f'lambda{suffix}_vs_3yr_forward_market_ret.png')
plt.close()
print(f"Saved lambda{suffix}_vs_3yr_forward_market_ret.png")

# Plot 2: absolute threshold vs realized market volatility
plt.figure(figsize=(14, 7))

for start, end in recession_periods:
    plt.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='lightsteelblue', alpha=0.3, zorder=0)
    
plt.plot(lambda_df['date'], lambda_df['ut_abs_std'], color='black', linestyle='-', label="Abs. Tail Threshold")
plt.plot(lambda_df['date'], lambda_df['market_rv_std'], color='grey', linestyle='--', label="Realized Market Volatility")

plt.xlabel('Date')
plt.ylabel('Standardized Value')
plt.title('Std. Absolute Tail Threshold and Std. Realized Market Volatility')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
plt.xticks(rotation=45)
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig(f'ut_abs{suffix}_vs_market_rv.png')
plt.close()
print(f"Saved ut_abs{suffix}_vs_market_rv.png")

# Plot 3: unstandardized tail risk 
plt.figure(figsize=(14, 7))

for start, end in recession_periods:
    plt.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='lightsteelblue', alpha=0.3, zorder=0)

plt.plot(lambda_df['date'], lambda_df['lambda'], color='black', linestyle='-', label='Estimated Tail Risk', linewidth=1.5)
plt.fill_between(
    lambda_df['date'], 
    lambda_df['ci_lower_95'], 
    lambda_df['ci_upper_95'], 
    color='gray', 
    alpha=0.4, 
    label='95% Confidence Interval'
)

plt.xlabel('Date')
plt.ylabel('Tail Risk')
plt.title(f'Estimated Tail Risk Measure with 95% Confidence Interval')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
plt.xticks(rotation=45)
plt.legend(loc="upper right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f'lambda_unstandardized_with_ci{suffix}.png')
plt.close()
print(f"Saved lambda_unstandardized_with_ci{suffix}.png")

# Save tail risk metrics
lambda_df.to_csv('lambda_estimates.csv', index=False)
print("Saved lambda_estimates.csv")