# Creates a table to analyze whether there is a risk premium for exposure to RMG factor in the cross-section of stock returns. 
# First calculates stock-specific sensitivities to RMG and then creates portfolios based on ranking those betas. 
# Analysis can be run for a specific start and end date. 
# Contains the option to not rerun the beta calculation if you've already ran it once, since this is the most time consuming part. 
#
# Required files: 
#   -'gmr_betas.csv' if RECALCULATE_BETAS is set to False. 
#   -'gmr_factor.csv' file created by 'sector_RMG_factor.py'
#   -'tail_risk_factor.csv' file created by 'cross_section_single.py'
#   -'processed_monthly_stock_data.csv' file created by 'monthly_returns.py'
#   -'ff3_monthly.csv' contains monthly data for FF3 factors (mktrf, smb, hml), the momentum factor (umd) and the risk-free rate (rf). 
#   -'liq_factor.csv' contains monthly data for the liquidity factor (PS_VWF)

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Configuration
RECALCULATE_BETAS = False
BETA_FILE = "gmr_betas.csv" 

START_DATE = None
END_DATE = "2009-12-31"

# Function to calculate betas for a single stock
def calculate_betas_for_one_stock(stock_data_group):
    permno, df_stock = stock_data_group
    df_stock = df_stock.sort_index()

    # Stock needs at least 120 months of data to begin
    if len(df_stock) < 120:
        return pd.DataFrame()
    
    betas_list = []
    # Start calculating betas after first 120 months have passed
    for i in range(120, len(df_stock)):
        # The window is always the most recent 120 months.
        window_data = df_stock.iloc[i - 120 : i]
        # Drop nan's
        clean_window = window_data.dropna(subset=['excess_ret', 'gmr_factor'])
        # Only calculate if there are at least 36 observations remaining
        if len(clean_window) >= 36:
            Y = clean_window['excess_ret']
            X = sm.add_constant(clean_window['gmr_factor'])
            model = sm.OLS(Y, X).fit()
            betas_list.append({'formation_date': window_data.index[-1], 'beta': model.params.get('gmr_factor')})
            
    result_df = pd.DataFrame(betas_list)
    result_df['PERMNO'] = permno
    return result_df

# Function to form portfolios based on beta quintiles
def form_portfolios(holding_period, weighting_scheme):
    portfolio_returns = []

    # Go through every formation date
    for form_date in sorted(beta_df['formation_date'].unique()):
        betas_at_date = beta_df[beta_df['formation_date'] == form_date]
        
        # Merge with financials and apply price filter
        financials = stock_data[stock_data['date'] == form_date].copy()
        financials.rename(columns={'date': 'formation_date'}, inplace=True)
        eligible_stocks = pd.merge(betas_at_date, financials, on=['PERMNO', 'formation_date'], how='inner')
        eligible_stocks = eligible_stocks[eligible_stocks['last_price'].abs() >= 5].copy()
            
        # Create quintiles
        eligible_stocks['quintile'] = pd.qcut(eligible_stocks['beta'].rank(method='first'), 5, labels=False, duplicates='drop') + 1

        # Calculate the monthly return of the portfolio for each month it is held
        for h in range(1, holding_period + 1):
            return_date = form_date + pd.DateOffset(months=h)
            future_returns_all = stock_data[stock_data['date'] == return_date]
            quintile_returns = pd.merge(eligible_stocks[['PERMNO', 'quintile', 'last_value']], future_returns_all[['PERMNO', 'RET_monthly']], on='PERMNO')
            
            if quintile_returns.empty: continue

            if weighting_scheme == 'ew':
                monthly_returns = quintile_returns.groupby('quintile')['RET_monthly'].mean()
            elif weighting_scheme == 'vw':
                monthly_returns = quintile_returns.groupby('quintile')[['RET_monthly', 'last_value']].apply(
                    lambda x: np.average(x['RET_monthly'], weights=x['last_value']))
            for q, ret in monthly_returns.items():
                portfolio_returns.append({'formation_cohort': form_date, 'quintile': q, 'return_date': return_date, 'portfolio_return': ret})
                
    return pd.DataFrame(portfolio_returns)

# Function to analyze portfolio performance
def analyze_portfolio_performance(portfolio_returns, newey_west_lags):
    # Filter start and end date
    if START_DATE: 
        portfolio_returns = portfolio_returns[portfolio_returns['date'] >= pd.to_datetime(START_DATE)]
    if END_DATE: 
        portfolio_returns = portfolio_returns[portfolio_returns['date'] <= pd.to_datetime(END_DATE)]
    
    # Calculate performance for each quintile
    performance_rows = []
    for q in range(1, 6):
        quintile_returns = portfolio_returns[portfolio_returns['quintile'] == q].set_index('date')['portfolio_return']
        merged_data = pd.merge(quintile_returns, all_factors_df, left_index=True, right_index=True, how='inner')        
        Y_excess = (merged_data['portfolio_return'] - merged_data['rf']) * 100
        
        X_capm = sm.add_constant(merged_data['mktrf'])
        model_capm = sm.OLS(Y_excess, X_capm).fit(cov_type='HAC', cov_kwds={'maxlags': newey_west_lags, 'use_correction':True})
        X_ff3 = sm.add_constant(merged_data[['mktrf', 'smb', 'hml']])
        model_ff3 = sm.OLS(Y_excess, X_ff3).fit(cov_type='HAC', cov_kwds={'maxlags': newey_west_lags, 'use_correction':True})
        X_ff4 = sm.add_constant(merged_data[['mktrf', 'smb', 'hml', 'umd']])
        model_ff4 = sm.OLS(Y_excess, X_ff4).fit(cov_type='HAC', cov_kwds={'maxlags': newey_west_lags, 'use_correction':True})
        X_ff5 = sm.add_constant(merged_data[['mktrf', 'smb', 'hml', 'umd', 'liq']])
        model_ff5 = sm.OLS(Y_excess, X_ff5).fit(cov_type='HAC', cov_kwds={'maxlags': newey_west_lags, 'use_correction':True})
        X_ff6 = sm.add_constant(merged_data[['mktrf', 'smb', 'hml', 'umd', 'liq', 'trf']])
        model_ff6 = sm.OLS(Y_excess, X_ff6).fit(cov_type='HAC', cov_kwds={'maxlags': newey_west_lags, 'use_correction':True})

        performance_rows.append({
            'Portfolio': f'Q{q}',
            'Avg Return': merged_data['portfolio_return'].mean(),
            'CAPM Alpha': model_capm.params['const'],
            'FF3 Alpha': model_ff3.params['const'],
            'FF4 Alpha': model_ff4.params['const'],
            'FF5 Alpha': model_ff5.params['const'],
            'FF6 Alpha': model_ff6.params['const'],
        })

    # Calculate high minus low performance
    q_high = portfolio_returns[portfolio_returns['quintile'] == 5].set_index('date')['portfolio_return']
    q_low = portfolio_returns[portfolio_returns['quintile'] == 1].set_index('date')['portfolio_return']    
    spread_returns = 100 * (q_high - q_low)
    merged_spread_data = pd.merge(spread_returns.to_frame('spread'), all_factors_df, left_index=True, right_index=True, how='inner')
    
    Y_spread = merged_spread_data['spread']
    model_avg_s = sm.OLS(Y_spread, sm.add_constant(np.ones(len(Y_spread)))).fit(cov_type='HAC', cov_kwds={'maxlags': newey_west_lags, 'use_correction':True})
    model_capm_s = sm.OLS(Y_spread, sm.add_constant(merged_spread_data['mktrf'])).fit(cov_type='HAC', cov_kwds={'maxlags': newey_west_lags, 'use_correction':True})
    model_ff3_s = sm.OLS(Y_spread, sm.add_constant(merged_spread_data[['mktrf', 'smb', 'hml']])).fit(cov_type='HAC', cov_kwds={'maxlags': newey_west_lags, 'use_correction':True})
    model_ff4_s = sm.OLS(Y_spread, sm.add_constant(merged_spread_data[['mktrf', 'smb', 'hml', 'umd']])).fit(cov_type='HAC', cov_kwds={'maxlags': newey_west_lags, 'use_correction':True})
    model_ff5_s = sm.OLS(Y_spread, sm.add_constant(merged_spread_data[['mktrf', 'smb', 'hml', 'umd', 'liq']])).fit(cov_type='HAC', cov_kwds={'maxlags': newey_west_lags, 'use_correction':True})
    model_ff6_s = sm.OLS(Y_spread, sm.add_constant(merged_spread_data[['mktrf', 'smb', 'hml', 'umd', 'liq', 'trf']])).fit(cov_type='HAC', cov_kwds={'maxlags': newey_west_lags, 'use_correction':True})

    performance_rows.append({
        'Portfolio': 'Q5-Q1',
        'Avg Return': model_avg_s.params['const'],
        'Avg Return t-stat': model_avg_s.tvalues['const'],
        'CAPM Alpha': model_capm_s.params['const'],
        'CAPM Alpha t-stat': model_capm_s.tvalues['const'],
        'FF3 Alpha': model_ff3_s.params['const'],
        'FF3 Alpha t-stat': model_ff3_s.tvalues['const'],
        'FF4 Alpha': model_ff4_s.params['const'],
        'FF4 Alpha t-stat': model_ff4_s.tvalues['const'],
        'FF5 Alpha': model_ff5_s.params['const'],
        'FF5 Alpha t-stat': model_ff5_s.tvalues['const'],
        'FF6 Alpha': model_ff6_s.params['const'],
        'FF6 Alpha t-stat': model_ff6_s.tvalues['const']
    })
    
    return pd.DataFrame(performance_rows).set_index('Portfolio')

# Load data and align all dates to the end of the month
gmr_df = pd.read_csv("gmr_factor.csv", index_col='date', parse_dates=True)
gmr_df.index = gmr_df.index + pd.offsets.MonthEnd(0)

stock_data = pd.read_csv("processed_monthly_stock_data.csv")
stock_data['date'] = pd.to_datetime(stock_data['date']) + pd.offsets.MonthEnd(0)

ff_df = pd.read_csv("ff3_monthly.csv", usecols=['dateff', 'mktrf', 'smb', 'hml', 'rf', 'umd'])
ff_df.rename(columns={'dateff': 'date'}, inplace=True)
ff_df['date'] = pd.to_datetime(ff_df['date'], format='%d/%m/%Y') + pd.offsets.MonthEnd(0)

liq_df = pd.read_csv("liq_factor.csv", usecols=['DATE', 'PS_VWF'], dtype={'PS_VWF': 'float32'})
liq_df.rename(columns={'DATE': 'date', 'PS_VWF': 'liq'}, inplace=True)
liq_df['date'] = pd.to_datetime(liq_df['date'], format='%d/%m/%Y') + pd.offsets.MonthEnd(0)

trf_df = pd.read_csv("tail_risk_factor.csv", index_col='date', parse_dates=True)
trf_df.index = trf_df.index + pd.offsets.MonthEnd(0)

# Merge all factors
all_factors_df = ff_df.merge(liq_df, on='date', how='inner').merge(trf_df, on='date', how='inner').set_index('date')

# Calculate tail risk sensitivities
if RECALCULATE_BETAS:
    # Merge relevant data
    beta_input_df = stock_data.merge(gmr_df, on='date', how='left').merge(all_factors_df[['rf']].reset_index(), on='date', how='left')
    beta_input_df['excess_ret'] = beta_input_df['RET_monthly'] - beta_input_df['rf']
    beta_input_df = beta_input_df.set_index('date')
    
    # Group by stock
    stock_data_groups = beta_input_df.groupby('PERMNO')
    results_list = []    
    for permno, stock_df in stock_data_groups:
        result_df = calculate_betas_for_one_stock((permno, stock_df))
        results_list.append(result_df)
    
    beta_df = pd.concat(results_list, ignore_index=True).dropna()
    beta_df.to_csv(BETA_FILE, index=False)
    print(f"Saved calculated GmR betas to {BETA_FILE}")
else:
    print(f"Loading pre-calculated GmR betas from {BETA_FILE}")
    beta_df = pd.read_csv(BETA_FILE)
beta_df['formation_date'] = pd.to_datetime(beta_df['formation_date'])

# Form portfolios for equal and value weights
for weighting in ['ew', 'vw']:
    # Calculate the monthly return for each portfolio for holding period 12 months
    portfolio_returns_12m = form_portfolios(holding_period=12, weighting_scheme=weighting)
    # The reported monthly return is the average return of all portfolios active that month. 
    portfolio_returns_12m = portfolio_returns_12m.groupby(['return_date', 'quintile'])['portfolio_return'].mean().reset_index().rename(columns={'return_date':'date'})
    # Calculate performance metrics
    panel_a_results = analyze_portfolio_performance(portfolio_returns_12m, newey_west_lags=12)
    # Save to csv
    panel_a_results.to_csv(f"gmr_beta_12m_{weighting}.csv")
    
    # Calculate the monthly return for each portfolio for holding period 1 month
    portfolio_returns_1m = form_portfolios(holding_period=1, weighting_scheme=weighting)
    # Since there is only one portfolio active each month, the averaging step for the 12-months isn't needed here.
    portfolio_returns_1m = portfolio_returns_1m.rename(columns={'return_date':'date'}) 
    # Calculate performance metrics
    panel_b_results = analyze_portfolio_performance(portfolio_returns_1m, newey_west_lags=1)
    # Save to csv
    panel_b_results.to_csv(f"gmr_beta_1m_{weighting}.csv")
