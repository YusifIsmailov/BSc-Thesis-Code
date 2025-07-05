# Replicates table 5 from Kelly and Jiang. Calcultes firm specific metrics: size, idiosyncratic volatility, downside beta, coskewness. 
# These are used together with previously calculated tail risk betas to create double sorted portfolios to calculate alpha 
# of high-minus-low tail risk strategy within different quartiles of the firm specific metric.
# Analysis can be run for a specific start and end date. 
# Contains the option to not rerun the secondary metric calculations if you've already ran it once, since this is a time consuming part. 
#
# Required files: 
#   -'idio_vol.csv' if RECALCULATE_METRICS is set to False.
#   -'dbeta_coskew.csv' if RECALCULATE_METRICS is set to False.
#   -'tail_risk_betas.csv' file created by 'cross_section_single.py'
#   -'cleaned_stock_data_ff3residuals.csv' file created by 'data_prep.py'
#   -'processed_monthly_stock_data.csv' file created by 'monthly_returns.py'
#   -'ff3_monthly.csv' contains monthly data for FF3 factors (mktrf, smb, hml), the momentum factor (umd) and the risk-free rate (rf). 
#   -'liq_factor.csv' contains monthly data for the liquidity factor (PS_VWF)

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Configuration
RECALCULATE_METRICS = True
IDIO_VOL_FILE = "idio_vol.csv"
DBETA_COSKEW_FILE = "dbeta_coskew.csv"
START_DATE = None
END_DATE = '2024-12-31'

# Calculate idiosyncratic volatility using daily ff3 residuals
def calculate_idio_vol():
    print("Calculating Idiosyncratic Volatility")
    dtypes = {'PERMNO': 'int32', 'date': 'str', 'residual': 'float32'}
    daily_residuals = pd.read_csv("cleaned_stock_data_ff3residuals.csv", usecols=list(dtypes.keys()), dtype=dtypes)
    daily_residuals['date'] = pd.to_datetime(daily_residuals['date'])
    daily_residuals['year_month'] = daily_residuals['date'].dt.to_period('M')

    # Idiosyncratic volatility is standard deviation of ff3 residuals per month
    idio_vol_calc = daily_residuals.groupby(['PERMNO', 'year_month'])['residual'].std().reset_index()
    idio_vol_calc.rename(columns={'residual': 'idio_vol'}, inplace=True)
    idio_vol_calc['formation_date'] = (idio_vol_calc['year_month'].dt.to_timestamp() + pd.offsets.MonthEnd(0))
    
    idio_vol_df = idio_vol_calc[['PERMNO', 'formation_date', 'idio_vol']].dropna()
    idio_vol_df.to_csv(IDIO_VOL_FILE, index=False)
    print(f"Idiosyncratic volatility saved to {IDIO_VOL_FILE}")
    return idio_vol_df

# Calculate downside beta and coskewness
def calculate_dbeta_coskew():
    print("Calculating Downside Beta and Coskewness")
    # Merge relevant data
    data = pd.merge(stock_data[['PERMNO', 'date', 'RET_monthly']], lambda_df[['market_ret']],
                    left_on='date', right_index=True, how='left').dropna()
    data.sort_values(['PERMNO', 'date'], inplace=True)

    results = []
    # Group by stock
    for permno, permno_data in data.groupby('PERMNO'):
        permno_data = permno_data.set_index('date')
        if len(permno_data) < 120: continue
        
        # 120 month rolling window calculation
        for i in range(120, len(permno_data)):
            window_data = permno_data.iloc[i - 120 : i]
            # Regress stock return on market return in months where market was negative
            downside_market_months = window_data[window_data['market_ret'] < 0]
            Y_dbeta = downside_market_months['RET_monthly']
            X_dbeta = sm.add_constant(downside_market_months['market_ret'])
            d_beta = sm.OLS(Y_dbeta, X_dbeta, missing='drop').fit().params.get('market_ret')

            # Regress stock return on squaredmarket return
            market_ret_sq = window_data['market_ret']**2
            Y_coskew = window_data['RET_monthly']
            X_coskew = sm.add_constant(market_ret_sq)
            coskew = sm.OLS(Y_coskew, X_coskew, missing='drop').fit().params.get('market_ret')
            
            results.append({
                'PERMNO': permno, 'formation_date': window_data.index[-1],
                'downside_beta': d_beta, 'coskewness': coskew
            })
            
    dbeta_coskew_df = pd.DataFrame(results).dropna(subset=['downside_beta', 'coskewness'], how='all')
    dbeta_coskew_df.to_csv(DBETA_COSKEW_FILE, index=False)
    print(f"Downside beta and coskewness saved to {DBETA_COSKEW_FILE}")
    return dbeta_coskew_df

# Form double-sorted portfolios and calculate their returns
def form_portfolios(all_data, holding_period, weighting_scheme, secondary_col):
    all_portfolio_monthly_returns = []
    
    for form_date in sorted(all_data['formation_date'].unique()):
        eligible_stocks = all_data[all_data['formation_date'] == form_date].copy()
        # Create quartiles
        eligible_stocks['beta_quartile'] = pd.qcut(eligible_stocks['beta'].rank(method='first'), 4, labels=False, duplicates='drop') + 1
        eligible_stocks['secondary_quartile'] = pd.qcut(eligible_stocks[secondary_col].rank(method='first'), 4, labels=False, duplicates='drop') + 1
        
        # Loop through high and low beta quartiles and all secondary quartiles
        for bq in [1, 4]:
            for sq in range(1, 5):
                portfolio_stocks = eligible_stocks[(eligible_stocks['beta_quartile'] == bq) & (eligible_stocks['secondary_quartile'] == sq)]
                
                # Calculate the monthly return of the portfolio for each month it is held
                for h in range(holding_period):
                    return_date = (form_date + pd.DateOffset(months = 1 + h))
                    future_returns = stock_data.loc[(stock_data['date'] == return_date) & 
                                                            (stock_data['PERMNO'].isin(portfolio_stocks['PERMNO']))]
                    if future_returns.empty: continue

                    if weighting_scheme == 'ew':
                        portfolio_return = future_returns['RET_monthly'].mean()
                    elif weighting_scheme == 'vw':
                        merged_for_vw = pd.merge(future_returns, portfolio_stocks[['PERMNO', 'last_value']], on='PERMNO')
                        portfolio_return = np.average(merged_for_vw['RET_monthly'], weights=merged_for_vw['last_value'])
                    
                    all_portfolio_monthly_returns.append({
                        'beta_q': bq, 'secondary_q': sq, 
                        'return_date': return_date, 
                        'portfolio_return': portfolio_return
                    })
    return pd.DataFrame(all_portfolio_monthly_returns)

# Calculate FF5 alpha
def get_ff5_alpha(return_series):
    merged = pd.merge(return_series, ff_df, left_on='date', right_index=True, how='inner')    
    merged.dropna(inplace=True)
    Y = 100 * merged['portfolio_return']
    X_ff5 = sm.add_constant(merged[['mktrf', 'smb', 'hml', 'umd', 'liq']])
    model = sm.OLS(Y, X_ff5).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    return model.params.get('const'), model.tvalues.get('const')

# Calculate performance of portfolios
def analyze_performance_double_sort(portfolio_returns):
    if START_DATE:
        portfolio_returns = portfolio_returns[portfolio_returns['date'] >= START_DATE].copy()
    if END_DATE:
        portfolio_returns = portfolio_returns[portfolio_returns['date'] <= END_DATE].copy()

    analysis_summary_rows = []
    for sq in range(1, 5):
        q_high = portfolio_returns[(portfolio_returns['beta_q'] == 4) & (portfolio_returns['secondary_q'] == sq)]
        q_low = portfolio_returns[(portfolio_returns['beta_q'] == 1) & (portfolio_returns['secondary_q'] == sq)]
        
        # Calculate high-minus-low beta spread within the secondary quartile
        combined_df = pd.merge(q_low[['date', 'portfolio_return']], q_high[['date', 'portfolio_return']], on='date', suffixes=('_q1', '_q4'))
        combined_df['ret_spread'] = combined_df['portfolio_return_q4'] - combined_df['portfolio_return_q1']
        alpha_hl, t_stat_hl = get_ff5_alpha(combined_df[['date', 'ret_spread']].rename(columns={'ret_spread': 'portfolio_return'}))
        
        # Save results
        row_data = {'Secondary_Quartile': sq}
        row_data['High_Low_Alpha'] = alpha_hl
        row_data['High_Low_t_stat'] = t_stat_hl
        analysis_summary_rows.append(row_data)

    results_df = pd.DataFrame(analysis_summary_rows).set_index('Secondary_Quartile')
    final_column_order = ['High_Low_Alpha', 'High_Low_t_stat']
    results_df = results_df[[col for col in final_column_order]]
    return results_df

# Load data
beta_df = pd.read_csv('tail_risk_betas.csv', dtype={'PERMNO': 'int32', 'beta': 'float32'})
beta_df['formation_date'] = pd.to_datetime(beta_df['formation_date']) + pd.offsets.MonthEnd(0)

stock_data = pd.read_csv("processed_monthly_stock_data.csv", usecols=['PERMNO', 'date', 'last_price', 'RET_monthly', 'last_value'])
stock_data['date'] = pd.to_datetime(stock_data['date'])

ff_df = pd.read_csv("ff3_monthly.csv", usecols=['dateff', 'mktrf', 'smb', 'hml', 'rf', 'umd'])
ff_df.rename(columns={'dateff': 'date'}, inplace=True)
ff_df['date'] = pd.to_datetime(ff_df['date'], format='%d/%m/%Y') + pd.offsets.MonthEnd(0)

liq_df = pd.read_csv("liq_factor.csv", usecols=['DATE', 'PS_VWF'])
liq_df.rename(columns={'DATE': 'date', 'PS_VWF': 'liq'}, inplace=True)
liq_df['date'] = pd.to_datetime(liq_df['date'], format='%d/%m/%Y') + pd.offsets.MonthEnd(0)

# Merge factors df
ff_df = pd.merge(ff_df, liq_df, on='date', how='inner')
ff_df.set_index('date', inplace=True)
ff_df = ff_df.apply(pd.to_numeric, errors='coerce').dropna()

# Load market return data
lambda_df = pd.read_csv('lambda_estimates.csv', usecols=['date', 'market_1m_ret'])
lambda_df['date'] = pd.to_datetime(lambda_df['date']) + pd.offsets.MonthEnd(0)
lambda_df.set_index('date', inplace=True)
lambda_df['market_ret'] = lambda_df['market_1m_ret'].shift(1)
lambda_df.dropna(inplace=True)

# Load or calculate secondary metrics
if RECALCULATE_METRICS:
    idio_vol_df = calculate_idio_vol()
    dbeta_coskew_df = calculate_dbeta_coskew()
else:
    print("Loading pre-calculated secondary metrics...")
    idio_vol_df = pd.read_csv(IDIO_VOL_FILE, dtype={'PERMNO': 'int32', 'idio_vol': 'float32'})
    dbeta_coskew_df = pd.read_csv(DBETA_COSKEW_FILE, dtype={'PERMNO': 'int32', 'downside_beta': 'float32', 'coskewness': 'float32'})

idio_vol_df['formation_date'] = pd.to_datetime(idio_vol_df['formation_date'])
dbeta_coskew_df['formation_date'] = pd.to_datetime(dbeta_coskew_df['formation_date'])

# Prepare full dataframe will all relevant columns
financials = stock_data[['PERMNO', 'date', 'last_price', 'last_value']].rename(columns={'date': 'formation_date'})
everything_df = pd.merge(beta_df, financials, on=['PERMNO', 'formation_date'], how='left')
everything_df = pd.merge(everything_df, idio_vol_df, on=['PERMNO', 'formation_date'], how='left')
everything_df = pd.merge(everything_df, dbeta_coskew_df, on=['PERMNO', 'formation_date'], how='left')
everything_df.dropna(subset=['beta', 'last_price'], inplace=True)
# Apply price filter
everything_df = everything_df[abs(everything_df['last_price']) >= 5]

# Link names of characteristics with column names
characteristics_setup = {
    'A_Size': {'col': 'last_value'},
    'B_IdioVol': {'col': 'idio_vol'},
    'C_DownsideBeta': {'col': 'downside_beta'},
    'D_Coskewness': {'col': 'coskewness'}
}

for label, config in characteristics_setup.items():
    secondary_col = config['col']
    print(f"Processing Panel {label}")

    # Drop nan rows only where this specific metric doesn't exist
    panel_data = everything_df.dropna(subset=[secondary_col]).copy()

    for weighting in ['ew', 'vw']:        
        # Calculate the monthly return for each portfolio 
        portfolio_returns = form_portfolios(panel_data, 12, weighting, secondary_col)
        # The reported monthly return is the average return of all portfolios active that month. 
        portfolios = portfolio_returns.groupby(['return_date', 'beta_q', 'secondary_q'])['portfolio_return'].mean().reset_index().rename(columns={'return_date': 'date'})
        # Calculate performance metrics
        performance = analyze_performance_double_sort(portfolios)
        # Save to csv
        performance.to_csv(f"table5_{label}_{weighting}.csv")
