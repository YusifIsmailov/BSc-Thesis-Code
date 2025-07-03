# Replicates tables 1, 2, 3 from Kelly and Jiang (2014). Does univariate, bivariate and expanding window pseudo OOS regressions to evaluate
# performance of tail risk metric in predicting future market returns, alongside various Goyal and Welch variables, as well as the total spillover index. 
# Uses the same sample for each variable to ensure comparability. 
#
# Required files: 
#   -'lambda_estimates.csv' file created by 'lambda.py'
#   -'total_spillover_index.csv' file created by 'sector_spillovers.py'
#   -'Goyal-Welch.csv' file containing monthly observations for many Goyal and Welch variables obtained from Amit Goyal's website. 

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load tail risk data
lambda_df = pd.read_csv('lambda_estimates.csv')
lambda_df['date'] = pd.to_datetime(lambda_df['date'])

# Load spillover data
spillover_df = pd.read_csv('total_spillover_index.csv')
spillover_df['date'] = pd.to_datetime(spillover_df['date'])

# Load Goyal-Welch data
gw_df = pd.read_csv('Goyal-Welch.csv', na_values=['NaN'])
gw_df['date'] = pd.to_datetime(gw_df['yyyymm'], format='%Y%m')

# Process Goyal-Welch variables
for col in ['b/m', 'corpr', 'ltr', 'BAA', 'AAA', 'D12', 'E12', 'Index','infl', 'lty', 'tbl', 'ntis', 'svar']:
    gw_df[col] = pd.to_numeric(gw_df[col], errors='coerce')

gw_df['dfr'] = gw_df['corpr'] - gw_df['ltr']
gw_df['dfy'] = gw_df['BAA'] - gw_df['AAA']
gw_df['de'] = np.log(gw_df['D12']) - np.log(gw_df['E12'])
gw_df['dp'] = np.log(gw_df['D12']) - np.log(gw_df['Index'])
gw_df['ep'] = np.log(gw_df['E12']) - np.log(gw_df['Index'])
gw_df['tms'] = gw_df['lty'] - gw_df['tbl']
gw_predictors = ['b/m', 'dfr', 'dfy', 'de', 'dp', 'ep', 'infl', 'ltr', 'lty', 'ntis', 'svar', 'tms', 'tbl']

# Merge all datasets
df = pd.merge(lambda_df, spillover_df, on='date', how='left')
df = pd.merge(df, gw_df[['date'] + gw_predictors], on='date', how='inner')
df.sort_values(by='date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Standardize variables
predictors = ['lambda', 'total_index'] + gw_predictors
for pred in predictors:
    df[f'{pred}_std'] = df[pred] / df[pred].std()

horizons_config = {
    '1M':  {'ret_col': 'market_1m_ret',  'years': 1/12, 'h_months': 1},
    '1Y':  {'ret_col': 'market_12m_ret', 'years': 1,    'h_months': 12},
    '3Y':  {'ret_col': 'market_36m_ret', 'years': 3,    'h_months': 36},
    '5Y':  {'ret_col': 'market_60m_ret', 'years': 5,    'h_months': 60}
}

# Annualize returns
for _, config in horizons_config.items():
    ret_col = config['ret_col']
    df[f'{ret_col}_annualized_pct'] = (np.power(1 + df[ret_col], 1/config['years']) - 1) * 100

# Filter by date to ensure each metric is tested on the same sample
df = df[df['date'] > '1971-12-31'].copy()

# Table 1
table1_data = []
# Go through all predictors
for pred in predictors:
    pred_std = f'{pred}_std'
    row = {'Predictor': pred}
    # Go through all horizons and perform regression
    for hor, config in horizons_config.items():
        ret_col = f"{config['ret_col']}_annualized_pct"
        row_df = df[[ret_col, pred_std]].dropna()
        Y = row_df[ret_col]
        X = sm.add_constant(row_df[pred_std])
        model = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': config['h_months'], 'use_correction':True})
        row[f'{hor}_Coeff'] = model.params.iloc[1]
        row[f'{hor}_t_stat'] = model.tvalues.iloc[1]
        row[f'{hor}_R2_pct'] = model.rsquared * 100
    table1_data.append(row)
# Save results
df_table1 = pd.DataFrame(table1_data)
df_table1.to_csv('table1.csv', index=False, float_format="%.2f")

# Table 2 
table2_data = []
predictors_for_table2 = ['total_index'] + gw_predictors
# Go through all predictors
for pred in predictors_for_table2:
    other_pred_std = f'{pred}_std'
    row = {'Predictor_1': 'lambda', 'Predictor_2': pred}
    # Go through all horizons and perform regression
    for hor, config in horizons_config.items():
        ret_col = f"{config['ret_col']}_annualized_pct"
        row_df = df[[ret_col, 'lambda_std', other_pred_std]].dropna()
        Y = row_df[ret_col]
        X = sm.add_constant(row_df[['lambda_std', other_pred_std]])
        model = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': config['h_months'], 'use_correction':True})
        row[f'{hor}_Coeff_1'] = model.params.get('lambda_std')
        row[f'{hor}_t_stat_1'] = model.tvalues.get('lambda_std')
        row[f'{hor}_Coeff_2'] = model.params.get(other_pred_std)
        row[f'{hor}_t_stat_2'] = model.tvalues.get(other_pred_std)
        row[f'{hor}_R2_pct'] = model.rsquared * 100
    table2_data.append(row)
df_table2 = pd.DataFrame(table2_data)
df_table2.to_csv('table2.csv', index=False, float_format="%.2f")

# Table 3
print("\nGenerating Table 3 (Out-of-Sample R2)...")
table3_data = []
# Go through all predictors
for pred in predictors:
    row = {'Predictor': pred}
    # Go through each horizon
    for hor, config in horizons_config.items():
        ret_col = f"{config['ret_col']}_annualized_pct"
        row_df = df[[pred, ret_col]].dropna().reset_index(drop=True)
        sum_sq_forecast_errors, sum_sq_benchmark_errors = 0, 0
        # Expanding window approach with initial 120 months
        for t in range(120, len(row_df)):
            # Make sure all training data is fully realized by the time of the prediction
            train_end = t - config['h_months']            
            Y_train = row_df[ret_col].iloc[:train_end + 1]
            benchmark = Y_train.mean()
            X_train = row_df[pred].iloc[:train_end + 1]
            model_oos = sm.OLS(Y_train, sm.add_constant(X_train)).fit()
            Y_pred = model_oos.predict(np.array([1, row_df[pred].iloc[t]]))[0]
            Y_actual = row_df[ret_col].iloc[t]
            sum_sq_forecast_errors += (Y_actual - Y_pred)**2
            sum_sq_benchmark_errors += (Y_actual - benchmark)**2
        
        # Save OOS R^2
        row[hor] = 100 * (1 - (sum_sq_forecast_errors / sum_sq_benchmark_errors))
    table3_data.append(row)
# Save results
df_table3 = pd.DataFrame(table3_data)
df_table3.to_csv('table3.csv', index=False, float_format="%.2f")
