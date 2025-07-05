# Calculates and plots Diebold-Yilmaz tail risk spillover metrics between sectors in a rolling window Group LASSO VAR approach. 
#
# Required files: 
#   -'sector_lambda_estimates.csv' file created by 'sector_lambda.py'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import MultiTaskLasso
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import optuna
import warnings

# Config
WINDOW_SIZE = 120           # rolling window size in months
FORECAST_HORIZON = 12       # forecast horizon for GFEVD

# Bootstrap config
N_BOOTSTRAP_REPS = 1000   

# Prepares lagged data for VAR model
def prepare_var_data(data, lags):
    X, Y = [], []
    for i in range(lags, len(data)):
        Y.append(data.iloc[i].values)
        X.append(data.iloc[i-lags:i].values.flatten())
    return np.array(X), np.array(Y)

# Calculates moving average coefficients from standard form VAR coefficients
def calculate_ma(var_coefs, forecast_horizon, num_vars):
    p = var_coefs.shape[0] # number of lags
    # Shape of MA matrix
    ma_coefs = np.zeros((forecast_horizon + 1, num_vars, num_vars))
    # Recursive method for calculating MA coefficients with identity matrix for the 0th lag
    ma_coefs[0] = np.eye(num_vars)
    for h in range(1, forecast_horizon + 1):
        for i in range(1, p + 1):
            if h - i >= 0:
                ma_coefs[h] += np.dot(var_coefs[i - 1], ma_coefs[h - i])
    return ma_coefs

# Calculates spillovers using GFEVD
def calculate_gfevd(var_coefs, residuals, forecast_horizon, var_names):
    num_vars = len(var_names)
    # Get MA coefficients from standard VAR coefficients
    ma_coefs = calculate_ma(var_coefs, forecast_horizon, num_vars)
    # Estimated variance matrix
    sigma = np.cov(residuals, rowvar=False)

    # GVEFD formula from Pesaran
    gfevd_unnormalized = np.zeros((num_vars, num_vars))
    for i in range(num_vars):
        e_i = np.zeros(num_vars)
        e_i[i] = 1.0
        for j in range(num_vars):
            e_j = np.zeros(num_vars)
            e_j[j] = 1.0
            numerator_ij = 0
            denominator_ij = 0
            for h in range(forecast_horizon + 1):
                numerator_term = np.dot(e_i, np.dot(ma_coefs[h], np.dot(sigma, e_j)))
                numerator_ij += numerator_term**2
                
                denominator_term = np.dot(e_i, np.dot(ma_coefs[h], np.dot(sigma, np.dot(ma_coefs[h], e_i))))
                denominator_ij += denominator_term
            gfevd_unnormalized[i, j] = (numerator_ij / sigma.diagonal()[j]) / denominator_ij

    # normalize the GFEVD table
    pairwise_spillover = gfevd_unnormalized / gfevd_unnormalized.sum(axis=1, keepdims=True)

    # 'to' and 'from' spillovers are row and column sums without the diagonals 
    spillovers_to = (pairwise_spillover.sum(axis=1) - np.diag(pairwise_spillover)) * 100
    spillovers_from = (pairwise_spillover.sum(axis=0) - np.diag(pairwise_spillover)) * 100

    # Calculate net spillover
    net_spillovers = spillovers_from - spillovers_to
    # Total spillover index is average of all elements without diagonals
    total_spillover_index = spillovers_to.sum() / num_vars

    return {
        'net_spillovers': pd.Series(net_spillovers, index=var_names),
        'total_index': total_spillover_index
    }

# Function to select best lag order for VAR, while ensuring it's at least 0. 
def find_best_lag(data):
    model = VAR(data)
    best_lag = model.select_order(maxlags=5).bic
    return best_lag if best_lag > 0 else 1

# Load data
data = pd.read_csv('sector_lambda_estimates.csv', index_col='date', parse_dates=True)
data.dropna(inplace=True)
data = data.asfreq('MS')
# Rename columns
SECTOR_NAME_MAP = {
    'Cnsmr': 'Consumer', 'Manuf': 'Manufacturing', 'HiTec': 'High-Tech',
    'Hlth': 'Healthcare', 'Other': 'Other'
}
data.rename(columns=SECTOR_NAME_MAP, inplace=True)

# Main loop
results = {
    'net_spillovers': {}, 'total': {},
    'total_lower_bound': {}, 'total_upper_bound': {},
    'net_lower_bound': {}, 'net_upper_bound': {}
}
analysis_dates = data.index[WINDOW_SIZE:]
num_vars = data.shape[1]

# Suppress logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

# Rolling window
for i, end_date in enumerate(analysis_dates):
    # Take most recent 120 months of data
    window_data = data.loc[data.index <= end_date].tail(WINDOW_SIZE)
    
    if (i + 1) % 50 == 0:
        print(f"Processing window {i + 1}/{len(analysis_dates)}")
    
    # Find best lag order
    best_lag = find_best_lag(window_data)

    # Define the objective function to tune the Lasso hyperparameter
    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-3, 1.0, log=True)
        tscv = TimeSeriesSplit(n_splits=4)
        mse_scores = []
        # Make time series split
        for train_idx, test_idx in tscv.split(window_data):
            scaler = StandardScaler()
            train_fold, test_fold = window_data.iloc[train_idx], window_data.iloc[test_idx]
            
            # Scale and prepare data
            scaled_train = pd.DataFrame(scaler.fit_transform(train_fold), index=train_fold.index, columns=train_fold.columns)
            X_train, y_train = prepare_var_data(scaled_train, best_lag)
            
            # Prepare test data using the same scaler
            test_prep_data = pd.concat([scaled_train.tail(best_lag), pd.DataFrame(scaler.transform(test_fold), index=test_fold.index, columns=test_fold.columns)])
            X_test, y_test = prepare_var_data(test_prep_data, best_lag)

            # Fit model and evaluate
            lasso = MultiTaskLasso(alpha=alpha, max_iter=3000, tol=1e-3, random_state=123)
            lasso.fit(X_train, y_train)
            mse_scores.append(mean_squared_error(y_test, lasso.predict(X_test)))
        return np.mean(mse_scores)

    # Run Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, n_jobs=-1)
    best_alpha = study.best_params['alpha']

    # Fit final model full window and best alpha
    final_scaler = StandardScaler()
    scaled_window_data = pd.DataFrame(final_scaler.fit_transform(window_data), index=window_data.index, columns=window_data.columns)
    X_full, y_full = prepare_var_data(scaled_window_data, best_lag)
    
    final_lasso = MultiTaskLasso(alpha=best_alpha, max_iter=3000, tol=1e-3, random_state=123)
    final_lasso.fit(X_full, y_full)

    # Reshape resulting coefficient matrix into standard form (lags, equations, predictors)
    var_coefs = final_lasso.coef_.reshape(num_vars, best_lag, num_vars).transpose(1, 0, 2)
    # Calculate residuals
    residuals = y_full - final_lasso.predict(X_full)
    # Calculate spillovers
    spillover_calcs = calculate_gfevd(var_coefs, residuals, FORECAST_HORIZON, data.columns)
    
    results['net_spillovers'][end_date] = spillover_calcs['net_spillovers']
    results['total'][end_date] = spillover_calcs['total_index']

    # Bootstrapped confidence intervals
    bootstrap_totals, bootstrap_nets = [], []
    for _ in range(N_BOOTSTRAP_REPS):
        # Resample residuals with replacement
        boot_indices = np.random.choice(len(residuals), size=len(residuals), replace=True)
        boot_residuals = residuals[boot_indices]
        boot_spillover_calcs = calculate_gfevd(var_coefs, boot_residuals, FORECAST_HORIZON, data.columns)
        bootstrap_totals.append(boot_spillover_calcs['total_index'])
        bootstrap_nets.append(boot_spillover_calcs['net_spillovers'])

    # Calculate intervals from empirical distribution percentiles
    results['total_lower_bound'][end_date] = np.percentile(bootstrap_totals, 2.5)
    results['total_upper_bound'][end_date] = np.percentile(bootstrap_totals, 97.5)
    
    boot_nets_df = pd.DataFrame(bootstrap_nets)
    results['net_lower_bound'][end_date] = boot_nets_df.quantile(0.025)
    results['net_upper_bound'][end_date] = boot_nets_df.quantile(0.975)


# Save results
net_df = pd.DataFrame(results['net_spillovers']).T
total_series = pd.Series(results['total'], name='total_index')

net_df.to_csv('net_spillovers.csv', index_label='date')
total_series.to_csv('total_spillover_index.csv', header=True, index_label='date')

total_lower = pd.Series(results['total_lower_bound'], name='lower_bound')
total_upper = pd.Series(results['total_upper_bound'], name='upper_bound')
total_df_plot = pd.concat([total_series, total_lower, total_upper], axis=1)

net_lower_df = pd.DataFrame(results['net_lower_bound']).T
net_upper_df = pd.DataFrame(results['net_upper_bound']).T

# Plot 1: Total Spillover Index
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(total_df_plot.index, total_df_plot['total_index'], linewidth=2.5, color='black', label='Total Spillover Index')
ax.fill_between(total_df_plot.index, total_df_plot['lower_bound'], total_df_plot['upper_bound'], color='gray', alpha=0.35, label='95% Confidence Band')
ax.set_title('Total Spillover Index', fontsize=18, pad=20)
ax.set_ylabel('Index Value', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig("total_spillover_with_ci.png", dpi=300)
plt.close(fig)

# Plot 2: net spillover per sector
for sector in data.columns:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(net_df.index, net_df[sector], color='black', linewidth=2, label='Net Spillover')
    ax.fill_between(net_df.index, net_lower_df[sector], net_upper_df[sector], color='gray', alpha=0.4, label='95% Confidence Interval')
    ax.axhline(0, color='red', linestyle='--', linewidth=1.2)
    # Fixed y-axis for comparability
    ax.set_ylim(-55, 85) 
    ax.set_title(f'Net Spillover for the {sector} Sector', fontsize=18, pad=20)
    ax.set_ylabel('Net Spillover', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f"net_spillover_{sector}.png", dpi=300)
    plt.close(fig)
