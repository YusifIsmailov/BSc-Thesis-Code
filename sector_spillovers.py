# Calculates Diebold-Yilmaz tail risk spillovers between sectors in a rolling window Group LASSO VAR approach. 
# Also generates network spillover graphs, and compares the total spillover index with athe market-wide tail risk metric.
#
# Required files: 
#   - 'sector_lambda_estimates.csv' file created by 'sector_lambda.py'
#   - 'lambda_estimates.csv' (file created by 'lambda.py'

import pandas as pd
import numpy as np
import matplotlib
# Necessary for network graphs
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import networkx as nx
from matplotlib.lines import Line2D
from sklearn.linear_model import MultiTaskLasso
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import optuna
import warnings

# Config
WINDOW_SIZE = 120           
FORECAST_HORIZON = 12     

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
def calculate_spillovers(var_coefs, residuals, forecast_horizon, var_names):
    num_vars = len(var_names)
    # Get MA coefficients from standard VAR coefficients
    ma_coefs = calculate_ma(var_coefs, forecast_horizon, num_vars)
    # Estimated variance matrix
    sigma = np.cov(residuals, rowvar=False)

    # GFEVD formula from Pesaran
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

    # Normalize the GFEVD table
    pairwise_spillover = gfevd_unnormalized / gfevd_unnormalized.sum(axis=1, keepdims=True)
    pairwise_spillover_df = pd.DataFrame(pairwise_spillover * 100, index=var_names, columns=var_names)

    # 'to' and 'from' spillovers are row and column sums without the diagonals 
    spillovers_to = (pairwise_spillover.sum(axis=1) - np.diag(pairwise_spillover)) * 100
    spillovers_from = (pairwise_spillover.sum(axis=0) - np.diag(pairwise_spillover)) * 100
    
    # Calculate net spillover
    net_spillovers = spillovers_from - spillovers_to
    # Total spillover index is average of all elements without diagonals
    total_spillover_index = spillovers_from.sum() / num_vars

    return {
        'net_spillovers': pd.Series(net_spillovers, index=var_names),
        'total_index': total_spillover_index,
        'pairwise_table': pairwise_spillover_df
    }

# Function to select best lag order for VAR, while ensuring it's at least 1.
def find_best_lag(data):
    model = VAR(data)
    best_lag = model.select_order(maxlags=5).bic
    return best_lag if best_lag > 0 else 1

# Function to make network stucture plots
def plot_spillover_network(pairwise_df, title, ax):
    cool_color, hot_color ='#d62728', '#1f77b4'
    # Create a circular graph
    G = nx.from_pandas_adjacency(pairwise_df, create_using=nx.DiGraph())
    pos = nx.circular_layout(G, scale=1.8)
    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Calculate net spillover from the spillover table
    net_spillovers = (pairwise_df.sum(axis=1) - np.diag(pairwise_df)) - (pairwise_df.sum(axis=0) - np.diag(pairwise_df))
    
    # Draw nodes (sectors)
    node_sizes = [6000 + 350 * abs(net_spillovers.get(node)) for node in G.nodes()]
    node_colors = [hot_color if net_spillovers.get(node) < 0 else cool_color for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, ax=ax, alpha=0.95)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', font_color='white', ax=ax)
    
    # Determine for each pair which spillover is stronger
    stronger_reciprocal, weaker_reciprocal = [], []
    processed_pairs = set()
    for u, v in G.edges():
        if (u, v) in processed_pairs: continue
        processed_pairs.add((u, v))
        processed_pairs.add((v, u))
        weight_uv = G[u][v]['weight']
        weight_vu = G[v][u]['weight']
        if weight_uv >= weight_vu:
            stronger_reciprocal.append((u, v))
            weaker_reciprocal.append((v, u))
        else:
            stronger_reciprocal.append((v, u))
            weaker_reciprocal.append((u, v))

    # Plot spillovers and labels in the right colors
    nx.draw_networkx_edges(G, pos, edgelist=stronger_reciprocal, connectionstyle='arc3,rad=0.20',
                           edge_color=cool_color, width=2.2, arrowsize=25, ax=ax, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, edgelist=weaker_reciprocal, connectionstyle='arc3,rad=0.20',
                           edge_color=hot_color, width=1.8, arrowsize=20, ax=ax, node_size=node_sizes)
    
    label_props = {'ax': ax, 'rotate': True, 'font_size': 10, 'font_weight': 'bold',
                    'bbox': dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2')}   
    stronger_labels = {(u, v): f"{G[u][v]['weight']:.1f}%" for u,v in stronger_reciprocal}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=stronger_labels, font_color=hot_color, 
                                    label_pos=0.35, **label_props)
    
    weaker_labels = {(u, v): f"{G[u][v]['weight']:.1f}%" for u,v in weaker_reciprocal}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weaker_labels, font_color=cool_color, 
                                    label_pos=0.35, **label_props)

    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Net Transmitter', markerfacecolor=hot_color, markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Net Receiver', markerfacecolor=cool_color, markersize=15),
        Line2D([0], [0], color=hot_color, lw=3, label='Stronger spillover in a pair'),
        Line2D([0], [0], color=cool_color, lw=3, label='Weaker spillover in a pair'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=11)
    ax.set_title(title, fontsize=20, pad=20, weight='bold')
    ax.axis('off')

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
num_vars = data.shape[1]

# Suppress logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

# Main loop
results = {
    'net_spillovers': {}, 'total': {}, 'pairwise_tables': {},
    'total_lower_bound': {}, 'total_upper_bound': {},
    'net_lower_bound': {}, 'net_upper_bound': {}
}
analysis_dates = data.index[WINDOW_SIZE:]

# Rolling window
for i, end_date in enumerate(analysis_dates):
    # Take most recent 120 months of data
    window_data = data.loc[data.index <= end_date].tail(WINDOW_SIZE)
    
    if (i + 1) % 50 == 0:
        print(f"Processing window {i + 1}/{len(analysis_dates)}")    
    
    # Find best lag order
    best_lag = find_best_lag(window_data)

    # Define the objective function to tune the lasso hyperparameter
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

    # Fit final model on full window with best alpha
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
    spillover_calcs = calculate_spillovers(var_coefs, residuals, FORECAST_HORIZON, data.columns)
    
    results['net_spillovers'][end_date] = spillover_calcs['net_spillovers']
    results['total'][end_date] = spillover_calcs['total_index']
    results['pairwise_tables'][end_date] = spillover_calcs['pairwise_table']

    # Bootstrapped confidence intervals
    bootstrap_totals, bootstrap_nets = [], []
    for _ in range(N_BOOTSTRAP_REPS):
        # Resample residuals with replacement
        boot_indices = np.random.choice(len(residuals), size=len(residuals), replace=True)
        boot_residuals = residuals[boot_indices]
        # Apply them to the estimated coefficients
        boot_spillovers = calculate_spillovers(var_coefs, boot_residuals, FORECAST_HORIZON, data.columns)
        bootstrap_totals.append(boot_spillovers['total_index'])
        bootstrap_nets.append(boot_spillovers['net_spillovers'])

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

# Prepare dataframes for plotting
total_lower = pd.Series(results['total_lower_bound'], name='lower_bound')
total_upper = pd.Series(results['total_upper_bound'], name='upper_bound')
total_df_plot = pd.concat([total_series, total_lower, total_upper], axis=1)

net_lower_df = pd.DataFrame(results['net_lower_bound']).T
net_upper_df = pd.DataFrame(results['net_upper_bound']).T

# Plot 1: total spillover index
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

# Plot 3: total spillover index vs tail risk
tail_risk_df = pd.read_csv('lambda_estimates.csv', index_col='date', parse_dates=True)
merged_df = pd.merge(total_df_plot, tail_risk_df, left_index=True, right_index=True, how='inner')

# Standardize
spillover_mean = merged_df['total_index'].mean()
spillover_std = merged_df['total_index'].std()
merged_df['total_index_std'] = (merged_df['total_index'] - spillover_mean) / spillover_std
merged_df['spillover_lower_std'] = (merged_df['lower_bound'] - spillover_mean) / spillover_std
merged_df['spillover_upper_std'] = (merged_df['upper_bound'] - spillover_mean) / spillover_std

lambda_mean = merged_df['lambda'].mean()
lambda_std = merged_df['lambda'].std()
merged_df['lambda_std'] = (merged_df['lambda'] - lambda_mean) / lambda_std
merged_df['lambda_lower_std'] = (merged_df['ci_lower_95'] - lambda_mean) / lambda_std
merged_df['lambda_upper_std'] = (merged_df['ci_upper_95'] - lambda_mean) / lambda_std

# US recession periods
recession_periods = [
    ('1969-12-01', '1970-11-01'), ('1973-11-01', '1975-03-01'), ('1980-01-01', '1980-07-01'),
    ('1981-07-01', '1982-11-01'), ('1990-07-01', '1991-03-01'), ('2001-03-01', '2001-11-01'),
    ('2007-12-01', '2009-06-01'), ('2020-02-01', '2020-04-01')
]

fig, ax = plt.subplots(figsize=(16, 8))
# Add recession shading
for start, end in recession_periods:
    ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='lightsteelblue', alpha=0.4, zorder=0, label='_nolegend_')

# Plot spillover index
ax.plot(merged_df.index, merged_df['total_index_std'], color='black', linestyle='-', label='Total Spillover Index', zorder=5)
ax.fill_between(merged_df.index, merged_df['spillover_lower_std'], merged_df['spillover_upper_std'],
    color='darkgrey', alpha=0.35, label='95% CI for Spillover Index', zorder=2)

# Plot tail risk metric
ax.plot(merged_df.index, merged_df['lambda_std'], color='darkgreen', linestyle='--', label='Market Tail Risk', zorder=4)
ax.fill_between(merged_df.index, merged_df['lambda_lower_std'], merged_df['lambda_upper_std'],
    color='darkseagreen', alpha=0.45, label='95% CI for Tail Risk', zorder=1)

ax.set_title('Standardized Total Spillover Index and Market Tail Risk', fontsize=18, pad=20)
ax.set_ylabel('Standardized Value', fontsize=14)
# Makes the grid look nicer
ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.savefig('spillover_vs_tailrisk_comparison.png', dpi=300)
plt.close(fig)

# Network structures
# Plot 4: rolling window average
all_pairwise_tables = list(results['pairwise_tables'].values())
average_pairwise_df = pd.concat(all_pairwise_tables).groupby(level=0).mean()
fig, ax = plt.subplots(figsize=(12, 12))
plot_spillover_network(average_pairwise_df, 'Rolling Window Average Spillover Network', ax)
plt.savefig("network_rolling_average.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# Plot 5: 6 disjoint time periods
data_chunks = np.array_split(data, 6)

# Go through each period and do the process again
for i, period_data in enumerate(data_chunks):
    start_date = period_data.index.min().strftime('%Y-%m')
    end_date = period_data.index.max().strftime('%Y-%m')
    
    best_lag_period = find_best_lag(period_data)
    def objective_period(trial):
        alpha = trial.suggest_float('alpha', 1e-3, 1.0, log=True)
        tscv = TimeSeriesSplit(n_splits=4)
        mse_scores = []
        for train_idx, test_idx in tscv.split(period_data):
            scaler = StandardScaler()
            train_fold, test_fold = period_data.iloc[train_idx], period_data.iloc[test_idx]
            
            scaled_train = pd.DataFrame(scaler.fit_transform(train_fold), index=train_fold.index, columns=train_fold.columns)
            X_train, y_train = prepare_var_data(scaled_train, best_lag_period)
            
            test_prep = pd.concat([scaled_train.tail(best_lag_period), pd.DataFrame(scaler.transform(test_fold), index=test_fold.index, columns=test_fold.columns)])
            X_test, y_test = prepare_var_data(test_prep, best_lag_period)
            
            lasso = MultiTaskLasso(alpha=alpha, max_iter=3000, tol=1e-3, random_state=123).fit(X_train, y_train)
            mse_scores.append(mean_squared_error(y_test, lasso.predict(X_test)))
        return np.mean(mse_scores) if mse_scores else float('inf')

    study_period = optuna.create_study(direction='minimize')
    study_period.optimize(objective_period, n_trials=100, n_jobs=-1)
    best_alpha_period = study_period.best_params['alpha']

    scaler_period = StandardScaler()
    scaled_period_data = pd.DataFrame(scaler_period.fit_transform(period_data), index=period_data.index, columns=period_data.columns)
    
    X_p, y_p = prepare_var_data(scaled_period_data, best_lag_period)
    final_lasso_p = MultiTaskLasso(alpha=best_alpha_period, max_iter=5000, tol=1e-4, random_state=123).fit(X_p, y_p)
    
    var_coefs_p = final_lasso_p.coef_.reshape(num_vars, best_lag_period, num_vars).transpose(1, 0, 2)
    residuals_p = y_p - final_lasso_p.predict(X_p)
    spillover_p = calculate_spillovers(var_coefs_p, residuals_p, FORECAST_HORIZON, data.columns)

    # Plot the network for the period
    fig, ax = plt.subplots(figsize=(12, 12))
    title = f'Spillover Network: {start_date} to {end_date}'
    plot_spillover_network(spillover_p['pairwise_table'], title, ax)
    plt.savefig(f"network_period_{i+1}_{start_date}_to_{end_date}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
