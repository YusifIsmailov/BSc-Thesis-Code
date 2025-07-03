# Create a scree plot for PCA on the sector tail risk series
#
# Required files: 
#   -'sector_lambda_estimates.csv' file created by 'sector_lambda.py'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load sector tail risk data
lambda_pivot_df = pd.read_csv("sector_lambda_estimates.csv", index_col='date', parse_dates=True)
lambda_pivot_df.dropna(inplace=True)

# Standardize
scaler = StandardScaler()
lambda_scaled = scaler.fit_transform(lambda_pivot_df)

# Perform PCA
pca = PCA()
pca.fit(lambda_scaled)
explained_variance = pca.explained_variance_ratio_

# Plot the explained variance for each PC
fig, ax = plt.subplots(figsize=(8, 5))

component_numbers = np.arange(1, 6)
ax.plot(component_numbers, explained_variance, marker='o', linestyle='-', color='darkblue')
ax.set_title('Scree Plot for PCA of Sector Tail Risk Series')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Proportion of Variance Explained')
ax.set_xticks(component_numbers)

# Fix y-axis to 0
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig("sector_scree_plot.png")
plt.close()
