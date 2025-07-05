# BSc Thesis Code

This repository contains the Python scripts used for the thesis: 'Tail risk spillovers between economic sectors and their implications for asset pricing' by Yusif Ismailov

## Data Requirements

To run the code, you need to obtain the required data files.

Required files:
*   `stock_data.csv`: Daily stock data from CRSP.
*   `market_return_daily.csv`: Daily value-weighted market return data from CRSP.
*   `ff3_daily.csv` & `ff3_monthly.csv`: Daily and monthly Fama-French factor data from WRDS (including momentum factor).
*   `liq_factor.csv`: Monthly Pastor-Stambaugh liquidity factor data from WRDS.
*   `Goyal-Welch.csv`: Monthly predictor data, which can be downloaded from [Amit Goyal's website](https://sites.google.com/view/agoyal145).

For more details on the data and processing, please refer to the thesis.

## Code Execution Order

The scripts are designed to be run sequentially to generate all results. The following order is recommended. 

The header of each file contains a description of what it does and the required input files. Refer to that if you don't want to run everything. 

### Part 1: Replication of Kelly and Jiang (2014)

This part replicates the figures and tables related to the tail risk metric. 

1.  `data_prep.py`
2.  `process_market_data.py`
3.  `lambda.py`
4.  `monthly_returns.py`
5.  `cross_section_single.py`
6.  `cross_section_double.py`

### Part 2: Sector Analysis Extension

This part extends the analysis to sector-level spillovers in tail risk.

1.  `sector_data_prep.py`
2.  `sector_lambda.py`
3.  `sector_scree.py`
4.  `sector_spillovers.py`
5.  `sector_monthly_returns.py`
6.  `sector_RMG_factor.py`
7.  `sector_alpha.py`
8.  `sector_cross_section_single.py`

### Part 3: Predictive Regressions

This part uses outputs from both Part 1 and Part 2 to run regressions on future market returns. 

1.  `predictive_regressions.py`
