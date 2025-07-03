# BSc-Thesis-Code
Code for 'Tail risk spillovers between economic sectors and their implications for asset pricing' by Yusif Ismailov

This repository contains the Python scripts used for the analysis in the thesis.

## Data Requirements

To run the code, you will first need to obtain the required data files. Most of the necessary data can be sourced from Wharton Research Data Services (WRDS), specifically from the CRSP and Fama-French libraries.

The following files are required external inputs:
*   `stock_data.csv`: Daily stock data from the CRSP database.
*   `market_return_daily.csv`: Daily market return data from the CRSP database.
*   `ff3_daily.csv` & `ff3_monthly.csv`: Daily and monthly Fama-French factor data from the Kenneth French Data Library (available via WRDS).
*   `liq_factor.csv`: Monthly Pástor-Stambaugh liquidity factor data (available on WRDS).
*   `Goyal-Welch.csv`: Monthly predictor variables, which must be downloaded from [Amit Goyal's website](https://sites.google.com/view/agoyal145).

For more specific details on variable names and data processing, please refer to the data section of the thesis.

## Code Execution Order

The scripts are designed to be run sequentially to generate all results from scratch. The following order is recommended for a full replication.

Note that the header of each `.py` file contains a detailed description of its purpose and required input files. This allows for more flexible execution if you only wish to reproduce specific parts of the analysis without running the entire pipeline.

### Part 1: Replication of Kelly and Jiang (2014)

This part focuses on replicating the core findings related to the aggregate tail risk measure.

1.  `data_prep.py`
2.  `process_market_data.py`
3.  `lambda.py`
4.  `monthly_returns.py`
5.  `cross_section_single.py`
6.  `cross_section_double.py`

### Part 2: Sector Analysis Extension

This part extends the analysis to the sector level, calculating sector-specific tail risk, spillovers, and asset pricing factors.

1.  `sector_data_prep.py`
2.  `sector_lambda.py`
3.  `sector_scree.py`
4.  `sector_spillovers.py`
5.  `sector_monthly_returns.py`
6.  `sector_RMG_factor.py`
7.  `sector_alpha.py`
8.  `sector_cross_section_single.py`

### Part 3: Predictive Regressions

This final part uses outputs generated in both Part 1 and Part 2 to run the market return predictability regressions.

1.  `sector_predictive_regressions.py`
