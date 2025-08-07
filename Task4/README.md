# Complex Data Munging & Statistical Modeling on S&P 500 Stock Dataset

## Project Description

This project demonstrates advanced data cleaning, feature engineering, and statistical modeling on a real-world messy financial dataset.  
We utilize daily stock prices from the S&P 500 (5 years of data) sourced from Kaggle, focusing on preparing the dataset and predicting next-day stock price movement with logistic regression.

## Project Components

### 1. Data Preparation (`data_prep.ipynb`)

- Loading and initial exploration of the raw dataset.
- Multi-step data cleaning:
  - Missing data imputation by forward/backward filling within each stock ticker.
  - Outlier detection using z-score and capping extreme values.
  - Conversion of data types for efficiency (categorical tickers, integer volumes).
- Schema inference:
  - Creation of a MultiIndex for time-series access.
  - Pivoting to wide format occurrences.
- Feature engineering for classification:
  - Polynomial features (squared closing prices).
  - Interaction terms (close price × volume).
  - Rolling 5-day means.
  - Lagged target variable indicating if next day price increased.

### 2. Statistical Modeling (`model.ipynb`)

- Logistic regression to predict binary target (price up or down next trading day).
- Model fitting using `statsmodels`.
- Parameter estimates with confidence intervals and p-values.
- Wald hypothesis tests for joint significance of polynomial and interaction features.
- Computation of odds ratios for interpretability.
- Visualization of predicted probabilities against close prices.

## How to Run

1. Install the required Python packages:

pip install -r requirements.txt


2. Download the dataset `all_stocks_5yr.csv` from Kaggle and place it in the project folder.

3. Run `data_prep.ipynb` to clean and engineer features.

4. Run `model.ipynb` to fit the logistic regression model and interpret results.

## Dataset Source

- Kaggle: [S&P 500 Stock Data](https://www.kaggle.com/datasets/camnugent/sandp500)  
- Note: Ensure to comply with the dataset license and citation requirements.

## Dependencies

- Python 3.7 or higher  
- pandas, numpy, scipy, statsmodels, matplotlib

---

