# Rossmann Store Sales - Project Summary

This repository contains the analysis and modeling for the Rossmann Store Sales prediction project. The goal is to forecast daily sales for Rossmann drug stores using historical sales data.

## 1. Project Overview & Data Preprocessing
- **Objective:** Predict daily sales for Rossmann drug stores.
- **Data Handling:**
    - Cleaned `Date` column and extracted Year, Month, Day.
    - Handled missing values in `CompetitionDistance` (filled with median) and other columns.
    - Merged store information with train/test datasets using `Store` ID.
    - Treated categorical variables using mapping (StateHoliday) and One-Hot Encoding.

## 2. Exploratory Data Analysis (EDA) Insights
- **Key Findings:**
    - Promos boost sales significantly.
    - Store Type 'b' has the highest average sales.
    - Seasonality: Sales peak during December (Christmas season).
    - Competitor distance impacts sales but varies by store type.
    - School holidays affect sales differently across store types.

## 3. Feature Engineering
- **Techniques Used:**
    - **One-Hot Encoding:** For categorical variables like `StateHoliday`.
    - **Label Encoding:** For ordinal variables like `StoreType` and `Assortment`.
    - **Date Features:** Extracted `DayOfWeek`, `Month`, `Year`, etc.
    - **Missing Value Imputation:** For `CompetitionDistance` and `CompetitionOpenSince...`.

## 4. Model Building & Evaluation
We trained and evaluated three regression models:
1.  **Linear Regression (Baseline):** Simple model for benchmarking.
2.  **Random Forest Regressor:** Handles non-linear relationships well.
3.  **XGBoost Regressor:** Optimized gradient boosting for high performance.

### Performance Summary:
- **Linear Regression:** Provided a baseline but struggled with store-specific complexities.
- **Random Forest:** Significantly outperformed Linear Regression, capturing non-linear patterns.
- **XGBoost:** Best overall performance, balancing accuracy and computational efficiency.
- **Metrics:** Evaluated using MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).

## 5. Feature Importance Analysis
From the tree-based models (Random Forest, XGBoost), the most influential features were:
1.  **CompetitionDistance:** Distance to the nearest competitor.
2.  **Promo:** Active promotions strongly drive sales.
3.  **StoreType:** Different store formats have distinct sales patterns.
4.  **DayOfWeek:** Weekly cycles are evident.

## 6. Time Series Analysis
We performed Time Series Analysis and Moving Average analysis on the sales data:
- **Trend Analysis:** Visualized daily sales trends over time.
- **Seasonality:** Confirmed clear weekly and yearly seasonality (Christmas peaks).
- **Moving Average:** Calculated a 30-day moving average as a baseline model.
- **Comparison:** While useful for trend identification, aggregate Time Series models (like Moving Average) are less accurate than store-specific ML models (XGBoost) due to the high variability between individual stores.

## Conclusion
The project successfully developed a robust sales forecasting pipeline. **XGBoost** proved to be the most effective model, leveraging features like `CompetitionDistance`, `Promo`, and `StoreType` to deliver accurate predictions. The analysis highlights the importance of store-specific characteristics and promotional activities in driving sales.
