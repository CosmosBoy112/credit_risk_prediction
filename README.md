# Give Me Some Credit — Loan Default Prediction

A binary classification project on the Kaggle [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset, predicting whether a borrower will experience serious delinquency (90+ days past due) within the next two years.

The approach is to diagnose the **root cause** of each data issue through EDA first, then let those findings drive the cleaning order and method, followed by a Logistic Regression baseline → XGBoost → SHAP interpretation.

## Results

| Model | Validation AUC-ROC | Recall (default) | Precision (default) |
|---|---|---|---|
| Logistic Regression (`class_weight='balanced'`) | 0.8607 | 0.749 | 0.222 |
| **XGBoost** (`scale_pos_weight=13.96`, early stopping) | **0.8699** | **0.780** | 0.218 |

The target is heavily imbalanced at 6.7% : 93.3%, so **AUC-ROC** is used as the primary metric instead of accuracy. Top entries in the original competition also cluster in the 0.86–0.87 range, so this performance is close to the practical ceiling for the dataset.

## Files

| File | Description |
|---|---|
| `credit_default_prediction_1.ipynb` | Full analysis (EDA → cleaning → modeling → SHAP) |
| `cs-training.csv` | Training data, 150,000 rows × 11 columns (target included) |
| `cs-test.csv` | Competition test data, 101,503 rows (target blank) |
| `sampleEntry.csv` | Submission format (`Id`, `Probability`) |
| `Data Dictionary.xls` | Original variable definitions |

> The CSV files are not tracked in this repository — download them from the [competition data page](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place them in the repository root.

> The notebook currently uses only `cs-training.csv`, up through training and validation on an 80/20 stratified split. Generating a submission file from `cs-test.csv` is not included.

## Variables

| Variable | Description |
|---|---|
| `SeriousDlqin2yrs` | **Target**. 90+ days past due within 2 years |
| `RevolvingUtilizationOfUnsecuredLines` | Utilization of unsecured credit lines |
| `age` | Age in years |
| `NumberOfTime30-59DaysPastDueNotWorse` | Count of 30–59 day delinquencies |
| `DebtRatio` | Debt ratio (monthly debt payments / monthly income) |
| `MonthlyIncome` | Monthly income |
| `NumberOfOpenCreditLinesAndLoans` | Number of open loans and credit lines |
| `NumberOfTimes90DaysLate` | Count of 90+ day delinquencies |
| `NumberRealEstateLoansOrLines` | Number of real estate loans or lines |
| `NumberOfTime60-89DaysPastDueNotWorse` | Count of 60–89 day delinquencies |
| `NumberOfDependents` | Number of dependents |

## What EDA Found, and How It Shaped the Cleaning

1. **Class imbalance (6.7% : 93.3%)** → use AUC-ROC as the metric, and `class_weight='balanced'` / `scale_pos_weight` in the models.
2. **Missing values** — `MonthlyIncome` 29,731 rows (19.8%), `NumberOfDependents` 3,924 rows (2.6%).
3. **96/98 codes in the delinquency columns (269 rows)** — 96/98 appear simultaneously across all three delinquency columns, and this group has a **54.6%** default rate, more than 8x the overall average (6.7%). These look like sentinel codes for special cases rather than genuine counts, so they are **treated as missing and median-imputed, with a `had_sentinel_code` flag to preserve the signal**.
4. **Extreme `DebtRatio` overlaps with missing income** — **94.3%** of the top 1% of `DebtRatio` (>4979) have a missing `MonthlyIncome`. Since `DebtRatio = debt / income`, the missing denominator explains the exploding ratio. Rather than treating the two issues separately, the pipeline **imputes income first, then winsorizes `DebtRatio`**.
5. **One row with `age = 0`** — physically impossible, so it is dropped.

## Cleaning Pipeline

```
1. Drop rows with age = 0                              → 149,999 rows
2. Delinquency 96/98 → set missing, median-impute + had_sentinel_code flag
3. MonthlyIncome missing → median-impute + income_missing flag
4. NumberOfDependents missing → fill with 0 + dependents_missing flag
5. DebtRatio, RevolvingUtilizationOfUnsecuredLines → cap at the 99.5th percentile
```

Result: 149,999 rows × 14 columns (including the 3 engineered flags), zero missing values.

## Modeling

- **Split**: stratified to preserve the target ratio — 119,999 train / 30,000 validation.
- **Logistic Regression**: `StandardScaler`, `class_weight='balanced'`, `max_iter=1000`.
- **XGBoost**: `n_estimators=300`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=13.96`, early stopping at 30 rounds (best iteration 219).
- `random_state=42` throughout.

## SHAP Interpretation

- `RevolvingUtilizationOfUnsecuredLines` is the strongest and most consistent predictor.
- Delinquency history behaves like a **threshold effect**: borrowers with zero past delinquencies cluster tightly on the low-risk side, but a single event pushes the prediction sharply toward higher risk.
- Older age is associated with lower risk.
- The engineered `had_sentinel_code` flag shows a clear positive SHAP contribution, confirming that preserving the 96/98 anomaly as a flag — rather than discarding it — was the right call.
- `DebtRatio` and `MonthlyIncome` have relatively modest impact after capping and imputation, suggesting the cleaning step successfully suppressed distortion from extreme values.

## Deployment Considerations

At the default threshold of 0.5, precision is only about 0.22, meaning a high false-positive rate. In production the threshold should be re-optimized against the **quantified cost of missing a default versus the cost of wrongly flagging a good customer**.

## Running It

```bash
pip install pandas numpy matplotlib scikit-learn xgboost shap jupyter
jupyter notebook credit_default_prediction_1.ipynb
```

Written against Python 3.12. The notebook reads the CSVs by relative path, so run it from the repository root.
