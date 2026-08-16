# week_4
# House Price Prediction — Model Tuning & Ensembles — ASL Internship Week 4 (AI/ML Track)

## Project Overview
This project extends the Week 3 house price prediction pipeline by adding 
k-fold cross-validation, systematic hyperparameter search, and ensemble 
methods (Random Forest, Gradient Boosting) to produce a more reliable and 
better-performing final model.

**Problem type:** Regression (predicting a continuous target: `price`)
**Class imbalance strategy:** Not applicable — this is a regression task, not classification.

## Dataset
- **Source:** Kaggle — House Price Prediction dataset (`modified_data.csv`)
- **Rows:** 4,600 (original) → 4,462 after cleaning (reused from Week 3)
- **Target column:** `price`
- **Features:** bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront,
  view, condition, sqft_above, sqft_basement, yr_built, yr_renovated, city, sale_year

## Setup Instructions
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `week4_model_tuning_ensembles.ipynb` in Jupyter or Google Colab
4. Run all cells in order (Runtime → Run all in Colab)

## Project / Folder Structure
```
asl-internship-aiml-week4-<yourname>/
├── README.md
├── requirements.txt
├── data/
│   └── modified_data.csv
├── notebooks/
│   └── week4_model_tuning_ensembles.ipynb
├── models/
│   └── final_house_price_model_week4.joblib
├── report/
│   └── week4_report.docx
└── screenshots/
    ├── cross_validation_scores.png
    ├── gridsearch_results.png
    ├── model_comparison.png
    └── feature_importance.png
```

## Data Cleaning & Preprocessing (reused from Week 3)
- Removed `price_per_sqft` (data leakage — directly derived from target)
- Removed `street` (high cardinality) and `statezip` (redundant with `city`)
- Dropped 49 rows with `price == 0`
- Extracted `sale_year` from `date`
- Removed 89 outlier rows above ~$1.65M (IQR method, 3× IQR upper bound)
- Preprocessing pipeline: `StandardScaler` on 13 numeric features, 
  `OneHotEncoder` on `city`, wrapped in a scikit-learn `Pipeline` alongside 
  every model so preprocessing is refit on each cross-validation fold — no 
  leakage from validation/test data into training.

## Methodology

### 1. Cross-Validation
Applied 5-fold `KFold` cross-validation (`cross_val_score`) to Linear 
Regression and Random Forest to get a stable, multi-split performance 
estimate rather than relying on a single train/test split.

### 2. Hyperparameter Search
Used `GridSearchCV` (5-fold CV) to tune Random Forest across:
- `n_estimators`: [50, 100, 200]
- `max_depth`: [5, 10, 15, None]
- `min_samples_leaf`: [1, 5, 10]

Best parameters found: `max_depth=None, min_samples_leaf=1, n_estimators=200`
(Best CV R² = 0.6878)

### 3. Ensemble Models
Trained and cross-validated:
- Random Forest (bagging)
- Gradient Boosting (boosting)

### 4. Feature Importance
Extracted from the final Gradient Boosting model — `sqft_living` was the 
dominant predictor (56% importance), followed by city/location features 
(Seattle, Bellevue, Mercer Island) and structural features (sqft_above, 
yr_built, view).

## Results

| Model | CV Mean R² ± Std | Test R² | Train R² | Train/Test Gap |
|---|---|---|---|---|
| Linear Regression (Week 3 baseline) | — | 0.676 | 0.711 | 0.035 |
| Random Forest constrained (Week 3 baseline) | — | 0.658 | 0.799 | 0.141 |
| Linear Regression (CV) | 0.6965 ± 0.0274 | — | — | — |
| Random Forest default (CV) | 0.6863 ± 0.0288 | — | — | — |
| Random Forest tuned (GridSearchCV) | 0.6878 | 0.6787 | 0.9577 | 0.279 |
| **Gradient Boosting (final model)** | **0.6964 ± 0.0171** | **0.6839** | **0.765** | **0.081** |

**Final model: Gradient Boosting.** It achieved the best test R² (0.684), the 
best and most stable cross-validation score (lowest std across folds), and by 
far the smallest overfitting gap of any tree-based model — outperforming the 
GridSearchCV-tuned Random Forest, which reintroduced significant overfitting 
(0.28 gap) despite having the best raw CV score during search.

## Key Finding: CV Score Alone Doesn't Guarantee Generalization
GridSearchCV selected an unconstrained Random Forest (`max_depth=None`) as 
"best" based on cross-validated R², but this model showed a large train/test 
gap (0.958 vs 0.679) on the final held-out test set. This demonstrates that 
optimizing purely for cross-validated score does not protect against 
overfitting — the train/test gap must be checked as a separate diagnostic, 
which is why Gradient Boosting (a slightly lower CV score but far smaller 
overfitting gap) was chosen as the final model instead.

## Features Implemented
- ✅ Reused Week 3 dataset and preprocessing pipeline
- ✅ 5-fold cross-validation on 2+ models (Linear Regression, Random Forest)
- ✅ Hyperparameter search via GridSearchCV (Random Forest)
- ✅ Ensemble models trained (Random Forest, Gradient Boosting)
- ✅ Preprocessing kept inside Pipeline throughout — no leakage during tuning
- ✅ CV mean ± standard deviation reported for each model
- ✅ Tuned models compared against Week 3 baseline in a single table
- ✅ Feature importance extracted and interpreted for the best model
- ✅ Class imbalance — not applicable (regression task), noted explicitly
- ✅ Final model saved with `joblib`, hyperparameters recorded
- ✅ 200–350 word conclusion covering tuning gains, tradeoffs, and next steps

## How to Reproduce
```python
import joblib
model = joblib.load("models/final_house_price_model_week4.joblib")
predictions = model.predict(X_new)
```

## Author
<Fajar Sarfraz> — ASL Internship, AI/ML Track, Week 4
## Screenshots

### Screenshot 1
![Screenshot 1](https://github.com/Fajar-Sarfraz/week_4/blob/2ff4b1c72d3580b98671cb6c86439ed9ba056b83/week_4_ss/Screenshot%20(78).png))

### Screenshot 2
![Screenshot 2](https://github.com/Fajar-Sarfraz/week_4/blob/2ff4b1c72d3580b98671cb6c86439ed9ba056b83/week_4_ss/Screenshot%20(77).png)

### Screenshot 3
![Screenshot 3](https://github.com/Fajar-Sarfraz/week_4/blob/2ff4b1c72d3580b98671cb6c86439ed9ba056b83/week_4_ss/Screenshot%20(76).png)

### Screenshot 4
![Screenshot 4](https://github.com/Fajar-Sarfraz/week_4/blob/2ff4b1c72d3580b98671cb6c86439ed9ba056b83/week_4_ss/Screenshot%20(75).png)
