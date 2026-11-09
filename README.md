# Early Breast Cancer Detection Using Discriminant Analysis and Imbalanced-Data Strategies

**Course:** Statistical Machine Learning for Data Science (Module 5: Discriminant Analysis)

## 1. Project Overview
In cancer screening, missing a malignant case is far more serious than a false alarm. This project studies how well Fisher's Linear Discriminant Analysis (LDA) and logistic regression (a generalized linear model) detect malignant tumours when malignant cases are rare, and which strategies improve detection.

## 2. Objectives
1. Create an artificially imbalanced dataset (about 10% malignant).
2. Analyse feature relationships with covariance, correlation and VIF.
3. Handle multicollinearity.
4. Apply Fisher's LDA and logistic regression (GLM).
5. Interpret coefficients and odds ratios (with confidence intervals and p-values).
6. Compare imbalance strategies: equal priors, threshold tuning, class weighting.
7. Evaluate reliably using repeated cross-validation, with malignant recall as the main metric.

## 3. Dataset
Breast Cancer Wisconsin (Diagnostic), UCI repository, loaded with `sklearn.datasets.load_breast_cancer()`.
569 samples, 30 features computed from digitized FNA images of breast masses (212 malignant, 357 benign).

**Artificial imbalance:** all 357 benign + 40 randomly chosen malignant samples (`random_state=42`) = **397 samples, 10.1% malignant**.
Split: 80/20 stratified (317 train / 80 test). The test set has only **8 malignant cases**.

## 4. How to Run
```bash
pip install -r requirements.txt
python main.py
```
All tables and plots are saved in `results/`.

## 5. Methodology
| Step | What was done |
|---|---|
| Scaling | `StandardScaler` fitted on training data only (no leakage) |
| EDA | Covariance, correlation heatmap and top correlated pairs (training data) |
| Multicollinearity | 18 feature pairs with correlation above 0.90. Iteratively dropped the highest-VIF feature until all VIF < 5, leaving **13 of 30 features** |
| Models | LDA (default priors), LDA (equal priors), Logistic Regression, Class-Weighted Logistic |
| Evaluation | 5-fold stratified CV repeated 10 times (50 folds) plus a single hold-out test |
| GLM interpretation | Binomial GLM (statsmodels) on 5 weakly correlated features, giving odds ratios with 95% CI and p-values |
| Threshold tuning | Lowering the decision threshold on LDA probabilities |

## 6. Results

### 6.1 Cross-validated comparison (reduced features, mean over 50 folds)
| Model | Accuracy | Precision | Malignant Recall | F1 |
|---|---|---|---|---|
| LDA (default priors) | 0.980 | 1.00 | 0.81 ± 0.12 | 0.89 |
| LDA (equal priors) | 0.984 | 0.95 | 0.89 ± 0.10 | 0.91 |
| Logistic Regression | 0.989 | 0.99 | 0.90 ± 0.11 | 0.94 |
| Class-Weighted Logistic | 0.976 | 0.85 | 0.95 ± 0.07 | 0.89 |

Full results for all 30 features are in `results/model_comparison_cv.csv`.

### 6.2 Hold-out test (80 samples, 8 malignant)
LDA (default) detected 6/8, LDA (equal priors) 7/8, Logistic and Class-Weighted Logistic 8/8. With so few malignant cases, one case changes recall by 12.5 percentage points, so the cross-validated numbers in 6.1 are more reliable.

### 6.3 Threshold tuning (LDA, hold-out)
| Threshold | Precision | Recall | False alarms |
|---|---|---|---|
| 0.50 | 1.00 | 0.75 | 0 |
| 0.10 | 1.00 | 0.875 | 0 |
| 0.01 | 1.00 | 1.00 | 0 |
| 0.001 | 0.80 | 1.00 | 2 |

### 6.4 GLM odds ratios (per 1 std-dev increase)
| Feature | Odds Ratio | 95% CI | p-value |
|---|---|---|---|
| mean area | 37.5 | 7.0 to 202.3 | < 0.001 |
| mean texture | 6.9 | 2.2 to 21.4 | 0.001 |
| mean smoothness | 4.7 | 1.6 to 14.0 | 0.006 |
| mean concavity | 2.7 | 1.3 to 5.5 | 0.010 |
| mean symmetry | 2.1 | 0.7 to 6.1 | 0.169 (not significant) |

## 7. Key Findings
1. **Accuracy is misleading on imbalanced data.** LDA has about 98% accuracy but misses roughly 1 in 5 malignant cases.
2. **Default LDA is biased toward the majority class** because it uses the 90/10 training priors. Setting equal priors raised recall by about 8 points.
3. **Lowering the decision threshold** raised LDA recall on the hold-out set from 0.75 to 1.00 with no false alarms until the threshold became very small.
4. **Class weighting gives the highest recall (0.95) but lower precision (0.85).** It trades more false alarms for fewer missed cancers, which is often acceptable in screening.
5. **Tumour size (mean area) is the strongest predictor**, followed by texture and smoothness.
6. **Multicollinearity matters.** Radius, perimeter and area are almost perfectly correlated (VIF up to about 3700), which makes coefficients unstable. After VIF pruning, coefficients are interpretable, though overall predictive performance does not improve.

## 8. Limitations
- The imbalance is artificial. A truly rare-event dataset may behave differently.
- Small dataset: 40 malignant samples in total, so confidence intervals are wide.
- The data is nearly linearly separable, so an unpenalised GLM on many features does not converge to sensible estimates. A 5-feature GLM was used for interpretation.
- Odds ratios describe association in this dataset, not causation.
- This is an academic exercise and **not a clinical diagnostic tool**.

## 9. Project Structure
```
Breast_Cancer_Detection/
├── main.py
├── requirements.txt
├── README.md
└── results/   (CSV tables and PNG plots)
```

## 10. Conclusion
On imbalanced medical data, overall accuracy hides missed cancer cases. Adjusting LDA priors, tuning the decision threshold, or using class weights improves malignant recall, at some cost in precision. The best choice depends on how costly a missed case is compared with a false alarm.
