"""
Early Breast Cancer Detection using Discriminant Analysis
and Imbalanced-Data Strategies
Course: Statistical Machine Learning for Data Science (Module 5)

Run:  python main.py      (outputs go to the results/ folder)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

RANDOM_STATE = 42
os.makedirs("results", exist_ok=True)


def header(title):
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


# ----------------------------------------------------------------
# PHASE 1: DATA + ARTIFICIAL CLASS IMBALANCE (~10% malignant)
# ----------------------------------------------------------------
header("PHASE 1: Data preparation")
raw = load_breast_cancer(as_frame=True)
df = raw.data.copy()
df["malignant"] = (raw.target == 0).astype(int)      # sklearn: 0 = malignant

malignant = df[df["malignant"] == 1].sample(n=40, random_state=RANDOM_STATE)
benign = df[df["malignant"] == 0]
data = (pd.concat([benign, malignant])
        .sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True))
data.to_csv("results/imbalanced_breast_cancer.csv", index=False)

X, y = data.drop(columns="malignant"), data["malignant"]
print(f"Samples: {len(y)} | Malignant: {y.sum()} ({y.mean():.1%}) | Benign: {(y == 0).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print(f"Train: {len(y_train)} ({y_train.sum()} malignant) | "
      f"Test: {len(y_test)} ({y_test.sum()} malignant)")
print("NOTE: the hold-out test set has very few malignant cases, so we also "
      "use repeated cross-validation for reliable comparison.")

# ----------------------------------------------------------------
# PHASE 2: EDA  (training data only -> no information from test set)
# ----------------------------------------------------------------
header("PHASE 2: Covariance & correlation (training data)")
scaler = StandardScaler().fit(X_train)
Xtr_s = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
Xte_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

cov = X_train.cov()
corr = X_train.corr()
print("Covariance matrix shape:", cov.shape)
print("Mean radius vs mean area correlation :", round(corr.loc["mean radius", "mean area"], 3))
print("Worst radius vs worst area correlation:", round(corr.loc["worst radius", "worst area"], 3))

pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
print("Feature pairs with |r| > 0.90:", int((pairs.abs() > 0.90).sum()))
print("\nTop 10 correlated pairs:\n", pairs.sort_values(ascending=False).head(10).round(3))

plt.figure(figsize=(14, 10))
plt.imshow(corr, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(30), corr.columns, rotation=90, fontsize=7)
plt.yticks(range(30), corr.columns, fontsize=7)
plt.title("Correlation Heatmap (training data)")
plt.tight_layout(); plt.savefig("results/correlation_heatmap.png", dpi=200); plt.close()

# ----------------------------------------------------------------
# PHASE 3: HANDLE MULTICOLLINEARITY (iteratively drop highest-VIF feature until all VIF < 5)
# ----------------------------------------------------------------
header("PHASE 3: Multicollinearity -> reduced feature set (VIF < 5)")
def vif_series(cols):
    return pd.Series([variance_inflation_factor(Xtr_s[cols].values, i)
                      for i in range(len(cols))], index=cols)

vif_before = vif_series(list(X.columns)).sort_values(ascending=False)
print("Highest VIF with all 30 features:\n", vif_before.head(5).round(0))
reduced = list(X.columns)
while vif_series(reduced).max() >= 5:
    reduced.remove(vif_series(reduced).idxmax())
vif_after = vif_series(reduced).sort_values(ascending=False)
vif_after.round(2).to_csv("results/vif_reduced_features.csv", header=["VIF"])
print(f"\nKept {len(reduced)} of 30 features (max VIF now {vif_after.max():.2f}):\n  " + ", ".join(reduced))

# ----------------------------------------------------------------
# PHASE 4: MODELS
# ----------------------------------------------------------------
def make_models():
    return {
        "LDA (default priors)":      LinearDiscriminantAnalysis(),
        "LDA (equal priors)":        LinearDiscriminantAnalysis(priors=[0.5, 0.5]),
        "Logistic Regression":       LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Class-Weighted Logistic":   LogisticRegression(class_weight="balanced", max_iter=2000,
                                                        random_state=RANDOM_STATE),
    }

# ----------------------------------------------------------------
# PHASE 5: RELIABLE COMPARISON - repeated stratified 5-fold CV
# ----------------------------------------------------------------
header("PHASE 5: Model comparison (5-fold CV x 10 repeats)")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}
rows = []
for fs_name, cols in [("All 30 features", list(X.columns)), (f"Reduced ({len(reduced)})", reduced)]:
    for name, model in make_models().items():
        res = cross_validate(make_pipeline(StandardScaler(), model), X[cols], y,
                             cv=cv, scoring=scoring)
        rows.append({"Feature set": fs_name, "Model": name,
                     "Accuracy": res["test_accuracy"].mean(),
                     "Precision": res["test_precision"].mean(),
                     "Malignant Recall": res["test_recall"].mean(),
                     "Recall Std": res["test_recall"].std(),
                     "F1": res["test_f1"].mean()})
comparison = pd.DataFrame(rows)
comparison.round(3).to_csv("results/model_comparison_cv.csv", index=False)
print(comparison.round(3).to_string(index=False))

# bar chart: malignant recall with error bars
fig, ax = plt.subplots(figsize=(10, 6))
w = 0.38
for i, (fs, grp) in enumerate(comparison.groupby("Feature set", sort=False)):
    ax.bar(np.arange(len(grp)) + i * w, grp["Malignant Recall"], w,
           yerr=grp["Recall Std"], capsize=4, label=fs)
ax.set_xticks(np.arange(4) + w / 2)
ax.set_xticklabels(list(make_models().keys()), rotation=12)
ax.set_ylim(0, 1.15); ax.set_ylabel("Malignant recall (mean ± std over 50 folds)")
ax.set_title("Cross-validated malignant recall"); ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig("results/model_comparison.png", dpi=200); plt.close()

# ----------------------------------------------------------------
# PHASE 6: HOLD-OUT RESULTS (reduced features) + confusion matrices
# ----------------------------------------------------------------
header("PHASE 6: Hold-out test results (reduced features)")
fitted, hold_rows = {}, []
for name, model in make_models().items():
    model.fit(Xtr_s[reduced], y_train)
    fitted[name] = model
    pred = model.predict(Xte_s[reduced])
    hold_rows.append({"Model": name,
                      "Accuracy": accuracy_score(y_test, pred),
                      "Precision": precision_score(y_test, pred, zero_division=0),
                      "Malignant Recall": recall_score(y_test, pred),
                      "F1": f1_score(y_test, pred)})
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                           display_labels=["Benign", "Malignant"]).plot()
    plt.title(name); plt.tight_layout()
    plt.savefig("results/cm_" + name.split(" (")[0].replace(" ", "_").lower()
                + ("_equal_priors" if "equal" in name else "") + ".png", dpi=200)
    plt.close()
hold = pd.DataFrame(hold_rows)
hold.round(3).to_csv("results/model_comparison_holdout.csv", index=False)
print(hold.round(3).to_string(index=False))
print(f"(Only {int(y_test.sum())} malignant cases in the test set - interpret with caution.)")

# ----------------------------------------------------------------
# PHASE 7: FISHER'S LDA - coefficients and class separation
# ----------------------------------------------------------------
header("PHASE 7: Fisher's LDA")
lda = fitted["LDA (default priors)"]
lda_coef = (pd.DataFrame({"Feature": reduced, "LDA_Coefficient": lda.coef_[0]})
            .sort_values("LDA_Coefficient", key=abs, ascending=False))
lda_coef.to_csv("results/lda_coefficients.csv", index=False)
print(lda_coef.round(3).to_string(index=False))

z = lda.transform(Xte_s[reduced]).ravel()
plt.figure(figsize=(9, 5))
plt.hist(z[y_test == 0], bins=15, alpha=0.6, label="Benign")
plt.hist(z[y_test == 1], bins=15, alpha=0.6, label="Malignant")
plt.xlabel("LDA discriminant score"); plt.ylabel("Samples")
plt.title("LDA class separation (test set)"); plt.legend()
plt.tight_layout(); plt.savefig("results/lda_class_separation.png", dpi=200); plt.close()

# ----------------------------------------------------------------
# PHASE 8: GLM (Binomial / logit) - odds ratios with CI and p-values
# ----------------------------------------------------------------
header("PHASE 8: Logistic GLM - odds ratios (per 1 std-dev increase)")
# This dataset is almost linearly separable, so an un-penalised GLM with many
# features does not converge to sensible estimates (huge odds ratios, p ~ 1).
# For an interpretable GLM we therefore use 5 clinically meaningful, weakly
# correlated features (all VIF < 2).
glm_features = ["mean texture", "mean area", "mean smoothness", "mean concavity", "mean symmetry"]
print("GLM max VIF:", round(vif_series(glm_features).max(), 2))
glm = sm.GLM(y_train, sm.add_constant(Xtr_s[glm_features]),
             family=sm.families.Binomial()).fit()
ci = np.exp(glm.conf_int())
odds = pd.DataFrame({"Feature": glm_features,
                     "Odds_Ratio": np.exp(glm.params[glm_features]).values,
                     "CI_low": ci.loc[glm_features, 0].values,
                     "CI_high": ci.loc[glm_features, 1].values,
                     "p_value": glm.pvalues[glm_features].values})
odds = odds.sort_values("Odds_Ratio", ascending=False)
odds.to_csv("results/logistic_odds_ratios.csv", index=False)
print(odds.round(3).to_string(index=False))

top = odds.iloc[::-1]
plt.figure(figsize=(8, 4.5))
plt.errorbar(top["Odds_Ratio"], range(len(top)),
             xerr=[top["Odds_Ratio"] - top["CI_low"], top["CI_high"] - top["Odds_Ratio"]],
             fmt="o", capsize=4)
plt.axvline(1, color="grey", ls="--")
plt.yticks(range(len(top)), top["Feature"]); plt.xscale("log")
plt.xlabel("Odds ratio per 1 std-dev increase (log scale, 95% CI)")
plt.title("GLM odds ratios for malignancy")
plt.tight_layout(); plt.savefig("results/odds_ratios.png", dpi=200); plt.close()

# ----------------------------------------------------------------
# PHASE 9: THRESHOLD TUNING on LDA (the model that misses cases)
# ----------------------------------------------------------------
header("PHASE 9: Threshold tuning on LDA (default priors)")
proba = lda.predict_proba(Xte_s[reduced])[:, 1]
thresholds = [0.5, 0.3, 0.1, 0.01, 0.001]
t_rows = []
for t in thresholds:
    p = (proba >= t).astype(int)
    t_rows.append({"Threshold": t,
                   "Precision": precision_score(y_test, p, zero_division=0),
                   "Recall": recall_score(y_test, p),
                   "F1": f1_score(y_test, p, zero_division=0),
                   "False alarms": int(((p == 1) & (y_test == 0)).sum())})
thr = pd.DataFrame(t_rows)
thr.round(3).to_csv("results/threshold_tuning_lda.csv", index=False)
print(thr.round(3).to_string(index=False))

plt.figure(figsize=(8, 5))
xs = range(len(thresholds))
plt.plot(xs, thr["Precision"], "o-", label="Precision")
plt.plot(xs, thr["Recall"], "o-", label="Malignant recall")
plt.xticks(xs, thresholds); plt.ylim(0, 1.1); plt.grid(True)
plt.xlabel("Probability threshold"); plt.ylabel("Score")
plt.title("LDA: effect of lowering the decision threshold"); plt.legend()
plt.tight_layout(); plt.savefig("results/threshold_performance.png", dpi=200); plt.close()

# ----------------------------------------------------------------
# PHASE 10: KEY FINDINGS (computed, not hard-coded)
# ----------------------------------------------------------------
header("KEY FINDINGS (cross-validated, reduced features)")
red = comparison[comparison["Feature set"].str.startswith("Reduced")].set_index("Model")
for m, r in red.iterrows():
    print(f"- {m:26s} recall {r['Malignant Recall']:.2f} ± {r['Recall Std']:.2f}, "
          f"precision {r['Precision']:.2f}, F1 {r['F1']:.2f}")
best = red["Malignant Recall"].idxmax()
print(f"\nHighest malignant recall: {best} ({red.loc[best, 'Malignant Recall']:.2f}).")
gain = red.loc['LDA (equal priors)', 'Malignant Recall'] - red.loc['LDA (default priors)', 'Malignant Recall']
print(f"Using equal priors changed LDA recall by {gain:+.2f}.")
print("\nAll results saved in the results/ folder.")
