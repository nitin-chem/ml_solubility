from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Create output directories if they do not exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_excel(DATA_DIR / "delaney.xlsx")

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()

print(df.head())
print(df.columns)


# ============================================================
# 2. SELECT COLUMNS
# ============================================================

smiles_col = "SMILES"
target_col = "measured log"

print("SMILES column:", smiles_col)
print("Target column:", target_col)


# ============================================================
# 3. MOLECULAR FEATURE GENERATION
# ============================================================

# Morgan fingerprint generator
morgan = GetMorganGenerator(
    radius=2,
    fpSize=512
)


def featurize(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # Basic molecular descriptors
    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
    ]

    # Morgan fingerprint
    fp = list(morgan.GetFingerprint(mol))

    # Combine descriptors + fingerprint
    return desc + fp


# ============================================================
# 4. GENERATE FEATURES AND TARGETS
# ============================================================

features = []
targets = []

for _, row in df.iterrows():

    feats = featurize(row[smiles_col])

    if feats is not None:
        features.append(feats)
        targets.append(row[target_col])


X = np.array(features)
y = np.array(targets)

print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 6. HELPER FUNCTION FOR MODEL EVALUATION
# ============================================================

def evaluate(y_true, y_pred, name):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name} Results:")
    print("RMSE:", rmse)
    print("R2:", r2)
    print("-" * 30)

    return rmse, r2


# ============================================================
# 7. BASELINE MODELS
# ============================================================

# -------------------------
# Linear Regression
# -------------------------

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

evaluate(
    y_test,
    lr_pred,
    "Linear Regression"
)


# -------------------------
# Random Forest
# -------------------------

rf_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

evaluate(
    y_test,
    rf_pred,
    "Random Forest"
)


# -------------------------
# Gradient Boosting
# -------------------------

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_test)

evaluate(
    y_test,
    gb_pred,
    "Gradient Boosting"
)


# ============================================================
# 8. 5-FOLD CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 50)
print("5-FOLD CROSS-VALIDATION")
print("=" * 50)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# -------------------------
# Random Forest CV
# -------------------------

rf_cv = cross_validate(
    rf_model,
    X_train,
    y_train,
    cv=kf,
    scoring={
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error"
    },
    n_jobs=-1
)

rf_r2 = rf_cv["test_r2"]
rf_rmse = -rf_cv["test_rmse"]

print("\nRandom Forest Cross-Validation:")

print("R² scores:", rf_r2)
print("Mean R²:", rf_r2.mean())
print("Std R²:", rf_r2.std())

print("\nRMSE scores:", rf_rmse)
print("Mean RMSE:", rf_rmse.mean())
print("Std RMSE:", rf_rmse.std())


# -------------------------
# Gradient Boosting CV
# -------------------------

gb_cv = cross_validate(
    gb_model,
    X_train,
    y_train,
    cv=kf,
    scoring={
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error"
    },
    n_jobs=-1
)

gb_r2 = gb_cv["test_r2"]
gb_rmse = -gb_cv["test_rmse"]

print("\nGradient Boosting Cross-Validation:")

print("R² scores:", gb_r2)
print("Mean R²:", gb_r2.mean())
print("Std R²:", gb_r2.std())

print("\nRMSE scores:", gb_rmse)
print("Mean RMSE:", gb_rmse.mean())
print("Std RMSE:", gb_rmse.std())


# ============================================================
# 9. RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 50)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 50)


rf_param_grid = {

    "n_estimators": [
        200, 300, 400, 500, 600
    ],

    "max_depth": [
        None, 10, 15, 20, 25, 30
    ],

    "min_samples_split": [
        2, 3, 5, 8, 10
    ],

    "min_samples_leaf": [
        1, 2, 4, 6
    ],

    "max_features": [
        "sqrt",
        "log2",
        0.5,
        0.75
    ]
}


rf_base = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)


rf_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=rf_param_grid,
    n_iter=30,
    scoring="neg_root_mean_squared_error",
    cv=kf,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


rf_search.fit(X_train, y_train)


print("\nBest Random Forest Parameters:")
print(rf_search.best_params_)

print("\nBest Random Forest CV RMSE:")
print(-rf_search.best_score_)


best_rf = rf_search.best_estimator_

best_rf_pred = best_rf.predict(X_test)

evaluate(
    y_test,
    best_rf_pred,
    "Tuned Random Forest"
)


# ============================================================
# 10. GRADIENT BOOSTING HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 50)
print("GRADIENT BOOSTING HYPERPARAMETER TUNING")
print("=" * 50)


gb_param_grid = {

    "n_estimators": [
        100, 150, 200, 300, 400, 500
    ],

    "learning_rate": [
        0.01, 0.03, 0.05, 0.08, 0.1
    ],

    "max_depth": [
        2, 3, 4, 5
    ],

    "min_samples_split": [
        2, 3, 5, 8
    ],

    "min_samples_leaf": [
        1, 2, 4, 6
    ],

    "subsample": [
        0.7, 0.8, 0.9, 1.0
    ]
}


gb_base = GradientBoostingRegressor(
    random_state=42
)


gb_search = RandomizedSearchCV(
    estimator=gb_base,
    param_distributions=gb_param_grid,
    n_iter=30,
    scoring="neg_root_mean_squared_error",
    cv=kf,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


gb_search.fit(X_train, y_train)


print("\nBest Gradient Boosting Parameters:")
print(gb_search.best_params_)

print("\nBest Gradient Boosting CV RMSE:")
print(-gb_search.best_score_)


best_gb = gb_search.best_estimator_


# ============================================================
# 11. FINAL MODEL EVALUATION
# ============================================================

best_gb_pred = best_gb.predict(X_test)

final_rmse, final_r2 = evaluate(
    y_test,
    best_gb_pred,
    "FINAL TUNED GRADIENT BOOSTING"
)


# ============================================================
# 12. SAVE FINAL MODEL
# ============================================================

joblib.dump(
    best_gb,
    MODELS_DIR / "model.pkl"
)

# ============================================================
# 13. SAVE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": best_gb_pred
})

results.to_csv(
    TABLES_DIR / "predictions.csv",
    index=False
)

print(f"Predictions saved as: {TABLES_DIR / 'predictions.csv'}")


# ============================================================
# 14. ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(figsize=(6, 6))

plt.scatter(
    y_test,
    best_gb_pred,
    alpha=0.6
)

# Perfect prediction line
min_value = min(
    y_test.min(),
    best_gb_pred.min()
)

max_value = max(
    y_test.max(),
    best_gb_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    "r--"
)

plt.xlabel("Actual logS")
plt.ylabel("Predicted logS")

plt.title(
    "Tuned Gradient Boosting: Actual vs Predicted"
)

plt.grid()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.show()


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("FINAL MODEL SUMMARY")
print("=" * 50)

print("Model: Tuned Gradient Boosting")
print("Test RMSE:", final_rmse)
print("Test R²:", final_r2)
print("CV RMSE:", -gb_search.best_score_)
print(f"Model saved: {MODELS_DIR / 'model.pkl'}")
print(f"Predictions saved: {TABLES_DIR / 'predictions.csv'}")
print(f"Plot saved: {FIGURES_DIR / 'actual_vs_predicted.png'}")
