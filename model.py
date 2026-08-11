import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)

# ---------------------------------------
# 1. Load dataset
# ---------------------------------------

df = pd.read_excel("delaney.xlsx")

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()

print(df.head())
print(df.columns)

# ---------------------------------------
# 2. Select columns
# ---------------------------------------



smiles_col = "SMILES"

target_col = "measured log"

print("SMILES column:", smiles_col)
print("Target column:", target_col)


# ---------------------------------------
# 3. Convert SMILES into molecular features
# ---------------------------------------

from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

morgan = GetMorganGenerator(radius=2, fpSize=512)

def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Basic descriptors
    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
    ]

    # Fingerprints (reduced size)
    fp = list(morgan.GetFingerprint(mol))

    return desc + fp   # combine both
# ---------------------------------------
# 4. Generate features and targets
# ---------------------------------------

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


# ---------------------------------------
# 5. Train-test split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------
# 6. Linear Regression
# ---------------------------------------

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)


# ---------------------------------------
# 7. Random Forest
# ---------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# ---------------------------------------
# 8. Evaluation
# ---------------------------------------

def evaluate(y_true, y_pred, name):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name} Results:")
    print("RMSE:", rmse)
    print("R2:", r2)
    print("-" * 30)


evaluate(y_test, lr_pred, "Linear Regression")

evaluate(y_test, rf_pred, "Random Forest")
evaluate(y_test, gb_pred, "Gradient Boosting")

# ============================================================
# 5-Fold Cross-Validation
# ============================================================

print("\n" + "=" * 50)
print("5-FOLD CROSS-VALIDATION")
print("=" * 50)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Random Forest
rf_cv = cross_validate(
    rf_model,
    X,
    y,
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


# Gradient Boosting
gb_cv = cross_validate(
    gb_model,
    X,
    y,
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
# Random Forest Hyperparameter Tuning
# ============================================================

print("\n" + "=" * 50)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 50)

# Parameter search space
param_grid = {
    "n_estimators": [200, 300, 400, 500, 600],
    "max_depth": [None, 10, 15, 20, 25, 30],
    "min_samples_split": [2, 3, 5, 8, 10],
    "min_samples_leaf": [1, 2, 4, 6],
    "max_features": ["sqrt", "log2", 0.5, 0.75]
}

# Cross-validation strategy
kf_tuning = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Base Random Forest
rf_base = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

# Randomized search
rf_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_grid,
    n_iter=30,
    scoring="neg_root_mean_squared_error",
    cv=kf_tuning,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# IMPORTANT:
# Tune only on the training data
rf_search.fit(X_train, y_train)

# Best parameters
print("\nBest Parameters:")
print(rf_search.best_params_)

# Best cross-validation RMSE
print("\nBest CV RMSE:")
print(-rf_search.best_score_)

# Best model
best_rf = rf_search.best_estimator_

# Evaluate on untouched test set
best_rf_pred = best_rf.predict(X_test)

print("\nTuned Random Forest Test Results:")
evaluate(y_test, best_rf_pred, "Tuned Random Forest")

# ============================================================
# Gradient Boosting Hyperparameter Tuning
# ============================================================

print("\n" + "=" * 50)
print("GRADIENT BOOSTING HYPERPARAMETER TUNING")
print("=" * 50)

gb_param_grid = {
    "n_estimators": [100, 150, 200, 300, 400, 500],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "max_depth": [2, 3, 4, 5],
    "min_samples_split": [2, 3, 5, 8],
    "min_samples_leaf": [1, 2, 4, 6],
    "subsample": [0.7, 0.8, 0.9, 1.0]
}

gb_base = GradientBoostingRegressor(
    random_state=42
)

gb_search = RandomizedSearchCV(
    estimator=gb_base,
    param_distributions=gb_param_grid,
    n_iter=30,
    scoring="neg_root_mean_squared_error",
    cv=kf_tuning,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Tune only on training data
gb_search.fit(X_train, y_train)

print("\nBest Gradient Boosting Parameters:")
print(gb_search.best_params_)

print("\nBest Gradient Boosting CV RMSE:")
print(-gb_search.best_score_)

# Best model
best_gb = gb_search.best_estimator_

# Evaluate on untouched test set
best_gb_pred = best_gb.predict(X_test)

print("\nTuned Gradient Boosting Test Results:")
evaluate(y_test, best_gb_pred, "Tuned Gradient Boosting")

# ---------------------------------------
# 9. Plot actual vs predicted
# ---------------------------------------

plt.figure(figsize=(6,6))
plt.scatter(y_test, rf_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Random Forest Predictions")
plt.grid()
plt.show()

# Best model
best_gb = gb_search.best_estimator_

# Save the tuned model
joblib.dump(best_gb, "model.pkl")

print("\nFinal model saved as model.pkl")

# Evaluate on untouched test set
best_gb_pred = best_gb.predict(X_test)

print("\nTuned Gradient Boosting Test Results:")
evaluate(y_test, best_gb_pred, "Tuned Gradient Boosting")

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": best_gb_pred
})

results.to_csv("predictions.csv", index=False)