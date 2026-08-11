import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import Descriptors

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("ESOL vs MACHINE LEARNING MODEL COMPARISON")
print("=" * 60)

df = pd.read_excel("delaney.xlsx")

df.columns = df.columns.str.strip()

print("\nDataset loaded.")
print("Number of molecules:", len(df))


# ============================================================
# 2. COLUMN NAMES
# ============================================================

smiles_col = "SMILES"
target_col = "measured log"

# IMPORTANT:
# Change this if your actual ESOL column has a different name.

esol_col = "ESOL predicted log(solubility:mol/L)"

print("\nColumns:")
print("SMILES:", smiles_col)
print("Target:", target_col)
print("ESOL:", esol_col)


# ============================================================
# 3. GENERATE SAME FEATURES AS MODEL
# ============================================================

morgan = GetMorganGenerator(
    radius=2,
    fpSize=512
)


def featurize(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
    ]

    fingerprint = list(
        morgan.GetFingerprint(mol)
    )

    return descriptors + fingerprint


# ============================================================
# 4. GENERATE FEATURE MATRIX
# ============================================================

features = []
targets = []
esol_values = []
valid_rows = []

for index, row in df.iterrows():

    feats = featurize(row[smiles_col])

    if feats is not None:

        features.append(feats)
        targets.append(row[target_col])
        esol_values.append(row[esol_col])
        valid_rows.append(index)


X = np.array(features)
y = np.array(targets)
esol = np.array(esol_values)


print("\nFeature matrix:", X.shape)
print("Target shape:", y.shape)
print("ESOL shape:", esol.shape)


# ============================================================
# 5. SAME TRAIN/TEST SPLIT AS MODEL
# ============================================================

X_train, X_test, y_train, y_test, esol_train, esol_test = train_test_split(
    X,
    y,
    esol,
    test_size=0.2,
    random_state=42
)


print("\nTest set size:", len(y_test))


# ============================================================
# 6. LOAD YOUR FINAL MODEL
# ============================================================

model = joblib.load("model.pkl")

print("\nFinal model loaded.")
print("Model:", type(model).__name__)


# ============================================================
# 7. MACHINE LEARNING PREDICTIONS
# ============================================================

ml_pred = model.predict(X_test)


# ============================================================
# 8. CALCULATE ML MODEL METRICS
# ============================================================

ml_rmse = np.sqrt(
    mean_squared_error(y_test, ml_pred)
)

ml_mae = mean_absolute_error(
    y_test,
    ml_pred
)

ml_r2 = r2_score(
    y_test,
    ml_pred
)


# ============================================================
# 9. CALCULATE ESOL METRICS
# ============================================================

esol_rmse = np.sqrt(
    mean_squared_error(y_test, esol_test)
)

esol_mae = mean_absolute_error(
    y_test,
    esol_test
)

esol_r2 = r2_score(
    y_test,
    esol_test
)


# ============================================================
# 10. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print("\nGradient Boosting Model:")
print(f"RMSE: {ml_rmse:.4f}")
print(f"MAE : {ml_mae:.4f}")
print(f"R²  : {ml_r2:.4f}")

print("\nESOL:")
print(f"RMSE: {esol_rmse:.4f}")
print(f"MAE : {esol_mae:.4f}")
print(f"R²  : {esol_r2:.4f}")


# ============================================================
# 11. CREATE COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "ESOL",
        "Gradient Boosting"
    ],

    "RMSE": [
        esol_rmse,
        ml_rmse
    ],

    "MAE": [
        esol_mae,
        ml_mae
    ],

    "R2": [
        esol_r2,
        ml_r2
    ]
})


print("\n" + "=" * 60)
print("COMPARISON TABLE")
print("=" * 60)

print(comparison.to_string(index=False))


# ============================================================
# 12. SAVE COMPARISON TABLE
# ============================================================

comparison.to_csv(
    "esol_model_comparison.csv",
    index=False
)

print("\nSaved:")
print("esol_model_comparison.csv")


# ============================================================
# 13. CALCULATE IMPROVEMENT
# ============================================================

rmse_improvement = (
    (esol_rmse - ml_rmse)
    / esol_rmse
) * 100

mae_improvement = (
    (esol_mae - ml_mae)
    / esol_mae
) * 100

print("\n" + "=" * 60)
print("IMPROVEMENT OF GRADIENT BOOSTING OVER ESOL")
print("=" * 60)

print(f"RMSE improvement: {rmse_improvement:.2f}%")
print(f"MAE improvement : {mae_improvement:.2f}%")


# ============================================================
# 14. ACTUAL vs PREDICTED PLOT
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    esol_test,
    alpha=0.6,
    label="ESOL"
)

plt.scatter(
    y_test,
    ml_pred,
    alpha=0.6,
    label="Gradient Boosting"
)

minimum = min(
    y_test.min(),
    esol_test.min(),
    ml_pred.min()
)

maximum = max(
    y_test.max(),
    esol_test.max(),
    ml_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "k--",
    label="Perfect Prediction"
)

plt.xlabel("Experimental logS")
plt.ylabel("Predicted logS")

plt.title(
    "ESOL vs Gradient Boosting"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "esol_vs_gradient_boosting.png",
    dpi=300
)

plt.show()

print("Saved:")
print("esol_vs_gradient_boosting.png")


# ============================================================
# 15. BAR CHART
# ============================================================

models = [
    "ESOL",
    "Gradient Boosting"
]

rmse_values = [
    esol_rmse,
    ml_rmse
]

plt.figure(figsize=(7, 5))

plt.bar(
    models,
    rmse_values
)

plt.ylabel("RMSE")
plt.title("RMSE Comparison")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "esol_vs_gradient_boosting_rmse.png",
    dpi=300
)

plt.show()

print("Saved:")
print("esol_vs_gradient_boosting_rmse.png")


# ============================================================
# 16. SAVE PREDICTIONS
# ============================================================

prediction_comparison = pd.DataFrame({

    "Experimental_logS": y_test,

    "ESOL_logS": esol_test,

    "GradientBoosting_logS": ml_pred

})

prediction_comparison.to_csv(
    "esol_vs_ml_predictions.csv",
    index=False
)

print("Saved:")
print("esol_vs_ml_predictions.csv")


print("\n" + "=" * 60)
print("ESOL COMPARISON COMPLETE")
print("=" * 60)