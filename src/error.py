from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ============================================================
# PROJECT PATHS 
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
# ============================================================
# 1. Load trained model
# ============================================================

print("=" * 60)
print("MODEL ERROR ANALYSIS")
print("=" * 60)



model = joblib.load(MODEL_DIR / "model.pkl")

print("\nModel loaded successfully.")
print("Model:", type(model).__name__)


# ============================================================
# 2. Load dataset
# ============================================================

df = pd.read_excel(DATA_DIR / "delaney.xlsx")

df.columns = df.columns.str.strip()

smiles_col = "SMILES"
target_col = "measured log"

print("\nDataset loaded.")
print("Number of molecules:", len(df))


# ============================================================
# 3. Morgan fingerprint generator
# ============================================================

morgan = GetMorganGenerator(
    radius=2,
    fpSize=512
)


# ============================================================
# 4. Feature generation
# ============================================================

def featurize(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # Molecular descriptors
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol)
    ]

    # Morgan fingerprint
    fingerprint = list(
        morgan.GetFingerprint(mol)
    )

    return descriptors + fingerprint


# ============================================================
# 5. Generate complete dataset
# ============================================================

features = []
targets = []
smiles_list = []
compound_ids = []

for _, row in df.iterrows():

    feat = featurize(row[smiles_col])

    if feat is not None:

        features.append(feat)
        targets.append(row[target_col])
        smiles_list.append(row[smiles_col])
        compound_ids.append(row["Compound ID"])


X = np.array(features)
y = np.array(targets)

print("\nFeature matrix:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 6. Recreate SAME train/test split
# ============================================================

X_train, X_test, y_train, y_test, smiles_train, smiles_test, ids_train, ids_test = train_test_split(
    X,
    y,
    smiles_list,
    compound_ids,
    test_size=0.2,
    random_state=42
)

print("\nTest set size:", len(y_test))


# ============================================================
# 7. Predict using saved model
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 8. Calculate errors
# ============================================================

residuals = y_test - y_pred

absolute_errors = np.abs(residuals)

squared_errors = residuals ** 2


# ============================================================
# 9. Overall metrics
# ============================================================

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nRMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")


# ============================================================
# 10. Create error dataframe
# ============================================================

results = pd.DataFrame({

    "Compound ID": ids_test,

    "SMILES": smiles_test,

    "Actual logS": y_test,

    "Predicted logS": y_pred,

    "Residual": residuals,

    "Absolute Error": absolute_errors

})


# ============================================================
# 11. Sort by worst prediction
# ============================================================

worst_predictions = results.sort_values(
    by="Absolute Error",
    ascending=False
)


# ============================================================
# 12. Display worst 20 molecules
# ============================================================

print("\n" + "=" * 60)
print("20 WORST PREDICTIONS")
print("=" * 60)

print(
    worst_predictions.head(20).to_string(
        index=False
    )
)


# ============================================================
# 13. Save all predictions
# ============================================================

results.to_csv(
    TABLES_DIR / "error_analysis_results.csv",
    index=False
)

print(
    "\nSaved: error_analysis_results.csv"
)


# ============================================================
# 14. Save worst predictions
# ============================================================

worst_predictions.head(20).to_csv(
    TABLES_DIR / "worst_predictions.csv",
    index=False
)

print(
    "Saved: worst_predictions.csv"
)


# ============================================================
# 15. Plot 1 — Actual vs Predicted
# ============================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

# Perfect prediction line
min_value = min(
    y_test.min(),
    y_pred.min()
)

max_value = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual logS")
plt.ylabel("Predicted logS")

plt.title(
    "Actual vs Predicted Solubility"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "error_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved: error_actual_vs_predicted.png"
)


# ============================================================
# 16. Plot 2 — Residual Distribution
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    residuals,
    bins=25,
    edgecolor="black"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Residual (Actual - Predicted)")

plt.ylabel("Number of Molecules")

plt.title(
    "Distribution of Prediction Errors"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "residual_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved: residual_distribution.png"
)


# ============================================================
# 17. Plot 3 — Residual vs Predicted
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted logS")

plt.ylabel(
    "Residual (Actual - Predicted)"
)

plt.title(
    "Residuals vs Predicted Solubility"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "residuals_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved: residuals_vs_predicted.png"
)


# ============================================================
# 18. Error statistics
# ============================================================

print("\n" + "=" * 60)
print("ERROR STATISTICS")
print("=" * 60)

print(
    f"\nMean residual: {residuals.mean():.4f}"
)

print(
    f"Median absolute error: "
    f"{np.median(absolute_errors):.4f}"
)

print(
    f"Maximum absolute error: "
    f"{absolute_errors.max():.4f}"
)

print(
    f"Minimum absolute error: "
    f"{absolute_errors.min():.4f}"
)


# ============================================================
# 19. Final summary
# ============================================================

print("\n" + "=" * 60)
print("ERROR ANALYSIS COMPLETED")
print("=" * 60)

print("""
Generated files:

1. error_analysis_results.csv
2. worst_predictions.csv
3. error_actual_vs_predicted.png
4. residual_distribution.png
5. residuals_vs_predicted.png
""")