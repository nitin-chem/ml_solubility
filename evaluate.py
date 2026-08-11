import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

model = joblib.load("model.pkl")

print("\nModel loaded successfully.")
print("Model:", type(model).__name__)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_excel("delaney.xlsx")

df.columns = df.columns.str.strip()

smiles_col = "SMILES"
target_col = "measured log"

print("\nDataset loaded.")
print("Number of molecules:", len(df))


# ============================================================
# 3. FEATURE GENERATION
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
        Descriptors.TPSA(mol)
    ]

    fingerprint = list(
        morgan.GetFingerprint(mol)
    )

    return descriptors + fingerprint


features = []
targets = []

for _, row in df.iterrows():

    feats = featurize(row[smiles_col])

    if feats is not None:

        features.append(feats)
        targets.append(row[target_col])


X = np.array(features)
y = np.array(targets)

print("\nFeature matrix:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 4. SAME TRAIN/TEST SPLIT USED DURING MODEL DEVELOPMENT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining molecules:", len(X_train))
print("Test molecules:", len(X_test))


# ============================================================
# 5. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 6. CALCULATE METRICS
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


# ============================================================
# 7. DISPLAY FINAL PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)

print(f"\nModel: GradientBoostingRegressor")

print(f"Test RMSE : {rmse:.4f}")
print(f"Test MAE  : {mae:.4f}")
print(f"Test R²   : {r2:.4f}")


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Metric": [
        "Test RMSE",
        "Test MAE",
        "Test R2"
    ],
    "Value": [
        rmse,
        mae,
        r2
    ]
})

results.to_csv(
    "final_model_performance.csv",
    index=False
)

print("\nSaved:")
print("final_model_performance.csv")


# ============================================================
# 9. SAVE PREDICTIONS
# ============================================================

prediction_results = pd.DataFrame({
    "Actual_logS": y_test,
    "Predicted_logS": y_pred,
    "Residual": y_test - y_pred,
    "Absolute_Error": np.abs(y_test - y_pred)
})

prediction_results.to_csv(
    "final_model_predictions.csv",
    index=False
)

print("final_model_predictions.csv")

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)