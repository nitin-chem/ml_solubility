from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_FILE = "model.pkl"
DATA_FILE = "delaney.xlsx"

SMILES_COL = "SMILES"
TARGET_COL = "measured log"

RANDOM_STATE = 42


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("=" * 60)
print("APPLICABILITY DOMAIN THRESHOLD VALIDATION")
print("=" * 60)

model = joblib.load(MODELS_DIR / MODEL_FILE)

print("\nModel loaded successfully.")
print("Model:", type(model).__name__)


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_excel(DATA_DIR / DATA_FILE)

df.columns = df.columns.str.strip()

print("\nDataset loaded.")
print("Number of molecules:", len(df))


# ============================================================
# 4. MORGAN FINGERPRINT GENERATOR
# ============================================================

morgan = GetMorganGenerator(
    radius=2,
    fpSize=512
)


# ============================================================
# 5. GENERATE FINGERPRINTS
# ============================================================

valid_smiles = []
fingerprints = []
targets = []

for _, row in df.iterrows():

    smiles = row[SMILES_COL]

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        continue

    fp = morgan.GetFingerprint(mol)

    valid_smiles.append(smiles)
    fingerprints.append(fp)
    targets.append(row[TARGET_COL])


print("Valid molecules:", len(valid_smiles))


# ============================================================
# 6. CONVERT TARGET TO ARRAY
# ============================================================

y = np.array(targets)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

indices = np.arange(len(valid_smiles))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("\nTraining molecules:", len(train_idx))
print("Test molecules:", len(test_idx))


# ============================================================
# 8. PREDICT TEST SET
# ============================================================

# IMPORTANT:
# Your saved model was trained on the molecular feature matrix.
# We therefore regenerate the same 517 features.

from rdkit.Chem import Descriptors


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

    fp = list(morgan.GetFingerprint(mol))

    return descriptors + fp


features = []

for smiles in valid_smiles:

    feat = featurize(smiles)

    if feat is not None:
        features.append(feat)

X = np.array(features)


X_test = X[test_idx]
y_test = y[test_idx]

test_smiles = [valid_smiles[i] for i in test_idx]

predictions = model.predict(X_test)


# ============================================================
# 9. MODEL PERFORMANCE
# ============================================================

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

mae = mean_absolute_error(
    y_test,
    predictions
)

print("\nModel performance on test set:")
print("RMSE:", round(rmse, 4))
print("MAE :", round(mae, 4))


# ============================================================
# 10. CALCULATE MAX Tanimoto SIMILARITY
# ============================================================

training_fps = [
    fingerprints[i]
    for i in train_idx
]

similarities = []

print("\nCalculating applicability-domain similarity...")

for count, test_i in enumerate(test_idx, start=1):

    test_fp = fingerprints[test_i]

    sims = DataStructs.BulkTanimotoSimilarity(
        test_fp,
        training_fps
    )

    max_similarity = max(sims)

    similarities.append(max_similarity)

    if count % 50 == 0:
        print(
            f"Processed {count}/{len(test_idx)}"
        )


similarities = np.array(similarities)


# ============================================================
# 11. CALCULATE ABSOLUTE ERROR
# ============================================================

absolute_errors = np.abs(
    y_test - predictions
)


# ============================================================
# 12. TEST DIFFERENT AD THRESHOLDS
# ============================================================

thresholds = np.arange(
    0.20,
    0.91,
    0.05
)

results = []

print("\n")
print("=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

for threshold in thresholds:

    inside = similarities >= threshold

    n_inside = np.sum(inside)

    coverage = (
        n_inside / len(test_idx)
    ) * 100

    if n_inside < 5:
        continue

    y_inside = y_test[inside]

    pred_inside = predictions[inside]

    threshold_rmse = np.sqrt(
        mean_squared_error(
            y_inside,
            pred_inside
        )
    )

    threshold_mae = mean_absolute_error(
        y_inside,
        pred_inside
    )

    mean_error = np.mean(
        absolute_errors[inside]
    )

    results.append({

        "Threshold": threshold,

        "Molecules_Inside_AD": n_inside,

        "Coverage_Percent": coverage,

        "RMSE": threshold_rmse,

        "MAE": threshold_mae,

        "Mean_Absolute_Error": mean_error

    })

    print(
        f"Threshold = {threshold:.2f} | "
        f"Inside = {n_inside:3d} | "
        f"Coverage = {coverage:5.1f}% | "
        f"RMSE = {threshold_rmse:.4f} | "
        f"MAE = {threshold_mae:.4f}"
    )


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    TABLES_DIR / "ad_threshold_validation.csv",
    index=False
)

print("\nSaved:")
print("ad_threshold_validation.csv")


# ============================================================
# 14. FIND BEST THRESHOLD
# ============================================================

# We don't simply choose the threshold
# with the lowest RMSE because a very high threshold
# could leave only a few molecules inside the domain.

# Require at least 50% coverage.

valid_results = results_df[
    results_df["Coverage_Percent"] >= 50
]

if len(valid_results) > 0:

    best_row = valid_results.loc[
        valid_results["RMSE"].idxmin()
    ]

    print("\n")
    print("=" * 60)
    print("RECOMMENDED AD THRESHOLD")
    print("=" * 60)

    print(
        "Threshold:",
        round(best_row["Threshold"], 2)
    )

    print(
        "Coverage:",
        round(best_row["Coverage_Percent"], 2),
        "%"
    )

    print(
        "RMSE:",
        round(best_row["RMSE"], 4)
    )

    print(
        "MAE:",
        round(best_row["MAE"], 4)
    )


# ============================================================
# 15. PLOT RMSE VS THRESHOLD
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["Threshold"],
    results_df["RMSE"],
    marker="o"
)

plt.xlabel(
    "Minimum Maximum Tanimoto Similarity"
)

plt.ylabel(
    "RMSE"
)

plt.title(
    "Applicability Domain Threshold vs Prediction Error"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "ad_threshold_vs_rmse.png",
    dpi=300
)

plt.show()


# ============================================================
# 16. PLOT COVERAGE VS THRESHOLD
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["Threshold"],
    results_df["Coverage_Percent"],
    marker="o"
)

plt.xlabel(
    "Minimum Maximum Tanimoto Similarity"
)

plt.ylabel(
    "Test Set Coverage (%)"
)

plt.title(
    "Applicability Domain Coverage vs Similarity Threshold"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "ad_threshold_vs_coverage.png",
    dpi=300
)

plt.show()


print("\n")
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print("\nGenerated files:")

print("1. ad_threshold_validation.csv")
print("2. ad_threshold_vs_rmse.png")
print("3. ad_threshold_vs_coverage.png")