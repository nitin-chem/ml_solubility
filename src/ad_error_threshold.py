from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# ============================================================
# APPLICABILITY DOMAIN vs MODEL ERROR ANALYSIS
# ============================================================

print("=" * 60)
print("APPLICABILITY DOMAIN vs MODEL ERROR")
print("=" * 60)


# ============================================================
# 1. LOAD MODEL
# ============================================================

model = joblib.load(MODELS_DIR / "model.pkl")

print("\nModel loaded successfully.")
print("Model:", type(model).__name__)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_excel(DATA_DIR / "delaney.xlsx")

df.columns = df.columns.str.strip()

smiles_col = "SMILES"
target_col = "measured log"

print("\nDataset loaded.")
print("Number of molecules:", len(df))


# ============================================================
# 3. MORGAN FINGERPRINT GENERATOR
# ============================================================

morgan = GetMorganGenerator(
    radius=2,
    fpSize=512
)


# ============================================================
# 4. CREATE MOLECULAR FEATURES
# ============================================================

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

    fingerprint = list(
        morgan.GetFingerprint(mol)
    )

    return descriptors + fingerprint


# ============================================================
# 5. GENERATE DATASET FEATURES
# ============================================================

features = []
targets = []
valid_smiles = []

for _, row in df.iterrows():

    smiles = row[smiles_col]

    feats = featurize(smiles)

    if feats is not None:

        features.append(feats)
        targets.append(row[target_col])
        valid_smiles.append(smiles)


X = np.array(features)
y = np.array(targets)

print("\nFeature matrix:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
    X,
    y,
    valid_smiles,
    test_size=0.20,
    random_state=42
)

print("\nTraining molecules:", len(X_train))
print("Test molecules:", len(X_test))


# ============================================================
# 7. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 8. CALCULATE MODEL ERRORS
# ============================================================

absolute_errors = np.abs(
    y_test - y_pred
)

residuals = (
    y_test - y_pred
)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nRMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")


# ============================================================
# 9. CREATE TRAINING FINGERPRINTS
# ============================================================

train_fingerprints = []

for smiles in smiles_train:

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:

        fp = morgan.GetFingerprint(mol)

        train_fingerprints.append(fp)


# ============================================================
# 10. CALCULATE AD SIMILARITY FOR TEST MOLECULES
# ============================================================

max_similarities = []
mean_top5_similarities = []

print("\nCalculating applicability-domain similarities...")

for i, smiles in enumerate(smiles_test):

    mol = Chem.MolFromSmiles(smiles)

    query_fp = morgan.GetFingerprint(mol)

    similarities = BulkTanimotoSimilarity(
        query_fp,
        train_fingerprints
    )

    similarities = np.array(similarities)

    similarities = np.sort(
        similarities
    )[::-1]

    # Maximum similarity
    max_similarity = similarities[0]

    # Mean of top 5 similarities
    top5 = similarities[:5]

    mean_top5 = np.mean(top5)

    max_similarities.append(
        max_similarity
    )

    mean_top5_similarities.append(
        mean_top5
    )

    if (i + 1) % 50 == 0:

        print(
            f"Processed {i + 1}/{len(smiles_test)}"
        )


# ============================================================
# 11. CREATE RESULTS TABLE
# ============================================================

results = pd.DataFrame({

    "SMILES": smiles_test,

    "Actual_logS": y_test,

    "Predicted_logS": y_pred,

    "Residual": residuals,

    "Absolute_Error": absolute_errors,

    "Max_Tanimoto": max_similarities,

    "Mean_Top5_Tanimoto": mean_top5_similarities

})


# ============================================================
# 12. AD CLASSIFICATION
# ============================================================

def classify_ad(max_similarity, mean_top5):

    if (
        max_similarity >= 0.70
        and mean_top5 >= 0.50
    ):

        return "HIGH"

    elif (
        max_similarity >= 0.50
        and mean_top5 >= 0.40
    ):

        return "MODERATE"

    else:

        return "LOW"


results["AD_Class"] = [

    classify_ad(
        max_similarity,
        mean_top5
    )

    for max_similarity, mean_top5
    in zip(
        results["Max_Tanimoto"],
        results["Mean_Top5_Tanimoto"]
    )
]


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results.to_csv(
    TABLES_DIR / "ad_error_results.csv",
    index=False
)

print("\nSaved:")
print("ad_error_results.csv")


# ============================================================
# 14. ERROR BY AD CLASS
# ============================================================

print("\n" + "=" * 60)
print("ERROR BY APPLICABILITY DOMAIN")
print("=" * 60)

summary = (
    results
    .groupby("AD_Class")
    .agg(
        Molecules=("Absolute_Error", "count"),
        Mean_Error=("Absolute_Error", "mean"),
        Median_Error=("Absolute_Error", "median"),
        RMSE=("Absolute_Error", lambda x: np.sqrt(np.mean(x**2))),
        Mean_Similarity=("Max_Tanimoto", "mean")
    )
    .sort_values("Mean_Error")
)

print("\n")
print(summary)


# ============================================================
# 15. SAVE SUMMARY
# ============================================================

summary.to_csv(
    TABLES_DIR / "ad_error_summary.csv"
)

print("\nSaved:")
print("ad_error_summary.csv")


# ============================================================
# 16. CORRELATION
# ============================================================

correlation = results[
    [
        "Max_Tanimoto",
        "Mean_Top5_Tanimoto",
        "Absolute_Error"
    ]
].corr()

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

print("\n")
print(correlation)


# ============================================================
# 17. PLOT: SIMILARITY vs ERROR
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    results["Max_Tanimoto"],
    results["Absolute_Error"],
    alpha=0.65
)

plt.axvline(
    0.50,
    linestyle="--",
    label="Similarity = 0.50"
)

plt.axvline(
    0.70,
    linestyle="--",
    label="Similarity = 0.70"
)

plt.xlabel(
    "Maximum Tanimoto Similarity"
)

plt.ylabel(
    "Absolute Prediction Error"
)

plt.title(
    "Applicability Domain vs Prediction Error"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "ad_vs_error.png",
    dpi=300
)

plt.show()


# ============================================================
# 18. PLOT: TOP-5 SIMILARITY vs ERROR
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    results["Mean_Top5_Tanimoto"],
    results["Absolute_Error"],
    alpha=0.65
)

plt.xlabel(
    "Mean Top-5 Tanimoto Similarity"
)

plt.ylabel(
    "Absolute Prediction Error"
)

plt.title(
    "Mean Top-5 Similarity vs Prediction Error"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "top5_similarity_vs_error.png",
    dpi=300
)

plt.show()


# ============================================================
# 19. WORST PREDICTIONS
# ============================================================

worst = results.sort_values(
    "Absolute_Error",
    ascending=False
).head(20)

print("\n" + "=" * 60)
print("20 WORST PREDICTIONS")
print("=" * 60)

print(
    worst[
        [
            "SMILES",
            "Actual_logS",
            "Predicted_logS",
            "Absolute_Error",
            "Max_Tanimoto",
            "Mean_Top5_Tanimoto",
            "AD_Class"
        ]
    ].to_string(index=False)
)

worst.to_csv(
    TABLES_DIR / "ad_worst_predictions.csv",
    index=False
)

print("\nSaved:")
print("ad_worst_predictions.csv")


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print("\nGenerated files:")

print("1. ad_error_results.csv")
print("2. ad_error_summary.csv")
print("3. ad_worst_predictions.csv")
print("4. ad_vs_error.png")
print("5. top5_similarity_vs_error.png")

print("\nNext step:")
print(
    "Check whether prediction error increases "
    "as molecular similarity decreases."
)