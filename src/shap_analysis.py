from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

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
# 1. Load trained model
# ============================================================

print("=" * 60)
print("SHAP MODEL INTERPRETABILITY")
print("=" * 60)

model = joblib.load(MODELS_DIR / "model.pkl")

print("\nModel loaded successfully.")
print("Model type:", type(model).__name__)


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
# 4. Convert SMILES → 517 features
# ============================================================

def featurize(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # ---- Molecular descriptors ----

    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol)
    ]

    # ---- Morgan fingerprint ----

    fingerprint = list(
        morgan.GetFingerprint(mol)
    )

    return descriptors + fingerprint


# ============================================================
# 5. Generate feature matrix
# ============================================================

features = []
valid_smiles = []

for smiles in df[smiles_col]:

    feat = featurize(smiles)

    if feat is not None:
        features.append(feat)
        valid_smiles.append(smiles)


X = np.array(features)

print("\nFeature matrix shape:", X.shape)

if X.shape[1] != 517:

    raise ValueError(
        f"Expected 517 features but got {X.shape[1]}"
    )


# ============================================================
# 6. Feature names
# ============================================================

feature_names = [
    "Molecular Weight",
    "LogP",
    "H-Bond Donors",
    "H-Bond Acceptors",
    "TPSA"
]

feature_names += [
    f"Morgan_{i}"
    for i in range(512)
]

print("Total feature names:", len(feature_names))


# ============================================================
# 7. Create SHAP explainer
# ============================================================

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

print("SHAP calculation completed.")


# ============================================================
# 8. SHAP feature importance
# ============================================================

mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0
)

shap_importance = pd.DataFrame({

    "Feature": feature_names,

    "Mean_Absolute_SHAP":
        mean_abs_shap

})

shap_importance = shap_importance.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)


# ============================================================
# 9. Print top 20 features
# ============================================================

print("\n" + "=" * 60)
print("TOP 20 FEATURES BY SHAP IMPORTANCE")
print("=" * 60)

print(
    shap_importance.head(20).to_string(
        index=False
    )
)


# ============================================================
# 10. Save SHAP importance
# ============================================================

shap_importance.to_csv(
    TABLES_DIR / "shap_feature_importance.csv",
    index=False
)

print(
    "\nSaved: shap_feature_importance.csv"
)


# ============================================================
# 11. SHAP Summary Plot
# ============================================================

print("\nGenerating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X,
    feature_names=feature_names,
    max_display=20,
    show=False
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: shap_summary.png")


# ============================================================
# 12. SHAP Bar Plot
# ============================================================

print("\nGenerating SHAP bar plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X,
    feature_names=feature_names,
    plot_type="bar",
    max_display=20,
    show=False
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: shap_bar.png")


# ============================================================
# 13. Explain one molecule
# ============================================================

molecule_index = 0

print("\n" + "=" * 60)
print("INDIVIDUAL MOLECULE EXPLANATION")
print("=" * 60)

print(
    "SMILES:",
    valid_smiles[molecule_index]
)

prediction = model.predict(
    X[molecule_index].reshape(1, -1)
)[0]

print(
    "Predicted logS:",
    prediction
)


# ============================================================
# 14. Waterfall plot
# ============================================================

print("\nGenerating waterfall plot...")

# Convert expected value to a scalar
base_value = np.asarray(
    explainer.expected_value
).reshape(-1)[0]

explanation = shap.Explanation(

    values=shap_values[molecule_index],

    base_values=base_value,

    data=X[molecule_index],

    feature_names=feature_names
)
shap.plots.waterfall(
    explanation,
    max_display=15,
    show=False
)

plt.figure()

shap.plots.waterfall(
    explanation,
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "shap_waterfall_molecule.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: shap_waterfall_molecule.png"
)


# ============================================================
# 15. Final message
# ============================================================

print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETED")
print("=" * 60)

print("""
Generated files:

1. shap_feature_importance.csv
2. shap_summary.png
3. shap_bar.png
4. shap_waterfall_molecule.png
""")