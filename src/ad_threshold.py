from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity

# ============================================================
# 1. PROJECT PATHS  
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
# APPLICABILITY DOMAIN THRESHOLD ANALYSIS
# ============================================================

print("=" * 60)
print("APPLICABILITY DOMAIN THRESHOLD ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_excel(DATA_DIR / "delaney.xlsx")

df.columns = df.columns.str.strip()

smiles_col = "SMILES"

print("\nDataset size:", len(df))


# ------------------------------------------------------------
# 2. Morgan fingerprint generator
# ------------------------------------------------------------

morgan = GetMorganGenerator(
    radius=2,
    fpSize=512
)


# ------------------------------------------------------------
# 3. Generate fingerprints
# ------------------------------------------------------------

fingerprints = []
valid_smiles = []

for smiles in df[smiles_col]:

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:

        fp = morgan.GetFingerprint(mol)

        fingerprints.append(fp)
        valid_smiles.append(smiles)


print("Valid molecules:", len(fingerprints))


# ------------------------------------------------------------
# 4. Calculate nearest-neighbour similarity
# ------------------------------------------------------------

max_similarities = []

print("\nCalculating similarities...")

for i, query_fp in enumerate(fingerprints):

    similarities = BulkTanimotoSimilarity(
        query_fp,
        fingerprints
    )

    # Remove similarity with itself
    similarities[i] = 0.0

    max_similarity = max(similarities)

    max_similarities.append(max_similarity)

    if (i + 1) % 100 == 0:

        print(
            f"Processed {i + 1}/{len(fingerprints)}"
        )


max_similarities = np.array(max_similarities)


# ------------------------------------------------------------
# 5. Statistics
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SIMILARITY STATISTICS")
print("=" * 60)

print(
    f"\nMinimum similarity: "
    f"{max_similarities.min():.4f}"
)

print(
    f"25th percentile: "
    f"{np.percentile(max_similarities, 25):.4f}"
)

print(
    f"Median similarity: "
    f"{np.median(max_similarities):.4f}"
)

print(
    f"75th percentile: "
    f"{np.percentile(max_similarities, 75):.4f}"
)

print(
    f"90th percentile: "
    f"{np.percentile(max_similarities, 90):.4f}"
)

print(
    f"95th percentile: "
    f"{np.percentile(max_similarities, 95):.4f}"
)

print(
    f"Maximum similarity: "
    f"{max_similarities.max():.4f}"
)


# ------------------------------------------------------------
# 6. Histogram
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    max_similarities,
    bins=25,
    edgecolor="black"
)

plt.xlabel(
    "Maximum Tanimoto Similarity to Other Training Molecules"
)

plt.ylabel("Number of Molecules")

plt.title(
    "Applicability Domain Similarity Distribution"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "ad_similarity_distribution.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 7. Save results
# ------------------------------------------------------------

ad_results = pd.DataFrame({

    "SMILES": valid_smiles,

    "Max_Tanimoto_Similarity":
        max_similarities

})

ad_results.to_csv(
    TABLES_DIR / "ad_similarity_distribution.csv",
    index=False
)


print("\nSaved:")
print("1. ad_similarity_distribution.csv")
print("2. ad_similarity_distribution.png")