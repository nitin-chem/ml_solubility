import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import TanimotoSimilarity

# ============================================================
# 1. LOAD MODEL
# ============================================================

print("=" * 60)
print("APPLICABILITY DOMAIN ANALYSIS")
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

print("Dataset loaded.")
print("Number of molecules:", len(df))


# ============================================================
# 3. MORGAN FINGERPRINT GENERATOR
# ============================================================

morgan = GetMorganGenerator(
    radius=2,
    fpSize=512
)


# ============================================================
# 4. CREATE TRAINING FINGERPRINTS
# ============================================================

training_data = []

for _, row in df.iterrows():

    smiles = row[smiles_col]

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        continue

    fingerprint = morgan.GetFingerprint(mol)

    training_data.append({
        "smiles": smiles,
        "fingerprint": fingerprint
    })


print("Valid training molecules:", len(training_data))


# ============================================================
# 5. FEATURE GENERATION
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

    return np.array(descriptors + fingerprint).reshape(1, -1)


# ============================================================
# 6. ENTER NEW MOLECULE
# ============================================================

query_smiles = input("\nEnter SMILES: ").strip()

query_mol = Chem.MolFromSmiles(query_smiles)

if query_mol is None:

    print("\nERROR: Invalid SMILES.")
    exit()  # noqa: PLR1722


# ============================================================
# 7. PREDICT SOLUBILITY
# ============================================================

X_query = featurize(query_smiles)

prediction = model.predict(X_query)[0]

print("\nSMILES:")
print(query_smiles)

print("\nPredicted logS:")
print(f"{prediction:.4f}")


# ============================================================
# 8. CALCULATE Tanimoto SIMILARITY
# ============================================================

query_fp = morgan.GetFingerprint(query_mol)

similarities = []

for item in training_data:

    training_smiles = item["smiles"]

    # --------------------------------------------------------
    # IMPORTANT:
    # Don't compare molecule with itself
    # --------------------------------------------------------

    if training_smiles == query_smiles:

        continue

    similarity = TanimotoSimilarity(
        query_fp,
        item["fingerprint"]
    )

    similarities.append(
        (similarity, training_smiles)
    )


# Sort from highest to lowest similarity

similarities.sort(
    key=lambda x: x[0],
    reverse=True
)


# ============================================================
# 9. SIMILARITY STATISTICS
# ============================================================

top_5 = similarities[:5]

top_5_scores = [
    x[0] for x in top_5
]

max_similarity = similarities[0][0]

mean_top5 = np.mean(top_5_scores)

most_similar_smiles = similarities[0][1]


print("\n" + "=" * 60)
print("APPLICABILITY DOMAIN")
print("=" * 60)

print("\nMaximum Tanimoto similarity:")
print(f"{max_similarity:.4f}")

print("\nMean top-5 similarity:")
print(f"{mean_top5:.4f}")

print("\nMost similar training molecule:")
print(most_similar_smiles)


# ============================================================
# 10. SHOW TOP 5 SIMILAR MOLECULES
# ============================================================

print("\nTop 5 most similar training molecules:")

for i, (score, smiles) in enumerate(top_5, start=1):

    print(
        f"{i}. Similarity = {score:.4f} | {smiles}"
    )


# ============================================================
# 11. APPLICABILITY DOMAIN CLASSIFICATION
# ============================================================

print("\n" + "=" * 60)
print("APPLICABILITY DOMAIN ASSESSMENT")
print("=" * 60)

AD_THRESHOLD = 0.55

if max_similarity >= AD_THRESHOLD:
    status = "HIGH CONFIDENCE / INSIDE DOMAIN"
else:
    status = "LOW CONFIDENCE / OUTSIDE DOMAIN"
if max_similarity >= 0.80:

    status = "HIGH CONFIDENCE"

elif max_similarity >= 0.60:

    status = "MODERATE CONFIDENCE"

else:

    status = "LOW CONFIDENCE / OUTSIDE DOMAIN"


print("\nPrediction status:")
print(status)

print("\nInterpretation:")

if status == "HIGH CONFIDENCE":

    print(
        "The molecule is structurally similar to compounds "
        "in the training dataset."
    )

elif status == "MODERATE CONFIDENCE":

    print(
        "The molecule has moderate structural similarity "
        "to the training dataset. Prediction should be "
        "interpreted with some caution."
    )

else:

    print(
        "The molecule is structurally dissimilar to the "
        "training dataset. Prediction may be unreliable."
    )

print("\n" + "=" * 60)