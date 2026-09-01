import joblib  # noqa: I001
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import TanimotoSimilarity
from pathlib import Path
from sklearn.model_selection import train_test_split
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
# 1. LOAD MODEL
# ============================================================

print("=" * 60)
print("APPLICABILITY DOMAIN ANALYSIS")
print("=" * 60)

model = joblib.load(MODELS_DIR / "model.pkl")

print("\nModel loaded successfully.")
print("Model:", type(model).__name__)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_excel(DATA_DIR / "delaney.xlsx")

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

valid_smiles = []

for smiles in df[smiles_col]:

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        valid_smiles.append(smiles)


# Use the SAME split as AD threshold validation
train_smiles, test_smiles = train_test_split(
    valid_smiles,
    test_size=0.20,
    random_state=42
)


training_data = []

for smiles in train_smiles:

    mol = Chem.MolFromSmiles(smiles)

    fingerprint = morgan.GetFingerprint(mol)

    training_data.append({
        "smiles": smiles,
        "fingerprint": fingerprint
    })


print("Training reference molecules:", len(training_data))
def check_applicability(smiles):
    """
    Calculate applicability-domain information
    for a new molecule.
    """

    query_mol = Chem.MolFromSmiles(smiles)

    if query_mol is None:
        raise ValueError(
            "Invalid SMILES. Please enter a valid chemical SMILES."
        )

    query_fp = morgan.GetFingerprint(query_mol)

    similarities = []

    for item in training_data:

        training_smiles = item["smiles"]

        # Do not compare a molecule with itself
        if training_smiles == smiles:
            continue

        similarity = TanimotoSimilarity(
            query_fp,
            item["fingerprint"]
        )

        similarities.append(
            (similarity, training_smiles)
        )

    similarities.sort(
        key=lambda x: x[0],
        reverse=True
    )

    top_5 = similarities[:5]

    top_5_scores = [
        score for score, _ in top_5
    ]

    if not similarities:
        return {
            "max_similarity": 0.0,
            "mean_top5": 0.0,
            "status": "LOW CONFIDENCE / OUTSIDE DOMAIN",
            "most_similar_smiles": "",
            "top_5": [],
        }

    max_similarity = similarities[0][0]

    mean_top5 = np.mean(top_5_scores)

    # Validated threshold
    AD_THRESHOLD = 0.55

    # Confidence classification
    if max_similarity >= 0.80:

        status = "HIGH CONFIDENCE"

    elif max_similarity >= AD_THRESHOLD:

        status = "MODERATE CONFIDENCE / INSIDE DOMAIN"

    else:

        status = "LOW CONFIDENCE / OUTSIDE DOMAIN"

    return {
        "max_similarity": float(max_similarity),
        "mean_top5": float(mean_top5),
        "status": status,
        "most_similar_smiles": top_5[0][1],
        "top_5": top_5,
    }

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
if __name__ == "__main__":
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

        AD_THRESHOLD = 0.55

        if max_similarity >= 0.80:

            status = "HIGH CONFIDENCE"

        elif max_similarity >= AD_THRESHOLD:

            status = "MODERATE CONFIDENCE / INSIDE DOMAIN"

        else:

            status = "LOW CONFIDENCE / OUTSIDE DOMAIN"


        print("\n" + "=" * 60)
        print("APPLICABILITY DOMAIN ASSESSMENT")
        print("=" * 60)

        print("\nPrediction status:")
        print(status)

        print("\nMaximum Tanimoto similarity:")
        print(f"{max_similarity:.4f}")

        print("\nMean top-5 similarity:")
        print(f"{mean_top5:.4f}")