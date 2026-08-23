from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


# ============================================================
# 1. Project paths and trained model
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
model_path = MODELS_DIR / "model.pkl"

model = joblib.load(model_path)


# ============================================================
# 2. Morgan fingerprint generator
# ============================================================

# IMPORTANT:
# These settings MUST remain the same as model.py

morgan = GetMorganGenerator(
    radius=2,
    fpSize=512
)


# ============================================================
# 3. Convert SMILES into 517 features
# ============================================================

def featurize(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # Five molecular descriptors
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol)
    ]

    # 512-bit Morgan fingerprint
    fingerprint = list(
        morgan.GetFingerprint(mol)
    )

    # 5 descriptors + 512 fingerprint bits = 517 features
    features = descriptors + fingerprint

    return np.array(features).reshape(1, -1)
#============================================================
# Add a function to compute molecular properties
#============================================================
def molecular_properties(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES")

    return {
        "Molecular Weight": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "H-Bond Donors": Descriptors.NumHDonors(mol),
        "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
    }
# ============================================================
# 4. Prediction function
# ============================================================

def predict_solubility(smiles):

    smiles = smiles.strip()

    if not smiles:
        raise ValueError("SMILES cannot be empty.")

    features = featurize(smiles)

    if features is None:
        raise ValueError("Invalid SMILES. Please enter a valid chemical SMILES.")

    prediction = model.predict(features)[0]

    return float(prediction)


# ============================================================
# 5. Terminal interface
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("MOLECULAR SOLUBILITY PREDICTION")
    print("=" * 50)

    while True:

        smiles = input("\nEnter SMILES: ").strip()

        try:

            prediction = predict_solubility(smiles)

            features = featurize(smiles)

            print("\nFeatures generated:", features.shape)

            print("\n" + "=" * 50)
            print("PREDICTION")
            print("=" * 50)

            print(f"SMILES: {smiles}")
            print(f"Predicted logS: {prediction:.4f}")

            print("=" * 50)

        except ValueError as error:

            print(f"\nERROR: {error}")

        again = input(
            "\nWould you like to predict another SMILES? (y/n): "
        ).strip().lower()

        if again != "y":

            print("\nExiting program.")
            break