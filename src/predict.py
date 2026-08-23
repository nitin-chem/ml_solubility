import sys

# ============================================================
# 1. Load trained model
# ============================================================
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained model
MODELS_DIR = BASE_DIR / "models"
model_path = MODELS_DIR / "model.pkl"
model = joblib.load(model_path)

print("=" * 50)
print("MOLECULAR SOLUBILITY PREDICTION")
print("=" * 50)


# ============================================================
# 2. Create Morgan fingerprint generator
# ============================================================

# IMPORTANT:
# These settings MUST be the same as model.py

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

    # 5 descriptors + 512 fingerprint bits
    features = descriptors + fingerprint

    return np.array(features).reshape(1, -1)


# ============================================================
# 4. Get SMILES from user
# ============================================================

smiles = input("\nEnter SMILES: ").strip()


# ============================================================
# 5. Generate features
# ============================================================

features = featurize(smiles)


if features is None:

    print("\nERROR: Invalid SMILES.")
    print("Please enter a valid chemical SMILES.")

else:

    print("\nFeatures generated:", features.shape)


    # ========================================================
    # 6. Make prediction
    # ========================================================

    prediction = model.predict(features)[0]


    # ========================================================
    # 7. Display result
    # ========================================================

    print("\n" + "=" * 50)
    print("PREDICTION")
    print("=" * 50)

    print(f"SMILES: {smiles}")

    print(
        f"Predicted logS: {prediction:.4f}"
    )

    print("=" * 50)
    
#=========================================================
# Add another Smiles input
#=========================================================
print("\nWould you like to predict another SMILES? (y/n)")
if input().strip().lower() != 'y':
    print("\nExiting program.")
    sys.exit()  
    
smiles2 = input("\nEnter another SMILES: ").strip()
features2 = featurize(smiles2)

if features2 is None:
    print("\nERROR: Invalid SMILES.")
    print("Please enter a valid chemical SMILES.")
else:
    prediction2 = model.predict(features2)[0]
    print(f"SMILES: {smiles2}")
    print(f"Predicted logS: {prediction2:.4f}")

