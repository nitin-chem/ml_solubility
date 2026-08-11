import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------
# 1. Load dataset
# ---------------------------------------

df = pd.read_csv("delaney.csv")

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()

print(df.head())
print(df.columns)

# ---------------------------------------
# 2. Select columns
# ---------------------------------------



smiles_col = "SMILES"

target_col = next(col for col in df.columns if "measured log" in col)

print("SMILES column:", smiles_col)
print("Target column:", target_col)


# ---------------------------------------
# 3. Convert SMILES into molecular features
# ---------------------------------------

from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

morgan = GetMorganGenerator(radius=2, fpSize=512)

def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Basic descriptors
    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
    ]

    # Fingerprints (reduced size)
    fp = list(morgan.GetFingerprint(mol))

    return desc + fp   # combine both
# ---------------------------------------
# 4. Generate features and targets
# ---------------------------------------

features = []
targets = []

for _, row in df.iterrows():

    feats = featurize(row[smiles_col])

    if feats is not None:
        features.append(feats)
        targets.append(row[target_col])


X = np.array(features)
y = np.array(targets)


print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# ---------------------------------------
# 5. Train-test split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------
# 6. Linear Regression
# ---------------------------------------

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)


# ---------------------------------------
# 7. Random Forest
# ---------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# ---------------------------------------
# 8. Evaluation
# ---------------------------------------

def evaluate(y_true, y_pred, name):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name} Results:")
    print("RMSE:", rmse)
    print("R2:", r2)
    print("-" * 30)


evaluate(y_test, lr_pred, "Linear Regression")

evaluate(y_test, rf_pred, "Random Forest")
evaluate(y_test, gb_pred, "Gradient Boosting")

# ---------------------------------------
# 9. Plot actual vs predicted
# ---------------------------------------

plt.figure(figsize=(6,6))
plt.scatter(y_test, rf_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Random Forest Predictions")
plt.grid()
plt.show()

import joblib

joblib.dump(rf_model, "model.pkl")

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": rf_pred
})

results.to_csv("predictions.csv", index=False)