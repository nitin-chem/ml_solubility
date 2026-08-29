# Molecular Solubility Prediction using Machine Learning

A machine learning project for predicting aqueous molecular solubility (logS) from molecular structure represented as SMILES.

This repository trains and evaluates a regression model based on RDKit molecular descriptors and Morgan fingerprints, then exposes the trained model through a Streamlit application for interactive prediction.

---

## 1. Project title and short description

This project predicts the aqueous solubility of a molecule from its structure using a supervised machine-learning approach.

The workflow is:

- convert a molecule to SMILES
- generate molecular descriptors and Morgan fingerprints
- build a feature matrix
- train a regression model
- save the trained model
- load the model in a Streamlit app for prediction
- estimate applicability-domain confidence using Tanimoto similarity

---

## 2. Overview

Aqueous solubility is an important molecular property in chemistry, pharmaceutical research, environmental chemistry, and drug discovery.

In this project, molecular structures are represented as SMILES and converted to numerical features for machine-learning prediction of:

> logS = logarithm of aqueous solubility in mol/L

The current implementation includes:

- Delaney ESOL dataset processing
- RDKit molecular descriptor generation
- Morgan fingerprint generation
- model training and evaluation
- trained model serialization using joblib
- Streamlit-based prediction interface
- applicability-domain analysis based on molecular similarity

---

## 3. Key features

- SMILES-based molecular feature generation
- RDKit molecular descriptors
- Morgan molecular fingerprints
- Gradient Boosting Regressor model
- 5-fold validation workflow in the model-development scripts
- model comparison and performance analysis
- application of applicability-domain logic
- interactive prediction via Streamlit
- molecule drawing, name lookup, and direct SMILES input
- model artifact saved as `model.pkl`

---

## 4. Repository structure

```text
ml_solubility/
├── app.py
├── README.md
├── requirements.txt
├── LICENSE
├── delaney.xlsx
├── project_files.txt
├── data/
│   └── delaney.xlsx
├── docs/
│   └── figures/
├── models/
│   └── model.pkl
├── results/
│   ├── figures/
│   └── tables/
├── src/
│   ├── actual_vs_predicted.py
│   ├── ad_error_threshold.py
│   ├── ad_threshold.py
│   ├── ad_threshold_validation.py
│   ├── applicability.py
│   ├── error.py
│   ├── esol_comparison.py
│   ├── evaluate.py
│   ├── feature_importance.py
│   ├── final_results.py
│   ├── model.py
│   ├── model_comparison.py
│   ├── predict.py
│   ├── run_pipeline.py
│   └── shap_analysis.py
└── .venv/
```

Key files:

- `app.py`: Streamlit application for interactive prediction
- `src/predict.py`: loads the trained model and generates features for prediction
- `src/applicability.py`: computes Tanimoto similarity and applicability-domain confidence
- `models/model.pkl`: trained GradientBoostingRegressor model artifact
- `requirements.txt`: runtime dependencies for the current app
- `results/`: plots and CSV results from model analysis

---

## 5. Dataset

The project uses the Delaney ESOL solubility dataset.

The dataset contains experimentally measured aqueous solubility values and molecular structures represented as SMILES.

The dataset contains:

- 1,144 molecules
- target variable: measured aqueous solubility (logS)
- SMILES strings for molecular structure representation

The relevant dataset source is the Delaney ESOL workbook used by the project pipeline.

---

## 6. Machine-learning model

The final model in the current repository is a `GradientBoostingRegressor` trained on molecular descriptors and Morgan fingerprints.

The trained model artifact is:

- `models/model.pkl`

This model is loaded by the prediction and applicability-domain code.

---

## 7. Feature engineering

Molecular structures are represented using two types of features:

### 1. Molecular descriptors

The current feature generation uses five molecular descriptors:

- Molecular Weight
- LogP
- H-Bond Donors
- H-Bond Acceptors
- TPSA

### 2. Morgan fingerprint

Morgan fingerprints are generated with:

```python
GetMorganGenerator(
    radius=2,
    fpSize=512
)
```

This yields:

- 5 molecular descriptor values
- 512-bit Morgan fingerprint features

Total feature vector length:

- 517 = 5 descriptors + 512 fingerprint bits

This is the same feature set used in the runtime prediction logic in `src/predict.py`.

---

## 8. Model performance

The current project reports the following test performance for the final model:

- Test RMSE = 0.5964
- Test MAE = 0.4553

The app sidebar also reports these values.

The final model is trained and evaluated with a standard train/test split and is saved for later use in the Streamlit app.

---

## 9. Applicability domain

The application includes an applicability-domain check using Tanimoto similarity between the query molecule and molecules in the training set.

The implemented logic in `src/applicability.py` calculates:

- maximum Tanimoto similarity
- mean top-5 similarity
- top 5 most similar training molecules
- an applicability-domain status

The validated threshold used by the app is:

- AD threshold = 0.55

Validation coverage reported by the project is:

- 58.08%

Inside-domain performance reported by the project is:

- inside-domain RMSE = 0.4588
- inside-domain MAE = 0.3596

The status is classified as:

- HIGH CONFIDENCE
- MODERATE CONFIDENCE / INSIDE DOMAIN
- LOW CONFIDENCE / OUTSIDE DOMAIN

These labels are displayed in the Streamlit app after prediction.

---

## 10. Input methods

The current Streamlit application supports three ways to provide a molecule:

### 1. Draw Molecule

The user draws a molecule in the chemical editor. The generated SMILES is used as the prediction input.

### 2. Molecule Name

The user enters a molecule name. The application uses the PubChem REST API to look up the compound and retrieve its SMILES.

This workflow requires network access to PubChem.

### 3. Enter SMILES

The user enters a SMILES string directly in the app.

The app then validates the SMILES, runs prediction, and computes applicability-domain metrics.

---

## 11. Installation

The project runtime dependencies are listed in `requirements.txt`.

Quick start for Windows PowerShell:

```powershell
cd C:\Users\pc\Desktop\ml_solubility
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The current runtime dependency list includes:

- streamlit
- streamlit-ketcher
- requests
- joblib
- numpy
- pandas
- rdkit
- scikit-learn

---

## 12. Running the Streamlit application

To launch the application from the project root:

```powershell
cd C:\Users\pc\Desktop\ml_solubility
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

The application entry point is:

```bash
streamlit run app.py
```

---

## 13. Prediction workflow

The current prediction workflow is:

1. User provides a molecule through one of the three input methods.
2. The app resolves that input to a canonical SMILES string.
3. The SMILES is validated with RDKit.
4. The app calls the prediction function in `src/predict.py`.
5. The model predicts solubility using the 517-feature representation.
6. Applicability-domain analysis is computed using molecular similarity.
7. The app displays:
   - predicted logS
   - molecular properties
   - applicability-domain statistics
   - confidence state
   - top 5 most similar molecules from the training set

The prediction functions used by the app are:

- `predict_solubility(current_smiles)`
- `check_applicability(current_smiles)`
- `molecular_properties(current_smiles)`

---

## 14. Limitations

This project is a focused machine-learning workflow for the Delaney ESOL dataset and is not a universal solubility model for all possible chemical compounds.

Important limitations include:

- the model is trained on a specific chemical domain
- predictions outside the validated applicability domain may be unreliable
- the applicability-domain analysis is a similarity-based estimate, not a guarantee of correctness
- the Molecule Name workflow depends on network access to the PubChem API
- the model should be interpreted with caution for compounds that are chemically far from the training set

Users should treat predictions outside the validated domain with caution.

---

## 15. References / notes

This project is based on:

- Delaney ESOL aqueous solubility dataset
- RDKit molecular descriptor generation
- Morgan fingerprint generation
- Gradient Boosting regression
- applicability-domain analysis using Tanimoto similarity

The trained model artifact and output results in the repository reflect the current implementation and should be used as the source of truth for the project state.

---

## Quick Start

```powershell
cd C:\Users\pc\Desktop\ml_solubility
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

This project is designed for local use in a Python environment with the installed runtime dependencies.

Combines the features
Loads model.pkl
Predicts logS
🔬 Model Interpretability

Machine learning predictions should not only provide a numerical result. It is also important to understand which molecular features influence the prediction.

Two approaches are implemented.

Feature Importance

The trained Gradient Boosting model is analyzed using feature importance.

The most important features include:

1. LogP
2. Molecular Weight
3. TPSA
4. Morgan fingerprint features
5. H-Bond Acceptors
6. H-Bond Donors

The feature importance results are saved as:

feature_importance.csv
feature_importance_descriptors.png
🧠 SHAP Analysis

SHAP (SHapley Additive exPlanations) is used to provide a more detailed interpretation of the model.

The analysis calculates the contribution of each feature to the model predictions.

Major features identified by SHAP
LogP
Molecular Weight
TPSA
Morgan_295
Morgan_356
H-Bond Acceptors
Morgan_66
Morgan_90
...

The analysis generates:

shap_feature_importance.csv
shap_summary.png
shap_bar.png
shap_waterfall_molecule.png

The waterfall plot can be used to understand why the model produced a particular prediction for an individual molecule.

❌ Error Analysis

Model errors were analyzed on the test set.

Final model:

RMSE = 0.5964
MAE = 0.4553
R² = 0.9183

Additional statistics:

Mean residual = 0.0229
Median absolute error = 0.3680
Maximum absolute error = 2.3088
Minimum absolute error = 0.0022

The largest prediction errors were observed for several chemically challenging molecules, including highly aromatic and polycyclic compounds.

Examples include:

Anthraquinone
Coumachlor
Coronene
Phenolphthalein
Uracil
Cycloheximide
Thiouracil

The analysis generates:

error_analysis_results.csv
worst_predictions.csv
error_actual_vs_predicted.png
residual_distribution.png
residuals_vs_predicted.png
🧪 Applicability Domain

A prediction can be accurate only when the new molecule is sufficiently similar to molecules represented in the training dataset.

Therefore, a molecular applicability-domain analysis was implemented using:

Morgan fingerprints + Tanimoto similarity

For a new molecule, the program calculates:

Maximum Tanimoto similarity
Mean similarity of the top 5 training molecules
Most similar training molecules
Applicability-domain class
📐 Applicability-Domain Classification

The model uses similarity to determine prediction confidence.

The analysis identified three practical categories:

Class Interpretation
HIGH High structural similarity to training data
MODERATE Moderate similarity
LOW Low similarity; prediction should be treated cautiously

For example:

SMILES: CCO

Maximum similarity: 0.5556
Mean top-5 similarity: 0.4696

Prediction status:
LOW CONFIDENCE / OUTSIDE DOMAIN

This demonstrates an important point:

A machine learning model can produce a numerical prediction even when the input molecule is outside the chemical space represented by the training data.

Therefore, applicability-domain analysis should accompany predictions.

📊 Applicability-Domain Threshold Validation

Different similarity thresholds were evaluated against prediction error.

Example results:

Threshold Coverage RMSE
0.20 99.1% 0.5974
0.30 93.4% 0.5773
0.40 83.8% 0.5353
0.50 72.1% 0.5121
0.55 58.1% 0.4588
0.60 46.7% 0.4385
0.65 33.6% 0.4152
0.70 24.5% 0.4360
0.80 15.7% 0.3785

A threshold of approximately:

0.55

was selected as a practical operating point.

At this threshold:

Coverage = 58.08%
RMSE = 0.4588
MAE = 0.3596

This illustrates the trade-off between:

Prediction Coverage
vs.
Prediction Reliability

Increasing the similarity threshold generally improves accuracy for predictions retained inside the domain, but reduces the fraction of molecules that qualify.

📁 Project Structure

The repository is organized as follows:

ml_solubility/
│
├── data/
│ └── delaney.xlsx
│
├── src/
│ ├── model.py
│ ├── predict.py
│ ├── feature_importance.py
│ ├── shap_analysis.py
│ ├── error.py
│ ├── applicability.py
│ ├── ad_threshold.py
│ ├── ad_error_threshold.py
│ └── ad_threshold_validation.py
│
├── models/
│ └── model.pkl
│
├── results/
│ ├── predictions.csv
│ ├── feature_importance.csv
│ ├── shap_feature_importance.csv
│ ├── error_analysis_results.csv
│ ├── worst_predictions.csv
│ ├── ad_similarity_distribution.csv
│ ├── ad_error_results.csv
│ ├── ad_error_summary.csv
│ ├── ad_worst_predictions.csv
│ └── ad_threshold_validation.csv
│
├── plots/
│ ├── actual_vs_predicted.png
│ ├── feature_importance_descriptors.png
│ ├── shap_summary.png
│ ├── shap_bar.png
│ ├── shap_waterfall_molecule.png
│ ├── error_actual_vs_predicted.png
│ ├── residual_distribution.png
│ ├── residuals_vs_predicted.png
│ ├── ad_similarity_distribution.png
│ ├── ad_vs_error.png
│ ├── top5_similarity_vs_error.png
│ ├── ad_threshold_vs_rmse.png
│ └── ad_threshold_vs_coverage.png
│
├── requirements.txt
├── README.md
└── .gitignore

If your actual folder names differ, update this section to match the final repository structure.

💻 Installation

1. Clone the repository
   git clone https://github.com/nitin-chem/ml_solubility.git

Move into the project directory:

cd ml_solubility 2. Create a virtual environment

Windows:

python -m venv .venv

Activate it:

.venv\Scripts\Activate.ps1

If PowerShell blocks activation, you can use:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Then activate again:

.venv\Scripts\Activate.ps1 3. Install dependencies

Install the required packages:

pip install -r requirements.txt

The major dependencies are:

numpy
pandas
scikit-learn
matplotlib
seaborn
rdkit
shap
joblib
openpyxl
📦 Requirements

A typical requirements.txt file contains:

numpy
pandas
scikit-learn
matplotlib
seaborn
rdkit
shap
joblib
openpyxl

Install them with:

pip install -r requirements.txt
▶️ Usage
Train the model

From the project root:

python src/model.py

This will:

Load the dataset
Generate molecular features
Split the data
Train baseline models
Perform cross-validation
Perform hyperparameter tuning
Select the optimized Gradient Boosting model
Evaluate the final model
Save the trained model

The trained model is saved as:

models/model.pkl
🔮 Predict Solubility

Run:

python src/predict.py

Enter a molecular SMILES:

Enter SMILES: CCO

The program returns the predicted logS.

📊 Feature Importance

Run:

python src/feature_importance.py

Outputs:

feature_importance.csv
feature_importance_descriptors.png
🧠 SHAP Interpretability

Run:

python src/shap_analysis.py

Outputs:

shap_feature_importance.csv
shap_summary.png
shap_bar.png
shap_waterfall_molecule.png
❌ Error Analysis

Run:

python src/error.py

This analyzes prediction errors and residuals.

Outputs include:

error_analysis_results.csv
worst_predictions.csv
error_actual_vs_predicted.png
residual_distribution.png
residuals_vs_predicted.png
🧪 Applicability Domain

Run:

python src/applicability.py

Enter a new SMILES to calculate its structural similarity to the training dataset.

📐 AD Threshold Analysis

Run:

python src/ad_threshold.py

This analyzes the distribution of molecular similarity within the dataset.

🔍 AD vs Model Error

Run:

python src/ad_error_threshold.py

This investigates whether prediction error increases as molecular similarity to the training dataset decreases.

✅ AD Threshold Validation

Run:

python src/ad_threshold_validation.py

This evaluates different similarity thresholds and calculates:

Coverage
RMSE
MAE

The results help identify a practical applicability-domain threshold.

📈 Generated Results

The project generates several types of results.

Model performance
actual_vs_predicted.png
predictions.csv
Feature interpretation
feature_importance.csv
feature_importance_descriptors.png
SHAP interpretation
shap_feature_importance.csv
shap_summary.png
shap_bar.png
shap_waterfall_molecule.png
Error analysis
error_analysis_results.csv
worst_predictions.csv
error_actual_vs_predicted.png
residual_distribution.png
residuals_vs_predicted.png
Applicability domain
ad_similarity_distribution.csv
ad_similarity_distribution.png
ad_error_results.csv
ad_error_summary.csv
ad_worst_predictions.csv
ad_vs_error.png
top5_similarity_vs_error.png
ad_threshold_validation.csv
ad_threshold_vs_rmse.png
ad_threshold_vs_coverage.png
🧪 Example

Input:

SMILES: CCO

Output:

Predicted logS: 1.1421

The prediction can then be evaluated together with its applicability-domain similarity.

For example:

Maximum Tanimoto similarity: 0.5556
Mean top-5 similarity: 0.4696

Prediction status:
LOW CONFIDENCE / OUTSIDE DOMAIN

This prevents the prediction from being interpreted solely from the numerical logS value.

⚠️ Limitations

This model has several limitations.

1. Dataset size

The model is trained on 1144 molecules. This is relatively small for representing the entire chemical space.

2. Chemical diversity

Some chemical classes may be poorly represented in the training dataset.

3. Applicability domain

Predictions for molecules structurally different from the training data may be unreliable.

4. Experimental variability

Measured solubility values can depend on:

Temperature
pH
Experimental protocol
Solid-state form
Measurement conditions

These factors are not explicitly modeled here.

5. Molecular representation

The model uses fixed molecular descriptors and Morgan fingerprints rather than directly learning molecular structures using graph neural networks.

🔬 Future Improvements

Possible future extensions include:

XGBoost
LightGBM
CatBoost
Support Vector Regression
Extra Trees Regression
XGBoost/Random Forest ensemble models
Stacking and voting ensembles
Hyperparameter optimization using Optuna
Molecular graph neural networks
More extensive molecular descriptors
Better applicability-domain methods
External validation using an independent dataset
Uncertainty estimation
Experimental-condition-aware solubility prediction
Web-based prediction interface
Streamlit application
REST API for model prediction
🧰 Technologies Used
Technology Purpose
Python Programming language
RDKit Molecular processing and fingerprints
NumPy Numerical computation
Pandas Data processing
Scikit-learn Machine learning
Matplotlib Visualization
Seaborn Statistical visualization
SHAP Model interpretability
Joblib Model serialization
Git/GitHub Version control
📚 Scientific Workflow

The project follows the following computational workflow:

Experimental Solubility Dataset
↓
SMILES Input
↓
RDKit Parsing
↓
┌─────────────────────────┐
│ Molecular Descriptors │
│ + │
│ Morgan Fingerprints │
└─────────────────────────┘
↓
Feature Matrix
↓
Train/Test Split
↓
┌─────────────────────────┐
│ Linear Regression │
│ Random Forest │
│ Gradient Boosting │
└─────────────────────────┘
↓
Cross-Validation
↓
Hyperparameter Optimization
↓
Optimized Gradient Boosting
↓
┌─────────────────────────┐
│ Prediction │
│ Feature Importance │
│ SHAP │
│ Error Analysis │
│ Applicability Domain │
└─────────────────────────┘
📌 Key Result

The optimized Gradient Boosting model achieved:

Test RMSE = 0.5964
Test R² = 0.9183

This demonstrates that molecular descriptors combined with Morgan fingerprints can provide strong predictive performance for the Delaney aqueous solubility dataset.

However, predictions should be interpreted together with the model's applicability domain, particularly for molecules with low structural similarity to the training data.

👨‍🔬 Author

Nitin Sharma

M.Sc. Chemistry
Indian Institute of Science Education and Research (IISER) Pune

📜 License

This project is intended for academic and research purposes.

If you reuse or modify this work, please provide appropriate attribution.

⭐ Acknowledgements
RDKit for cheminformatics functionality
Scikit-learn for machine learning algorithms
SHAP for model interpretability
The authors of the Delaney ESOL dataset
Open-source Python scientific computing community

```

```
