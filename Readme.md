# Molecular Solubility Prediction using Machine Learning

A machine learning project for predicting aqueous molecular solubility (`logS`) from molecular structure represented as SMILES.

The project compares multiple regression models and uses RDKit molecular descriptors and Morgan fingerprints as molecular features. The best-performing model is further optimized using hyperparameter tuning and analyzed using cross-validation, SHAP interpretability, error analysis, and applicability-domain analysis.

---

## 📌 Project Overview

Aqueous solubility is an important molecular property in chemistry, pharmaceutical research, environmental chemistry, and drug discovery.

In this project, molecular structures are converted from SMILES representations into numerical molecular features and used to train machine learning regression models for predicting:

> **logS = logarithm of aqueous solubility in mol/L**

The complete workflow includes:

```text
SMILES
   ↓
RDKit Molecular Processing
   ↓
Molecular Descriptors + Morgan Fingerprints
   ↓
Feature Matrix
   ↓
Train / Test Split
   ↓
Machine Learning Models
   ├── Linear Regression
   ├── Random Forest
   └── Gradient Boosting
   ↓
Cross-Validation
   ↓
Hyperparameter Optimization
   ↓
Best Model
   ↓
Prediction
   ↓
Interpretability + Error Analysis
   ↓
Applicability Domain
✨ Features

This project currently includes:

SMILES-based molecular feature generation
RDKit molecular descriptors
Morgan molecular fingerprints
Linear Regression
Random Forest Regression
Gradient Boosting Regression
5-fold cross-validation
Randomized hyperparameter optimization
Model comparison using RMSE and R²
Prediction of solubility for new molecules
Feature importance analysis
SHAP-based model interpretability
Individual molecule explanation using SHAP waterfall plots
Prediction error analysis
Residual analysis
Applicability Domain analysis
Tanimoto molecular similarity analysis
Applicability-domain threshold validation
Prediction confidence classification
Actual vs predicted plots
Model serialization using Joblib
Prediction output saved as CSV
Reproducible Python workflow
📊 Dataset

The project uses the Delaney ESOL solubility dataset.

The dataset contains molecular structures represented by SMILES together with experimentally measured aqueous solubility values.

Main columns
Column	Description
Compound ID	Compound identifier
measured log(solubility:mol/L)	Experimental solubility
ESOL predicted log(solubility:mol/L)	ESOL reference prediction
SMILES	Molecular structure

After preprocessing, the dataset contains:

Number of molecules: 1144
🧬 Molecular Feature Generation

Molecular structures are read from SMILES using RDKit.

Each molecule is represented using two types of features.

1. Molecular Descriptors

The following descriptors are used:

Molecular Weight
Molecular LogP
Number of Hydrogen Bond Donors
Number of Hydrogen Bond Acceptors
Topological Polar Surface Area (TPSA)
2. Morgan Fingerprints

Morgan fingerprints are generated using:

GetMorganGenerator(
    radius=2,
    fpSize=512
)

Therefore, each molecule contains:

5 molecular descriptors
+
512 Morgan fingerprint features
=
517 total features

Final feature matrix:

(1144, 517)
🤖 Machine Learning Models

Three regression algorithms were initially compared.

Linear Regression

A simple baseline model used to establish the relationship between molecular features and solubility.

Random Forest Regression

An ensemble tree-based model that combines predictions from multiple decision trees.

Gradient Boosting Regression

An ensemble method that sequentially builds decision trees to minimize prediction error.

📈 Initial Model Performance

Using the same train/test split:

Test size = 20%
Random state = 42
Model comparison
Model	RMSE	R²
Linear Regression	1.2918	0.6168
Random Forest	0.6242	0.9105
Gradient Boosting	0.6347	0.9075

The tree-based models significantly outperform the linear regression baseline.

🔍 Cross-Validation

To evaluate model stability, 5-fold cross-validation was performed.

Random Forest
Mean R²   = 0.8888
Std R²    = 0.0102

Mean RMSE = 0.6966
Std RMSE  = 0.0497
Gradient Boosting
Mean R²   = 0.8921
Std R²    = 0.0098

Mean RMSE = 0.6867
Std RMSE  = 0.0546

The cross-validation results indicate that Gradient Boosting provides slightly better average performance than Random Forest.

⚙️ Hyperparameter Optimization

RandomizedSearchCV was used to optimize both tree-based models.

Random Forest

The optimized Random Forest produced:

Best CV RMSE:
0.68085

Best parameters:

n_estimators      = 200
max_depth         = 25
min_samples_split = 2
min_samples_leaf  = 2
max_features      = 0.5

Test performance:

RMSE = 0.6174
R²   = 0.9125
🚀 Optimized Gradient Boosting Model

The Gradient Boosting model was optimized using RandomizedSearchCV.

Best parameters:

n_estimators      = 500
learning_rate     = 0.08
max_depth         = 4
min_samples_split = 5
min_samples_leaf  = 6
subsample         = 0.9
Best Cross-Validation Performance
CV RMSE = 0.6686
Final Test Performance
RMSE = 0.5964
R²   = 0.9183

The optimized Gradient Boosting model is therefore selected as the final model.

🏆 Final Model

The final model is:

GradientBoostingRegressor

Performance:

Metric	Value
RMSE	0.5964
R²	0.9183
MAE	0.4553

The trained model is saved as:

model.pkl
🔮 Making Predictions

A separate prediction script is provided:

predict.py

Run:

python predict.py

The program asks for a SMILES string:

Enter SMILES: CCO

Example output:

Features generated: (1, 517)

PREDICTION

SMILES: CCO
Predicted logS: 1.1421

The script automatically:

Reads the SMILES
Converts it into an RDKit molecule
Generates molecular descriptors
Generates Morgan fingerprints
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
MAE  = 0.4553
R²   = 0.9183

Additional statistics:

Mean residual              = 0.0229
Median absolute error      = 0.3680
Maximum absolute error     = 2.3088
Minimum absolute error     = 0.0022

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

Class	Interpretation
HIGH	High structural similarity to training data
MODERATE	Moderate similarity
LOW	Low similarity; prediction should be treated cautiously

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

Threshold	Coverage	RMSE
0.20	99.1%	0.5974
0.30	93.4%	0.5773
0.40	83.8%	0.5353
0.50	72.1%	0.5121
0.55	58.1%	0.4588
0.60	46.7%	0.4385
0.65	33.6%	0.4152
0.70	24.5%	0.4360
0.80	15.7%	0.3785

A threshold of approximately:

0.55

was selected as a practical operating point.

At this threshold:

Coverage = 58.08%
RMSE     = 0.4588
MAE      = 0.3596

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
│   └── delaney.xlsx
│
├── src/
│   ├── model.py
│   ├── predict.py
│   ├── feature_importance.py
│   ├── shap_analysis.py
│   ├── error.py
│   ├── applicability.py
│   ├── ad_threshold.py
│   ├── ad_error_threshold.py
│   └── ad_threshold_validation.py
│
├── models/
│   └── model.pkl
│
├── results/
│   ├── predictions.csv
│   ├── feature_importance.csv
│   ├── shap_feature_importance.csv
│   ├── error_analysis_results.csv
│   ├── worst_predictions.csv
│   ├── ad_similarity_distribution.csv
│   ├── ad_error_results.csv
│   ├── ad_error_summary.csv
│   ├── ad_worst_predictions.csv
│   └── ad_threshold_validation.csv
│
├── plots/
│   ├── actual_vs_predicted.png
│   ├── feature_importance_descriptors.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   ├── shap_waterfall_molecule.png
│   ├── error_actual_vs_predicted.png
│   ├── residual_distribution.png
│   ├── residuals_vs_predicted.png
│   ├── ad_similarity_distribution.png
│   ├── ad_vs_error.png
│   ├── top5_similarity_vs_error.png
│   ├── ad_threshold_vs_rmse.png
│   └── ad_threshold_vs_coverage.png
│
├── requirements.txt
├── README.md
└── .gitignore

If your actual folder names differ, update this section to match the final repository structure.

💻 Installation
1. Clone the repository
git clone https://github.com/nitin-chem/ml_solubility.git

Move into the project directory:

cd ml_solubility
2. Create a virtual environment

Windows:

python -m venv .venv

Activate it:

.venv\Scripts\Activate.ps1

If PowerShell blocks activation, you can use:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Then activate again:

.venv\Scripts\Activate.ps1
3. Install dependencies

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
Technology	Purpose
Python	Programming language
RDKit	Molecular processing and fingerprints
NumPy	Numerical computation
Pandas	Data processing
Scikit-learn	Machine learning
Matplotlib	Visualization
Seaborn	Statistical visualization
SHAP	Model interpretability
Joblib	Model serialization
Git/GitHub	Version control
📚 Scientific Workflow

The project follows the following computational workflow:

Experimental Solubility Dataset
            ↓
       SMILES Input
            ↓
       RDKit Parsing
            ↓
 ┌─────────────────────────┐
 │ Molecular Descriptors   │
 │ +                       │
 │ Morgan Fingerprints     │
 └─────────────────────────┘
            ↓
       Feature Matrix
            ↓
       Train/Test Split
            ↓
 ┌─────────────────────────┐
 │ Linear Regression       │
 │ Random Forest           │
 │ Gradient Boosting       │
 └─────────────────────────┘
            ↓
    Cross-Validation
            ↓
 Hyperparameter Optimization
            ↓
   Optimized Gradient Boosting
            ↓
 ┌─────────────────────────┐
 │ Prediction              │
 │ Feature Importance      │
 │ SHAP                    │
 │ Error Analysis          │
 │ Applicability Domain    │
 └─────────────────────────┘
📌 Key Result

The optimized Gradient Boosting model achieved:

Test RMSE = 0.5964
Test R²   = 0.9183

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
