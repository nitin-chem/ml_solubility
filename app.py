import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from src.predict import (
    molecular_properties,
    predict_solubility,
)

# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="ML Solubility Predictor",
    page_icon="🧪",
    layout="centered"
)


# ============================================================
# Title
# ============================================================

st.title("🧪 Molecular Solubility Predictor")

st.write(
    "Predict the aqueous solubility (LogS) of a molecule "
    "using a machine-learning model."
)


# ============================================================
# SMILES input
# ============================================================

smiles = st.text_input(
    "Enter SMILES",
    placeholder="Example: CCO"
)


# ============================================================
# Prediction
# ============================================================

if st.button("Predict Solubility"):

    if not smiles.strip():

        st.warning("Please enter a SMILES string.")

    else:

        try:

            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                st.error("Invalid SMILES. Please enter a valid chemical SMILES.")
                st.stop()

            prediction = predict_solubility(smiles)

            st.success("Prediction completed!")

            st.subheader("Molecular Structure")

            image = Draw.MolToImage(
                mol,
                size=(400, 400)
            )

            st.image(image)
            properties = molecular_properties(smiles)

            st.subheader("Molecular Properties")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Molecular Weight",
                    f"{properties['Molecular Weight']:.2f}"
                )

                st.metric(
                    "H-Bond Donors",
                    int(properties["H-Bond Donors"])
                )

                st.metric(
                    "TPSA",
                    f"{properties['TPSA']:.2f}"
                )

            with col2:
                st.metric(
                    "LogP",
                    f"{properties['LogP']:.2f}"
                )

                st.metric(
                    "H-Bond Acceptors",
                    int(properties["H-Bond Acceptors"])
                )
                st.metric(
                    "Predicted LogS",
                    f"{prediction:.4f}"
                )

        except ValueError as error:

            st.error(str(error))

        except Exception as error:

            st.error(f"Prediction failed: {error}")