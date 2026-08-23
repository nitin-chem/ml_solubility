import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from src.applicability import check_applicability
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
    layout="centered",
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
    placeholder="Example: CCO",
)


# ============================================================
# Prediction
# ============================================================

if st.button("Predict Solubility"):

    if not smiles.strip():

        st.warning("Please enter a SMILES string.")

    else:

        try:

            # ------------------------------------------------
            # Validate SMILES
            # ------------------------------------------------

            mol = Chem.MolFromSmiles(smiles)

            if mol is None:

                st.error(
                    "Invalid SMILES. "
                    "Please enter a valid chemical SMILES."
                )

                st.stop()


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction = predict_solubility(smiles)

            # ------------------------------------------------
            # Applicability Domain
            # ------------------------------------------------

            ad_result = check_applicability(smiles)

            st.success("Prediction completed!")


            # =================================================
            # Molecular Structure + Prediction
            # =================================================

            col_structure, col_prediction = st.columns(2)

            with col_structure:

                st.subheader("Molecular Structure")

                image = Draw.MolToImage(
                    mol,
                    size=(400, 400),
                )

                st.image(image)


            with col_prediction:

                st.subheader("Prediction")

                st.metric(
                    "Predicted LogS",
                    f"{prediction:.4f}",
                )


            # =================================================
            # Molecular Properties
            # =================================================

            properties = molecular_properties(smiles)

            st.subheader("Molecular Properties")

            col1, col2, col3, col4, col5 = st.columns(5)


            with col1:

                st.metric(
                    "MW",
                    f"{properties['Molecular Weight']:.2f}",
                )


            with col2:

                st.metric(
                    "LogP",
                    f"{properties['LogP']:.2f}",
                )


            with col3:

                st.metric(
                    "HBD",
                    int(properties["H-Bond Donors"]),
                )


            with col4:

                st.metric(
                    "HBA",
                    int(properties["H-Bond Acceptors"]),
                )


            with col5:

                st.metric(
                    "TPSA",
                    f"{properties['TPSA']:.2f}",
                )


            # =================================================
            # Applicability Domain
            # =================================================

            st.subheader("Applicability Domain")

            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Maximum Tanimoto Similarity",
                    f"{ad_result['max_similarity']:.4f}",
                )


            with col2:

                st.metric(
                    "Mean Top-5 Similarity",
                    f"{ad_result['mean_top5']:.4f}",
                )


            # ------------------------------------------------
            # AD Status
            # ------------------------------------------------

            status = ad_result["status"]


            if status == "HIGH CONFIDENCE":

                st.success(
                    "🟢 HIGH CONFIDENCE\n\n"
                    "The molecule is highly similar to compounds "
                    "in the model's training chemical space."
                )


            elif status == "MODERATE CONFIDENCE / INSIDE DOMAIN":

                st.warning(
                    "🟡 MODERATE CONFIDENCE / INSIDE DOMAIN\n\n"
                    "The molecule is within the validated "
                    "applicability domain, but the prediction "
                    "should be interpreted with some caution."
                )


            else:

                st.error(
                    "🔴 LOW CONFIDENCE / OUTSIDE DOMAIN\n\n"
                    "The molecule is outside the validated "
                    "applicability domain. The prediction "
                    "may be unreliable."
                )

            # ============================================================
            # Most Similar Training Molecules
            # ============================================================

            st.subheader("Most Similar Training Molecules")

            with st.expander("Show Top 5 Similar Molecules"):

                for i, (similarity, similar_smiles) in enumerate(
                    ad_result["top_5"],
                    start=1,
                ):

                    st.write(
                        f"**{i}.** `{similar_smiles}` "
                        f"— Tanimoto similarity: **{similarity:.4f}**"
                    )
        # =====================================================
        # Error handling
        # =====================================================

        except ValueError as error:

            st.error(str(error))


        except Exception as error:

            st.error(f"Prediction failed: {error}")