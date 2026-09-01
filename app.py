import requests
import streamlit as st
from io import BytesIO
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from src.applicability import check_applicability
from src.predict import (
    molecular_properties,
    predict_solubility,
)
from streamlit_ketcher import st_ketcher

# ============================================================
# Convert molecule name to SMILES using PubChem
# ============================================================

def name_to_smiles(name):

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        f"compound/name/{requests.utils.quote(name)}"
        "/property/SMILES/JSON"
    )

    response = requests.get(
        url,
        timeout=10,
    )

    if response.status_code == 404:
        return None

    response.raise_for_status()

    data = response.json()

    return data["PropertyTable"]["Properties"][0]["SMILES"]


# ============================================================
# Molecule rendering (headless Cairo backend for Streamlit Cloud)
# ============================================================

def render_molecule(mol, size=400):
    """Render molecule to PNG bytes using Cairo (no X11 required)."""
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    drawer.DrawMolecule(mol)
    return BytesIO(drawer.GetDrawingText())


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="ML Solubility Predictor",
    page_icon="🧪",
    layout="centered",
)


# ============================================================
# Session State
# ============================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "predicted_smiles" not in st.session_state:
    st.session_state.predicted_smiles = ""

if "name_smiles" not in st.session_state:
    st.session_state.name_smiles = ""

if "name_smiles_source" not in st.session_state:
    st.session_state.name_smiles_source = ""

# ============================================================
# Sidebar - Model Information
# ============================================================

with st.sidebar:

    st.title("🧪 Model Information")

    st.write(
        "Overview of the machine-learning model used "
        "for aqueous solubility prediction."
    )

    st.divider()

    st.subheader("Dataset")

    st.write("**Dataset:** Delaney ESOL")
    st.write("**Molecules:** 1,144")

    st.subheader("Machine Learning Model")

    st.write("**Algorithm:** Gradient Boosting Regressor")
    st.write("**Features:** 517")
    st.write("• 5 molecular descriptors")
    st.write("• 512-bit Morgan fingerprint")

    st.subheader("Model Performance")

    st.metric(
        "Test RMSE",
        "0.5964",
    )

    st.metric(
        "Test MAE",
        "0.4553",
    )

    st.subheader("Applicability Domain")

    st.write("**Validated threshold:** 0.55")
    st.write("**Validation coverage:** 58.08%")

    st.caption(
        "Predictions outside the validated chemical "
        "domain should be interpreted with caution."
    )


# ============================================================
# Title
# ============================================================

st.title("🧪 Molecular Solubility Predictor")

st.write(
    "Estimate the aqueous solubility (LogS) of a molecule "
    "from its structure and molecular descriptors."
)


# ============================================================
# Molecule Input Method
# ============================================================

st.subheader("Choose an input method")
st.caption("Select the workflow you want to use for your molecule.")

input_method = st.radio(
    "Input method",
    [
        "✏️ Draw Molecule",
        "🔤 Molecule Name",
        "</> Enter SMILES",
    ],
    horizontal=True,
    label_visibility="collapsed",
)


# ============================================================
# Clear previous prediction when input method changes
# ============================================================

if "last_input_method" not in st.session_state:

    st.session_state.last_input_method = input_method

elif input_method != st.session_state.last_input_method:

    st.session_state.prediction_result = None
    st.session_state.predicted_smiles = ""

    st.session_state.last_input_method = input_method


# ============================================================
# Draw Molecule
# ============================================================

drawn_smiles = ""
direct_smiles = ""
molecule_name = ""


if input_method == "✏️ Draw Molecule":

    st.caption("Draw a molecule to generate the SMILES used for prediction.")

    drawn_smiles = st_ketcher("")


# ============================================================
# Molecule Name
# ============================================================
elif input_method == "🔤 Molecule Name":

    molecule_name = st.text_input(
        "Molecule name",
        placeholder="Example: ethanol",
        key="molecule_name",
    )
    st.caption("Type a name and click Convert to SMILES.")

    if molecule_name.strip():

        if molecule_name.strip() != st.session_state.name_smiles_source:
            st.session_state.name_smiles = ""
            st.session_state.name_smiles_source = ""

        if st.button("Convert to SMILES", type="secondary"):

            try:

                converted_smiles = name_to_smiles(
                    molecule_name.strip()
                )

                if converted_smiles:

                    st.session_state.name_smiles = converted_smiles
                    st.session_state.name_smiles_source = molecule_name.strip()

                else:

                    st.error(
                        "Molecule not found in PubChem."
                    )

            except requests.RequestException:

                st.error(
                    "Could not connect to PubChem. "
                    "Please try again."
                )

            except (KeyError, IndexError):

                st.error(
                    "PubChem returned an unexpected response."
                )


    # --------------------------------------------------------
    # Display current SMILES for prediction
    # --------------------------------------------------------

    if st.session_state.name_smiles and (
        molecule_name.strip()
        and molecule_name.strip() == st.session_state.name_smiles_source
    ):

        st.caption("Current SMILES for prediction")

        st.code(
            st.session_state.name_smiles,
            language="text",
        )

# ============================================================
# Direct SMILES
# ============================================================

elif input_method == "</> Enter SMILES":

    direct_smiles = st.text_input(
        "SMILES",
        placeholder="Example: CCO",
        key="direct_smiles",
    )

    if direct_smiles.strip():

        st.caption("Molecule used for prediction")

        st.code(
            direct_smiles.strip(),
            language="text",
        )
    
# ============================================================
# Determine current SMILES
# ============================================================

current_smiles = ""

if input_method == "✏️ Draw Molecule":
    current_smiles = drawn_smiles.strip()

elif input_method == "🔤 Molecule Name":
    if (
        st.session_state.name_smiles
        and molecule_name.strip()
        and molecule_name.strip() == st.session_state.name_smiles_source
    ):
        current_smiles = st.session_state.name_smiles.strip()

elif input_method == "</> Enter SMILES":
    current_smiles = direct_smiles.strip()

# ============================================================
# Show generated SMILES
# ============================================================

if input_method == "✏️ Draw Molecule" and drawn_smiles:

    st.caption("Generated SMILES for prediction")

    st.code(
        drawn_smiles,
        language="text",
    )

# ============================================================
# Clear result when user changes molecule
# ============================================================

if st.session_state.predicted_smiles and (
    not current_smiles
    or current_smiles != st.session_state.predicted_smiles
):

    st.session_state.prediction_result = None
    st.session_state.predicted_smiles = ""


# ============================================================
# Prediction Button
# ============================================================

if st.button(
    "Predict Solubility",
    type="primary",
):

    if not current_smiles:

        if molecule_name.strip():

            st.warning(
                "This name hasn't been converted to a SMILES string yet. "
                "Click 'Convert to SMILES' above before predicting."
            )

        else:

            st.warning(
                "Please draw a molecule or enter a SMILES string."
            )

    else:

        try:

            # ------------------------------------------------
            # Validate SMILES
            # ------------------------------------------------

            mol = Chem.MolFromSmiles(current_smiles)

            if mol is None:

                st.error(
                    "Invalid SMILES. "
                    "Please enter a valid chemical SMILES."
                )

                st.stop()


            # ------------------------------------------------
            # Run prediction
            # ------------------------------------------------

            prediction = predict_solubility(current_smiles)

            ad_result = check_applicability(current_smiles)

            properties = molecular_properties(current_smiles)


            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            st.session_state.prediction_result = {
                "smiles": current_smiles,
                "prediction": prediction,
                "ad_result": ad_result,
                "properties": properties,
            }

            st.session_state.predicted_smiles = current_smiles


        except ValueError as error:

            st.error(str(error))


        except (TypeError, KeyError, RuntimeError) as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ============================================================
# DISPLAY STORED PREDICTION
# ============================================================

result = st.session_state.prediction_result


if result is not None:

    smiles = result["smiles"]
    prediction = result["prediction"]
    ad_result = result["ad_result"]
    properties = result["properties"]

    mol = Chem.MolFromSmiles(smiles)

    # ========================================================
    # Molecular Structure + Prediction
    # ========================================================

    col_structure, col_prediction = st.columns(2)


    with col_structure:

        st.subheader("Molecular Structure")

        image = render_molecule(mol, size=400)

        st.image(image)


    with col_prediction:

        st.subheader("Prediction")

        with st.container(border=True):
            st.metric(
                "Predicted LogS",
                f"{prediction:.4f}",
            )
            st.caption(
                "Aqueous solubility at 25°C (mol/L, log scale)"
            )


    # ========================================================
    # Molecular Properties
    # ========================================================

    st.subheader("Molecular Properties")

    with st.container(border=True):

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Molecular Weight",
                f"{properties['Molecular Weight']:.2f}",
            )
            st.caption("molecular mass in g/mol.")

        with col2:
            st.metric(
                "LogP",
                f"{properties['LogP']:.2f}",
            )
            st.caption("a measure of molecular lipophilicity.")

        with col3:
            st.metric(
                "H-Bond Donors (HBD)",
                int(properties["H-Bond Donors"]),
            )
            st.caption("number of hydrogen-bond donor sites.")

        with col4:
            st.metric(
                "H-Bond Acceptors (HBA)",
                int(properties["H-Bond Acceptors"]),
            )
            st.caption("number of hydrogen-bond acceptor sites.")

        with col5:
            st.metric(
                "TPSA",
                f"{properties['TPSA']:.2f}",
            )
            st.caption("topological polar surface area in Å².")


    # ========================================================
    # Applicability Domain
    # ========================================================

    st.subheader("Applicability Domain")

    with st.container(border=True):

        # Similarity metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Max Similarity",
                f"{ad_result['max_similarity']:.4f}",
                delta=None,
            )

        with col2:
            st.metric(
                "Mean Top-5",
                f"{ad_result['mean_top5']:.4f}",
                delta=None,
            )

        with col3:
            st.metric(
                "Threshold",
                "0.55",
                delta=None,
            )

    # AD Status
    status = ad_result["status"]
    max_sim = ad_result['max_similarity']

    if status == "HIGH CONFIDENCE":

        st.success(
            f"🟢 **HIGH CONFIDENCE**\n\n"
            f"Maximum similarity: **{max_sim:.4f}**\n\n"
            f"The molecule shows strong similarity to compounds "
            f"in the training set. The prediction is highly reliable."
        )

    elif status == "MODERATE CONFIDENCE / INSIDE DOMAIN":

        st.warning(
            f"🟡 **MODERATE CONFIDENCE**\n\n"
            f"Maximum similarity: **{max_sim:.4f}**\n\n"
            f"The molecule is within the validated applicability domain "
            f"(similarity ≥ 0.55), but the prediction should be "
            f"interpreted with appropriate caution."
        )

    else:

        st.error(
            f"🔴 **LOW CONFIDENCE / OUTSIDE DOMAIN**\n\n"
            f"Maximum similarity: **{max_sim:.4f}**\n\n"
            f"The molecule is outside the validated applicability domain "
            f"(similarity < 0.55) and the prediction may be unreliable."
        )

    # ========================================================
    # Most Similar Training Molecules
    # ========================================================

    st.subheader("Most Similar Training Molecules")

    with st.expander("Show Top 5 Similar Molecules"):

        for i, (
            similarity,
            similar_smiles,
        ) in enumerate(
            ad_result["top_5"],
            start=1,
        ):

            col_rank, col_sim, col_smiles = st.columns([1, 2, 7])

            with col_rank:
                st.write(f"**#{i}**")

            with col_sim:
                st.write(f"**{similarity:.4f}**")

            with col_smiles:
                st.code(similar_smiles, language="text")