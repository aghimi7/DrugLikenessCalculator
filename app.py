import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


def calculate_drug_likeness(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None, "Invalid SMILES"

    # 1. Applicability Domain Filter (Safety First)
    # Drugs need at least one Nitrogen or Oxygen and a minimum weight
    tpsa = Descriptors.TPSA(mol)
    mw = Descriptors.MolWt(mol)
    h_atoms = mol.GetNumHeavyAtoms()
    
    if tpsa < 5 or h_atoms < 7 or mw < 100:
        # Prevents Hexane, DMSO, and Ethanol from being called drugs
        return 0.01, "Small/Simple Molecule Filter"

    # 2. Extract our 10 High-Resolution Heavyweights
    # These are the features the NN said were most important
    vals = {
        'fr_Ar_N': Descriptors.fr_Ar_N(mol),
        'BCUT2D_CHGHI': Descriptors.BCUT2D_CHGHI(mol),
        'NumAmideBonds': Descriptors.NumAmideBonds(mol),
        'fr_ether': Descriptors.fr_ether(mol),
        'PEOE_VSA3': Descriptors.PEOE_VSA3(mol),
        'EState_VSA2': Descriptors.EState_VSA2(mol),
        'fr_NH2': Descriptors.fr_NH2(mol),
        'fr_imidazole': Descriptors.fr_imidazole(mol),
        'NumUnspecified': Descriptors.NumUnspecifiedAtomStereoCenters(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol)
    }

    # 3. Apply the Distilled Formula (Calibrated for High Precision)
    # Z = Intercept + Boosters - Brakes
    z = -5.50  # Balanced Intercept
    z += (0.55 * vals['fr_Ar_N'])
    z += (2.50 * vals['BCUT2D_CHGHI'])
    z += (0.65 * vals['NumAmideBonds'])
    z += (0.35 * vals['fr_ether'])
    z += (0.05 * vals['PEOE_VSA3'])
    z += (0.04 * vals['EState_VSA2'])
    z -= (0.67 * vals['fr_NH2'])
    z -= (1.05 * vals['fr_imidazole'])
    z -= (0.53 * vals['NumUnspecified'])
    z -= (0.15 * vals['RotBonds'])

    prob = 1 / (1 + np.exp(-z))
    return prob, vals

# --- STREAMLIT UI ---
st.set_page_config(page_title="Drug Likeness Calculator", page_icon="💊")
st.title("💊 Drug Likeness Calculator")

st.markdown("A successor to the Rule of 5, utilizing high-resolution electronic and structural descriptors.")

# Clean up input (removes any accidental spaces/newlines)
smiles_input = st.text_input("Enter SMILES:", "CC(C)c1c(C(=O)Nc2ccccc2)c(c(n1CCC(O)CC(O)CC(=O)O)c3ccc(F)cc3)c4ccccc4").strip()

if st.button("Calculate Score"):
    prob, data = calculate_drug_likeness(smiles_input)
    
    if prob is not None:
        st.write("---")
        score = prob * 100
        if prob > 0.5:
            st.success(f"### Medicinal Score: {score:.1f}% (Drug-like)")
        else:
            st.error(f"### Medicinal Score: {score:.1f}% (Non-drug-like)")
        
        st.progress(prob)
        
        # Display the real science
        col1, col2, col3 = st.columns(3)
        mol = Chem.MolFromSmiles(smiles_input)
        col1.metric("Weight", f"{Descriptors.MolWt(mol):.1f}")
        col2.metric("LogP", f"{Descriptors.MolLogP(mol):.2f}")
        col3.metric("Aromatic N", int(Descriptors.fr_Ar_N(mol)))
    else:
        st.error("Invalid SMILES structure. Please ensure it is a valid organic molecule.")

st.sidebar.markdown("""
### The Analytical Rule of 10
This calculator is an **analytical distillation** of a Self-Normalizing Neural Network. 
It replaces the Rule of 5 with a weighted index of electronic charge and heterocyclic density.
""")




