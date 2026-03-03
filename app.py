import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import json
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# 1. Architecture
class DrugLikenessNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.SELU(), nn.AlphaDropout(0.1),
            nn.Linear(1024, 512), nn.SELU(), nn.AlphaDropout(0.1),
            nn.Linear(512, 256), nn.SELU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# 2. Load Assets
@st.cache_resource
def load_assets():
    scaler = joblib.load("scaler.pkl")
    with open("feature_names.json", "r") as f:
        feature_order = json.load(f)
    model = DrugLikenessNN(1241) 
    model.load_state_dict(torch.load("DrugLikenessModel.pth", map_location='cpu'))
    model.eval()
    return model, scaler, feature_order

# 3. Featurization with Safety Guardrail
def featurize(smiles, scaler, feature_order):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    # --- APPLICABILITY DOMAIN FILTER (The Safety Guardrail) ---
    # Molecules must have heteroatoms and polarity to be medicinal
    tpsa = Descriptors.TPSA(mol)
    heteroatoms = Descriptors.NumHeteroatoms(mol)
    if tpsa == 0 or heteroatoms == 0:
        return "OUT_OF_DOMAIN"
    
    # Standard featurization
    available_descriptors = {name: func(mol) for name, func in Descriptors._descList}
    fp = list(GetMorganFingerprintAsBitVect(mol, 2, 1024))
    for i, bit in enumerate(fp):
        available_descriptors[f'bit_{i}'] = bit

    final_vector = [available_descriptors.get(col, 0.0) for col in feature_order]
    X = np.nan_to_num(np.array(final_vector).reshape(1, -1))

    # Scale only the first 217 (Physical descriptors)
    phys_scaled = np.clip(scaler.transform(X[:, :217]), -100, 100)
    X_final = np.hstack([phys_scaled, X[:, 217:]])
    return torch.tensor(X_final, dtype=torch.float32)

# 4. User Interface
st.set_page_config(page_title="Drug Likeness Calculator", page_icon="💊")
st.title("💊 Drug Likeness Calculator")

st.markdown("""
Evaluate the medicinal potential of a molecule using a **High-Resolution Self-Normalizing Neural Network**. 
This model analyzes 1,241 chemical dimensions to predict drug-likeness.
""")

# Default changed to Atorvastatin (Lipitor)
atorvastatin_smiles = "CC(C)c1c(C(=O)Nc2ccccc2)c(c(n1CCC(O)CC(O)CC(=O)O)c3ccc(F)cc3)c4ccccc4"
smiles_input = st.text_input("Enter SMILES String:", atorvastatin_smiles)

if st.button("Calculate Drug-Likeness"):
    model, scaler, feature_order = load_assets()
    features = featurize(smiles_input, scaler, feature_order)
    
    st.write("---")
    
    if features == "OUT_OF_DOMAIN":
        st.error("### Score: 0.0% (Non-drug-like)")
        st.warning("**Safety Filter triggered:** This molecule is a pure hydrocarbon or lacks the minimum heteroatom density required for medicinal activity.")
        st.progress(0.0)
    elif features is not None:
        with torch.no_grad():
            prob = model(features).item()
        
        score = prob * 100
        if prob > 0.5:
            st.success(f"### Score: {score:.1f}% (Drug-like)")
            st.balloons()
        else:
            st.error(f"### Score: {score:.1f}% (Non-drug-like)")
        
        st.progress(prob)
        
        # Display key metrics for the user
        mol = Chem.MolFromSmiles(smiles_input)
        col1, col2, col3 = st.columns(3)
        col1.metric("Mol Weight", f"{Descriptors.MolWt(mol):.1f}")
        col2.metric("LogP", f"{Descriptors.MolLogP(mol):.2f}")
        col3.metric("TPSA", f"{Descriptors.TPSA(mol):.1f}")
    else:
        st.error("Invalid SMILES format. Please check the structure.")

st.sidebar.info("""
**Applicability Domain:**
This model is optimized for organic medicinal compounds. Simple solvents (like Hexane) or inorganic acids will be correctly filtered as non-drug-like.
""")

st.sidebar.title("Technical Stats")
st.sidebar.markdown("""
### Model Specifications
- **Architecture:** 4-Layer SNN
- **Input Resolution:** 1,241 Descriptors
- **Validation Accuracy:** 92.0%
- **Matthews Correlation (MCC):** 0.84

""")



