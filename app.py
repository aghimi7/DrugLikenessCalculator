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

@st.cache_resource
def load_assets():
    scaler = joblib.load("scaler.pkl")
    # Load the strict column order from HPC
    with open("feature_names.json", "r") as f:
        feature_order = json.load(f)
    model = DrugLikenessNN(1241) 
    model.load_state_dict(torch.load("DrugLikenessModel.pth", map_location='cpu'))
    model.eval()
    return model, scaler, feature_order

def featurize(smiles, scaler, feature_order):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    # Calculate ALL available descriptors
    names = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
    ds_values = calc.CalcDescriptors(mol)
    ds_dict = dict(zip(names, ds_values))
    
    # Calculate Bits
    fp = list(GetMorganFingerprintAsBitVect(mol, 2, 1024))
    for i, bit in enumerate(fp):
        ds_dict[f'bit_{i}'] = bit

    # FORCE DATA INTO THE EXACT ORDER USED DURING TRAINING
    final_features = []
    for col in feature_order:
        final_features.append(ds_dict.get(col, 0)) # Default to 0 if missing
    
    X = np.array(final_features).reshape(1, -1)
    
    # Separate Phys and Bits for scaling (Phys are the first 217)
    phys_scaled = np.clip(scaler.transform(X[:, :217]), -100, 100)
    X_final = np.hstack([phys_scaled, X[:, 217:]])
    
    return torch.tensor(X_final, dtype=torch.float32)

# --- UI Logic ---
st.set_page_config(page_title="Drug Likeness Calculator", page_icon="💊")
st.title("💊 Drug Likeness Calculator")

smiles_input = st.text_input("Enter SMILES String:", "CC1C(C(C(OC1OC2C3C=C4C=C3OC5=C(C=C(C=C5)C(C(C(=O)NC(C(=O)NC6C=C(C=C(C=C6OC7C(C(C(C(O7)CO)O)O)O)O2)OC8C(C(C(C(O8)CO)O)O)O)C(=O)NC(C4=O)C9=C(C=C(C=C9)O)O)N)O)Cl)O)O)(N)O")

if st.button("Calculate Probability"):
    model, scaler, feature_order = load_assets()
    features = featurize(smiles_input, scaler, feature_order)
    
    if features is not None:
        with torch.no_grad():
            prob = model(features).item()
        
        st.write("---")
        if prob > 0.5:
            st.success(f"### Score: {prob*100:.1f}% (Drug-like)")
        else:
            st.error(f"### Score: {prob*100:.1f}% (Non-drug-like)")
        st.progress(prob)
    else:
        st.error("Invalid SMILES structure.")

st.sidebar.title("Technical Stats")
st.sidebar.markdown("""
### Model Specifications
- **Architecture:** 4-Layer SNN
- **Input Resolution:** 1,241 Descriptors
- **Validation Accuracy:** 92.0%
- **Matthews Correlation (MCC):** 0.84

""")

