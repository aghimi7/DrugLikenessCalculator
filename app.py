import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import json
from rdkit import Chem
from rdkit.Chem import Descriptors
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
    with open("feature_names.json", "r") as f:
        feature_order = json.load(f)
    model = DrugLikenessNN(1241) 
    model.load_state_dict(torch.load("DrugLikenessModel.pth", map_location='cpu'))
    model.eval()
    return model, scaler, feature_order

def featurize(smiles, scaler, feature_order):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    # 1. Create a dictionary of ALL possible RDKit descriptors for this molecule
    # This is more reliable than the 'Calculator' object
    available_descriptors = {name: func(mol) for name, func in Descriptors._descList}
    
    # 2. Add structural bits to the dictionary
    fp = list(GetMorganFingerprintAsBitVect(mol, 2, 1024))
    for i, bit in enumerate(fp):
        available_descriptors[f'bit_{i}'] = bit

    # 3. Reconstruct the vector using the EXACT order from the HPC JSON
    # This guarantees that the weights match the chemistry
    final_vector = []
    for col_name in feature_order:
        # Fetch value, default to 0.0 if the descriptor is missing in this RDKit version
        val = available_descriptors.get(col_name, 0.0)
        final_vector.append(val)
    
    X = np.array(final_vector).reshape(1, -1)
    
    # 4. Clean NaNs/Infs (Just like we did in training)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 5. Scale physical props (Indices 0 to 216) and keep bits raw
    # Note: We use the feature names to find the split point dynamically
    phys_len = len([c for c in feature_order if 'bit_' not in c])
    
    X_phys = X[:, :phys_len]
    X_bits = X[:, phys_len:]
    
    # Apply scaling to physical part only
    X_phys_scaled = np.clip(scaler.transform(X_phys), -100, 100)
    X_final = np.hstack([X_phys_scaled, X_bits])
    
    return torch.tensor(X_final, dtype=torch.float32)

# --- UI Setup ---
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
        score = prob * 100
        if prob > 0.5:
            st.success(f"### Medicinal Score: {score:.1f}%")
            st.write("**Assessment:** Drug-like")
        else:
            st.error(f"### Medicinal Score: {score:.1f}%")
            st.write("**Assessment:** Non-drug-like")
        st.progress(prob)
    else:
        st.error("Invalid structure.")

st.sidebar.title("Technical Stats")
st.sidebar.markdown("""
### Model Specifications
- **Architecture:** 4-Layer SNN
- **Input Resolution:** 1,241 Descriptors
- **Validation Accuracy:** 92.0%
- **Matthews Correlation (MCC):** 0.84

""")


