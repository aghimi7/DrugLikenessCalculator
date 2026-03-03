import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# 1. Architecture (Renamed for the Calculator project)
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

# 2. Load Assets (Updated to look for DrugLikenessModel.pth)
@st.cache_resource
def load_assets():
    scaler = joblib.load("scaler.pkl")
    model = DrugLikenessNN(1241) 
    # Must match the filename in your GitHub repo exactly
    model.load_state_dict(torch.load("DrugLikenessModel.pth", map_location='cpu'))
    model.eval()
    return model, scaler

# 3. Featurization Logic
def featurize(smiles, scaler):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    # 217 PhysChem Descriptors
    names = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
    phys = np.array(calc.CalcDescriptors(mol)).reshape(1, -1)
    
    # 1,024 Morgan Bits
    bits = np.array(GetMorganFingerprintAsBitVect(mol, 2, 1024)).reshape(1, -1)
    
    # Scaling and Clipping
    phys_scaled = np.clip(scaler.transform(phys), -100, 100)
    X = np.hstack([phys_scaled, bits])
    return torch.tensor(X, dtype=torch.float32)

# 4. User Interface
st.set_page_config(page_title="Drug Likeness Calculator", page_icon="💊")

st.title("💊 Drug Likeness Calculator")
st.markdown("""
This tool uses a high-resolution **Self-Normalizing Neural Network (SNN)** 
trained on the ChEMBL 34 database to predict if a molecule possesses 
the structural and electronic signatures characteristic of a drug.
""")

smiles_input = st.text_input("Enter SMILES String:", "Cc1cc(-c2csc(N=C(N)N)n2)cn1C")

if st.button("Calculate Probability"):
    model, scaler = load_assets()
    features = featurize(smiles_input, scaler)
    
    if features is not None:
        with torch.no_grad():
            prob = model(features).item()
        
        st.write("---")
        score = prob * 100
        
        if prob > 0.5:
            st.success(f"### Score: {score:.1f}% (Drug-like)")
        else:
            st.error(f"### Score: {score:.1f}% (Non-drug-like)")
            
        st.progress(prob)
        st.caption("Based on 1,241 physicochemical and structural descriptors.")
    else:
        st.error("Invalid SMILES. Please check the structure.")

st.sidebar.title("Technical Stats")
st.sidebar.markdown("""
### Model Specifications
- **Architecture:** 4-Layer SNN
- **Input Resolution:** 1,241 Descriptors
- **Validation Accuracy:** 92.0%
- **Matthews Correlation (MCC):** 0.84

""")
