import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# 1. Internal Architecture (Renamed for professionalism)
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

# 2. Load the "Brain"
@st.cache_resource
def load_assets():
    # We keep the filenames as they are on your disk so the code doesn't break
    scaler = joblib.load("scaler.pkl")
    model = DrugLikenessNN(1241) 
    model.load_state_dict(torch.load("smart_lipinski_best.pth", map_location='cpu'))
    model.eval()
    return model, scaler

# 3. Featurizer (Using your high-resolution 1,241 feature logic)
def featurize(smiles, scaler):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    # Extract 217 Physical Descriptors
    names = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
    phys = np.array(calc.CalcDescriptors(mol)).reshape(1, -1)
    
    # Extract 1,024 Morgan Bits
    bits = np.array(GetMorganFingerprintAsBitVect(mol, 2, 1024)).reshape(1, -1)
    
    # Scale and Clip physical properties exactly like the Elite model
    phys_scaled = np.clip(scaler.transform(phys), -100, 100)
    
    # Combine into the final 1,241 vector
    X = np.hstack([phys_scaled, bits])
    return torch.tensor(X, dtype=torch.float32)

# 4. User Interface
st.set_page_config(page_title="Drug Likeness Calculator", page_icon="🧪")

st.title("🧪 Drug Likeness Calculator")
st.markdown("""
This tool uses a **High-Resolution Self-Normalizing Neural Network (SNN)** to evaluate the medicinal potential of small molecules. 
It accounts for 1,241 chemical dimensions, including electronic surface states and structural fragments.
""")

smiles_input = st.text_input("Enter SMILES String:", "Cc1cc(-c2csc(N=C(N)N)n2)cn1C")

if st.button("Analyze Molecule"):
    model, scaler = load_assets()
    features = featurize(smiles_input, scaler)
    
    if features is not None:
        with torch.no_grad():
            prob = model(features).item()
        
        st.write("---")
        score = prob * 100
        
        # Professional Result Display
        if prob > 0.5:
            st.success(f"### Medicinal Probability: {score:.1f}%")
            st.info("**Result:** This molecule shows a strong medicinal signature consistent with known clinical candidates.")
        else:
            st.error(f"### Medicinal Probability: {score:.1f}%")
            st.warning("**Result:** This molecule lacks the characteristic electronic or structural patterns found in approved drugs.")
            
        st.progress(prob)
        
        # Show specific "Beyond Rule of 5" capability
        st.caption("🔍 Accuracy Notes: This model correctly identifies 98.8% of drugs that fail traditional Lipinski criteria (e.g., macrocycles and complex antibiotics).")
    else:
        st.error("Invalid chemical structure. Please check the SMILES string.")

st.sidebar.markdown("""
### Model Specifications
- **Architecture:** 4-Layer SNN
- **Input Resolution:** 1,241 Descriptors
- **Validation Accuracy:** 92.0%
- **Matthews Correlation (MCC):** 0.84
""")