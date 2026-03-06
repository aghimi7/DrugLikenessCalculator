import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import json
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# 1. ARCHITECTURE (Final 208k-sample Elite Version)
class DrugLikenessNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.BatchNorm1d(2048), nn.SELU(),
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.SELU(),
            nn.AlphaDropout(0.2),
            nn.Linear(1024, 512), nn.SELU(),
            nn.Linear(512, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# 2. ASSET LOADER (Points to the new professional filename)
@st.cache_resource
def load_assets():
    scaler = joblib.load("scaler_augmented.pkl")
    with open("feature_names_augmented.json", "r") as f:
        feature_order = json.load(f)
    model = DrugLikenessNN(len(feature_order)) # 1238 features
    model.load_state_dict(torch.load("DrugLikenessModel.pth", map_location='cpu'))
    model.eval()
    return model, scaler, feature_order

# 3. ROBUST FEATURIZER (Standardizes input to match training data)
def featurize_medicinal(smiles, scaler, feature_order):
    clean_smi = "".join(smiles.split())
    raw_mol = Chem.MolFromSmiles(clean_smi)
    if not raw_mol: return None, None
    
    # Force Canonicalization (The fix for Atorvastatin/Hexane alignment)
    chembl_style_smi = Chem.MolToSmiles(raw_mol, isomericSmiles=True, canonical=True)
    mol = Chem.MolFromSmiles(chembl_style_smi)

    all_rdkit_funcs = {name: func for name, func in Descriptors._descList}
    fp = list(GetMorganFingerprintAsBitVect(mol, 2, 1024))
    bit_dict = {f'bit_{i}': val for i, val in enumerate(fp)}

    final_vector = []
    for name in feature_order:
        if name in all_rdkit_funcs: val = all_rdkit_funcs[name](mol)
        elif name == 'qed': val = Descriptors.qed(mol)
        elif name in bit_dict: val = bit_dict[name]
        else: val = 0.0
        final_vector.append(val)
        
    X = np.nan_to_num(np.array(final_vector).reshape(1, -1))
    
    # Scaling (214 physical descriptors)
    X_phys = X[:, :214] 
    X_bits = X[:, 214:]
    X_phys_scaled = np.clip(scaler.transform(X_phys), -15, 15)
    X_final = np.hstack([X_phys_scaled, X_bits])
    
    return torch.tensor(X_final, dtype=torch.float32), mol

# 4. USER INTERFACE
st.set_page_config(page_title="Drug Likeness Calculator", page_icon="💊")
st.title("💊 Drug Likeness Calculator")
st.markdown("#### High-Resolution Neural Predictor")

atorvastatin_smi = "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"
smiles_input = st.text_area("Enter SMILES String:", atorvastatin_smi, height=100)

if st.button("Evaluate Molecule"):
    with st.spinner("Analyzing Molecular Signature..."):
        model, scaler, feature_order = load_assets()
        features, mol_obj = featurize_medicinal(smiles_input, scaler, feature_order)
        
        st.write("---")
        if features is not None:
            with torch.no_grad():
                prob = model(features).item()
            
            score = prob * 100
            if prob > 0.5:
                st.success(f"### Medicinal Score: {score:.2f}% (Drug-like)")
            else:
                st.error(f"### Medicinal Score: {score:.2f}% (Non-drug-like)")
            
            st.progress(prob)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Mol Weight", f"{Descriptors.MolWt(mol_obj):.1f}")
            col2.metric("LogP", f"{Descriptors.MolLogP(mol_obj):.2f}")
            col3.metric("TPSA", f"{Descriptors.TPSA(mol_obj):.1f}")
        else:
            st.error("Invalid SMILES format.")

st.sidebar.info("""
**Model Specifications:**
- **Training Data:** 208,312 molecules (ChEMBL 34)
- **Features:** 1,238 High-Res Descriptors
- **Architecture:** 4-Layer Self-Normalizing NN

""")
