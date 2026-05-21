import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import json
from scipy.special import expit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CORE ARCHITECTURE
# ==========================================
class DrugLikenessModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.BatchNorm1d(2048), nn.SELU(),
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.SELU(), nn.AlphaDropout(0.2),
            nn.Linear(1024, 512), nn.SELU(),
            nn.Linear(512, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

@st.cache_resource
def load_assets():
    scaler = joblib.load("scaler_augmented.pkl")
    with open("feature_names_augmented.json", "r") as f:
        feature_order = json.load(f)
    model = DrugLikenessModel(len(feature_order))
    model.load_state_dict(torch.load("DrugLikenessModel.pth", map_location='cpu'))
    model.eval()
    return model, scaler, feature_order

# ==========================================
# 2. FEATURE EXTRACTION & SCALING
# ==========================================
def process_molecule(smiles, scaler, feature_order):
    clean_smi = "".join(smiles.split())
    mol = Chem.MolFromSmiles(clean_smi)
    if not mol: return None, None, None

    # Calculate all RDKit properties
    all_rdkit_funcs = {name: func for name, func in Descriptors._descList}
    fp = list(GetMorganFingerprintAsBitVect(mol, 2, 1024))
    
    raw_features = []
    raw_desc_dict = {}
    
    for name in feature_order:
        if name.startswith("bit_"):
            val = fp[int(name.split("_")[1])]
        else:
            val = all_rdkit_funcs.get(name, lambda x: 0.0)(mol)
            if np.isnan(val) or np.isinf(val): val = 0.0
            raw_desc_dict[name] = val
        raw_features.append(val)
        
    X_raw = np.array(raw_features).reshape(1, -1)
    
    # Scale only the continuous features correctly
    cont_idx = [i for i, f in enumerate(feature_order) if not f.startswith("bit_")]
    X_scaled = X_raw.copy().astype(np.float32)
    
    try:
        df_cont = pd.DataFrame(X_raw[:, cont_idx], columns=[feature_order[i] for i in cont_idx])
        X_scaled[:, cont_idx] = scaler.transform(df_cont)
    except:
        X_scaled[:, cont_idx] = scaler.transform(X_raw[:, cont_idx])
        
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    return X_tensor, raw_desc_dict, mol

# ==========================================
# 3. PIECEWISE FORMULA LOGIC
# ==========================================
def calculate_formula(mol, desc_dict):
    mw = Descriptors.MolWt(mol)
    
    # Safe fallback for NumArHetero (sometimes named differently in older RDKit versions)
    num_ar_hetero = desc_dict.get('NumArHetero', desc_dict.get('NumAromaticHeterocycles', 0.0))
    
    v = {
        'BCUT2D_CHGHI': desc_dict.get('BCUT2D_CHGHI', 0.0),
        'fr_Ar_N': desc_dict.get('fr_Ar_N', 0.0),
        'TPSA': desc_dict.get('TPSA', 0.0),
        'EState_VSA2': desc_dict.get('EState_VSA2', 0.0),
        'EState_VSA10': desc_dict.get('EState_VSA10', 0.0),
        'NumArHetero': num_ar_hetero,
        'fr_guanido': desc_dict.get('fr_guanido', 0.0),
        'RingCount': desc_dict.get('RingCount', 0.0),
        'SMR_VSA6': desc_dict.get('SMR_VSA6', 0.0),
        'Chi1n': desc_dict.get('Chi1n', 0.0)
    }

    if mw < 500:
        Z = (-9.7165 
             + 4.3450 * v['BCUT2D_CHGHI'] 
             + 0.4140 * v['fr_Ar_N'] 
             + 0.0100 * v['TPSA'] 
             + 0.0422 * v['EState_VSA2'] 
             + 0.0292 * v['EState_VSA10'] 
             + 0.2297 * v['NumArHetero'] 
             - 1.6688 * v['fr_guanido'] 
             - 0.1624 * v['RingCount'] 
             - 0.0265 * v['SMR_VSA6'] 
             - 0.1161 * v['Chi1n'])
    else:
        Z = (-17.2985 
             + 8.1213 * v['BCUT2D_CHGHI'] 
             + 0.0104 * v['TPSA'] 
             + 0.0406 * v['EState_VSA2'] 
             + 0.4550 * v['NumArHetero'] 
             + 0.1382 * v['fr_Ar_N'] 
             - 0.9592 * v['fr_guanido'] 
             - 0.0368 * v['EState_VSA10'] 
             - 0.2622 * v['RingCount'] 
             - 0.0095 * v['SMR_VSA6'] 
             - 0.1402 * v['Chi1n'])
             
    return expit(Z)

# ==========================================
# 4. USER INTERFACE (UX/UI)
# ==========================================
st.set_page_config(page_title="Advanced Drug-Likeness Predictor", page_icon="🧬", layout="wide")

# Custom CSS for a cleaner, modern look
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; font-weight: 700; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 30px; }
    .metric-card { background-color: #F3F4F6; padding: 20px; border-radius: 10px; text-align: center; }
    .prob-high { color: #059669; font-size: 2rem; font-weight: 800; }
    .prob-low { color: #DC2626; font-size: 2rem; font-weight: 800; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🧬 Advanced Drug-Likeness Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Going one step beyond Lipinski to define the physical characteristics of modern therapeutics.</p>', unsafe_allow_html=True)

st.info("💡 **About this Tool:** Lipinski's Rule of 5 laid the foundation for drug discovery. This tool does not aim to replace it, but rather expands upon it. By analyzing over 200,000 known bioactives, we capture the deeper, non-linear physicochemical gradients that define true modern drug-likeness.")

# Input Section
atorvastatin_smi = "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"
smiles_input = st.text_input("Enter a SMILES String to evaluate:", atorvastatin_smi)

if st.button("Evaluate Molecule", type="primary"):
    with st.spinner("Analyzing Molecular Signature..."):
        model, scaler, feature_order = load_assets()
        X_tensor, raw_desc_dict, mol_obj = process_molecule(smiles_input, scaler, feature_order)
        
        if X_tensor is not None:
            # 1. Deep Model Prediction
            with torch.no_grad():
                model_prob = model(X_tensor).item() * 100
                
            # 2. Formula Prediction
            formula_prob = calculate_formula(mol_obj, raw_desc_dict) * 100
            
            st.markdown("### 📊 Evaluation Results")
            col1, col2 = st.columns(2)
            
            # Column 1: The Model
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("#### Deep Model Prediction")
                color_class = "prob-high" if model_prob >= 50 else "prob-low"
                st.markdown(f'<p class="{color_class}">{model_prob:.1f}%</p>', unsafe_allow_html=True)
                st.progress(int(model_prob))
                st.caption("*Analyzes all 1,238 structural and physical features.*")
                st.markdown('</div>', unsafe_allow_html=True)

            # Column 2: The Formula
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("#### Formula-Based Prediction")
                color_class = "prob-high" if formula_prob >= 50 else "prob-low"
                st.markdown(f'<p class="{color_class}">{formula_prob:.1f}%</p>', unsafe_allow_html=True)
                st.progress(int(formula_prob))
                st.caption("*Distilled using only the top 10 highest-impact variables.*")
                st.markdown('</div>', unsafe_allow_html=True)

            # Physical Properties
            st.markdown("---")
            st.markdown("#### Primary Physicochemical Properties")
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            p_col1.metric("Molecular Weight", f"{Descriptors.MolWt(mol_obj):.1f} Da")
            p_col2.metric("LogP", f"{Descriptors.MolLogP(mol_obj):.2f}")
            p_col3.metric("TPSA", f"{Descriptors.TPSA(mol_obj):.1f} Å²")
            p_col4.metric("H-Bond Donors", f"{Descriptors.NumHDonors(mol_obj)}")
            
        else:
            st.error("Invalid SMILES format. Please check the structure and try again.")

# Transparency Section
st.markdown("---")
with st.expander("🔍 Transparency & Methodology (View the Formula)"):
    st.markdown("""
    ### How does this work?
    Our deep learning model evaluates molecules using **1,238 distinct physical and structural features**. While highly accurate, deep learning models are often considered "black boxes." 
    
    To ensure complete transparency, we analyzed the model to find the variables that capture the most signal. We distilled the model's logic into a simple, human-readable **Piecewise Mathematical Formula** using only the Top 10 features. The formula shifts its logic at the 500 Da mark to account for macrocycles and larger therapeutics (Beyond Rule of 5).
    """)
    
    st.latex(r"P = \frac{1}{1 + e^{-Z}}")
    
    st.markdown("**Tier 1: Standard Molecules (MW < 500 Da)**")
    st.latex(r"""
    Z = -9.72 + 4.35(\text{BCUT2D}) + 0.41(\text{fr\_Ar\_N}) + 0.01(\text{TPSA}) + 0.04(\text{VSA2}) + 0.03(\text{VSA10}) 
    """)
    st.latex(r"""
    + 0.23(\text{ArHetero}) - 1.67(\text{fr\_guanido}) - 0.16(\text{Rings}) - 0.03(\text{SMR}) - 0.12(\text{Chi1n})
    """)
    
    st.markdown("**Tier 2: Macrocycles & Large Therapeutics (MW ≥ 500 Da)**")
    st.latex(r"""
    Z = -17.30 + 8.12(\text{BCUT2D}) + 0.01(\text{TPSA}) + 0.04(\text{VSA2}) + 0.45(\text{ArHetero}) + 0.14(\text{fr\_Ar\_N})
    """)
    st.latex(r"""
    - 0.96(\text{fr\_guanido}) - 0.04(\text{VSA10}) - 0.26(\text{Rings}) - 0.01(\text{SMR}) - 0.14(\text{Chi1n})
    """)
    
    st.caption("Variables: BCUT2D_CHGHI (charge distribution), fr_Ar_N (aromatic nitrogens), TPSA (polar surface area), EState_VSA (electrotopological states), NumArHetero (aromatic heteroatoms), fr_guanido (guanidine groups), RingCount (total rings), SMR_VSA6 (polarizability), Chi1n (structural branching).")
