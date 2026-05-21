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

def process_molecule(smiles, scaler, feature_order):
    clean_smi = "".join(smiles.split())
    mol = Chem.MolFromSmiles(clean_smi)
    if not mol:
        return None, None, None

    all_rdkit_funcs = {name: func for name, func in Descriptors._descList}
    fp = list(GetMorganFingerprintAsBitVect(mol, 2, 1024))

    raw_features = []
    raw_desc_dict = {}

    for name in feature_order:
        if name.startswith("bit_"):
            val = fp[int(name.split("_")[1])]
        else:
            val = all_rdkit_funcs.get(name, lambda x: 0.0)(mol)
            if np.isnan(val) or np.isinf(val):
                val = 0.0
            raw_desc_dict[name] = val
        raw_features.append(val)

    X_raw = np.array(raw_features).reshape(1, -1)
    cont_idx = [i for i, f in enumerate(feature_order) if not f.startswith("bit_")]
    X_scaled = X_raw.copy().astype(np.float32)

    try:
        df_cont = pd.DataFrame(X_raw[:, cont_idx], columns=[feature_order[i] for i in cont_idx])
        X_scaled[:, cont_idx] = scaler.transform(df_cont)
    except Exception:
        X_scaled[:, cont_idx] = scaler.transform(X_raw[:, cont_idx])

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    return X_tensor, raw_desc_dict, mol

def calculate_formula(mol, desc_dict):
    mw = Descriptors.MolWt(mol)
    num_ar_hetero = desc_dict.get('NumArHetero', desc_dict.get('NumAromaticHeterocycles', 0.0))

    v = {
        'BCUT2D_CHGHI': desc_dict.get('BCUT2D_CHGHI', 0.0),
        'fr_Ar_N':       desc_dict.get('fr_Ar_N', 0.0),
        'TPSA':          desc_dict.get('TPSA', 0.0),
        'EState_VSA2':   desc_dict.get('EState_VSA2', 0.0),
        'EState_VSA10':  desc_dict.get('EState_VSA10', 0.0),
        'NumArHetero':   num_ar_hetero,
        'fr_guanido':    desc_dict.get('fr_guanido', 0.0),
        'RingCount':     desc_dict.get('RingCount', 0.0),
        'SMR_VSA6':      desc_dict.get('SMR_VSA6', 0.0),
        'Chi1n':         desc_dict.get('Chi1n', 0.0),
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

st.set_page_config(
    page_title="DrugLikenessModel",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Source+Sans+3:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    color: #1a1a2e;
}

.stApp {
    background-color: #f7f7f5;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1100px; }

.masthead {
    border-bottom: 2px solid #1a1a2e;
    padding-bottom: 1.4rem;
    margin-bottom: 2.2rem;
}
.masthead-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.3rem;
}
.masthead-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: #1a1a2e;
    line-height: 1.15;
    margin: 0;
}
.masthead-sub {
    font-size: 1.0rem;
    font-weight: 300;
    color: #4b5563;
    margin-top: 0.5rem;
    max-width: 680px;
    line-height: 1.6;
}

.input-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.3rem;
}

.result-panel {
    background: #ffffff;
    border: 1.5px solid #e5e7eb;
    border-radius: 4px;
    padding: 1.8rem 2rem;
}
.result-panel-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 0.6rem;
}
.result-score {
    font-family: 'Playfair Display', serif;
    font-size: 3.4rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.result-score.high { color: #065f46; }
.result-score.low  { color: #991b1b; }
.result-verdict {
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    margin-bottom: 1.1rem;
}
.result-verdict.high { color: #059669; }
.result-verdict.low  { color: #dc2626; }
.result-caption {
    font-size: 0.8rem;
    color: #9ca3af;
    font-weight: 300;
    line-height: 1.5;
}

.stProgress > div > div > div > div {
    background-color: #1a1a2e !important;
    border-radius: 2px !important;
}
.stProgress > div > div > div {
    background-color: #e5e7eb !important;
    border-radius: 2px !important;
    height: 4px !important;
}

.prop-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-top: 1.6rem;
}
.prop-cell {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    padding: 1rem 1.2rem;
}
.prop-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 0.35rem;
}
.prop-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem;
    font-weight: 500;
    color: #1a1a2e;
}
.prop-unit {
    font-size: 0.72rem;
    color: #9ca3af;
    margin-left: 3px;
}

.divider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 2rem 0;
}

.section-head {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 1rem;
    margin-top: 2rem;
}

.method-box {
    background: #ffffff;
    border: 1.5px solid #e5e7eb;
    border-left: 3px solid #1a1a2e;
    border-radius: 4px;
    padding: 1.6rem 1.8rem;
    font-size: 0.9rem;
    color: #374151;
    line-height: 1.7;
}
.method-box p { margin: 0 0 0.8rem 0; }
.method-box p:last-child { margin-bottom: 0; }

.notice {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 4px;
    padding: 0.9rem 1.2rem;
    font-size: 0.85rem;
    color: #14532d;
    margin-bottom: 1.8rem;
}

.stButton > button {
    background-color: #1a1a2e !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.15s ease !important;
}
.stButton > button:hover {
    opacity: 0.82 !important;
}

.stTextInput > div > div > input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 3px !important;
    background: #ffffff !important;
    color: #1a1a2e !important;
    padding: 0.6rem 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1a1a2e !important;
    box-shadow: none !important;
}

.streamlit-expanderHeader {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #6b7280 !important;
    background: transparent !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 3px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="masthead">
    <div class="masthead-label">LSU · Biological Sciences</div>
    <div class="masthead-title">DrugLikenessModel</div>
    <div class="masthead-sub">
        A bioactivity-grounded successor to Lipinski's Rule of Five.
        Trained on 209,269 ChEMBL compounds across the full chemical space,
        including macrocycles and bRo5 therapeutics.
    </div>
</div>
""", unsafe_allow_html=True)

atorvastatin_smi = "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"

st.markdown('<div class="input-label">SMILES Input</div>', unsafe_allow_html=True)
smiles_input = st.text_input(
    label="smiles",
    value=atorvastatin_smi,
    label_visibility="collapsed",
    placeholder="Enter canonical SMILES string...",
)

st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
run = st.button("Run Analysis")

if run:
    with st.spinner(""):
        model, scaler, feature_order = load_assets()
        X_tensor, raw_desc_dict, mol_obj = process_molecule(smiles_input, scaler, feature_order)

    if X_tensor is None:
        st.markdown("""
        <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:4px;
                    padding:0.9rem 1.2rem;font-size:0.85rem;color:#991b1b;margin-top:1rem;">
            Invalid SMILES string. Please verify the structure and try again.
        </div>
        """, unsafe_allow_html=True)
    else:
        with torch.no_grad():
            model_prob = model(X_tensor).item() * 100
        formula_prob = calculate_formula(mol_obj, raw_desc_dict) * 100

        mw      = Descriptors.MolWt(mol_obj)
        logp    = Descriptors.MolLogP(mol_obj)
        tpsa    = Descriptors.TPSA(mol_obj)
        hbd     = Descriptors.NumHDonors(mol_obj)
        hba     = Descriptors.NumHAcceptors(mol_obj)
        rings   = Descriptors.RingCount(mol_obj)
        tier    = "Tier 2 — bRo5 / Macrocycle" if mw >= 500 else "Tier 1 — Standard"

        def score_class(p): return "high" if p >= 50 else "low"
        def verdict(p): return "Drug-like" if p >= 50 else "Non drug-like"

        st.markdown('<div class="section-head">Prediction Results</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            sc = score_class(model_prob)
            st.markdown(f"""
            <div class="result-panel">
                <div class="result-panel-title">Deep Model &nbsp;·&nbsp; 1,238 features</div>
                <div class="result-score {sc}">{model_prob:.1f}%</div>
                <div class="result-verdict {sc}">{verdict(model_prob)}</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(int(model_prob))
            st.markdown(f'<div class="result-caption">SNN evaluated across all 1,024 structural bits and 214 physicochemical descriptors. Molecule classified under <strong>{tier}</strong>.</div>', unsafe_allow_html=True)

        with col2:
            sc2 = score_class(formula_prob)
            st.markdown(f"""
            <div class="result-panel">
                <div class="result-panel-title">Piecewise Formula &nbsp;·&nbsp; Top 10 features</div>
                <div class="result-score {sc2}">{formula_prob:.1f}%</div>
                <div class="result-verdict {sc2}">{verdict(formula_prob)}</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(int(formula_prob))
            st.markdown('<div class="result-caption">Surrogate logistic regression distilled from model outputs. Interpretable but constrained — Pearson R = 0.32 against full model.</div>', unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-head">Physicochemical Profile</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prop-grid">
            <div class="prop-cell">
                <div class="prop-name">Mol. Weight</div>
                <div class="prop-value">{mw:.1f}<span class="prop-unit">Da</span></div>
            </div>
            <div class="prop-cell">
                <div class="prop-name">LogP</div>
                <div class="prop-value">{logp:.2f}</div>
            </div>
            <div class="prop-cell">
                <div class="prop-name">TPSA</div>
                <div class="prop-value">{tpsa:.1f}<span class="prop-unit">Å²</span></div>
            </div>
            <div class="prop-cell">
                <div class="prop-name">HB Donors</div>
                <div class="prop-value">{hbd}</div>
            </div>
            <div class="prop-cell">
                <div class="prop-name">HB Acceptors</div>
                <div class="prop-value">{hba}</div>
            </div>
            <div class="prop-cell">
                <div class="prop-name">Ring Count</div>
                <div class="prop-value">{rings}</div>
            </div>
            <div class="prop-cell">
                <div class="prop-name">MW Class</div>
                <div class="prop-value" style="font-size:0.82rem;margin-top:3px;">{"bRo5" if mw >= 500 else "Ro5"}</div>
            </div>
            <div class="prop-cell">
                <div class="prop-name">Ro5 Violations</div>
                <div class="prop-value">{sum([mw > 500, logp > 5, hbd > 5, hba > 10])}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='margin-top:2.5rem'></div>", unsafe_allow_html=True)
with st.expander("Model architecture & scoring formula"):
    st.markdown("""
    <div class="method-box">
    <p>
    <strong>Architecture.</strong> The DrugLikenessModel is a four-layer Self-Normalizing Neural Network (SNN)
    with 2,048 → 1,024 → 512 → 1 neuron layers. SELU activations and AlphaDropout (rate 0.2) enforce
    self-normalization across the full 1,238-dimensional input space. Continuous descriptors are normalized
    via a QuantileTransformer before entry; Morgan bits are left unscaled.
    </p>
    <p>
    <strong>Training data.</strong> 209,269 molecules from ChEMBL 34. Positives: IC50 &lt; 50 nM
    or Phase 1–4 clinical advancement. Negatives: verified biological inertness across all recorded assays,
    augmented with 1,000 synthetic structural decoys to correct for complexity bias.
    </p>
    <p>
    <strong>Piecewise formula.</strong> Top 10 descriptors selected by XGBoost feature importance.
    A surrogate logistic regression was trained on these 10 features against the SNN's continuous
    output probability (not the original binary labels), stratified by molecular weight to yield
    two tiers. Pearson R = 0.32 against full model; ~80% binary classification accuracy retained.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tier 1 — Standard molecules (MW < 500 Da)**")
        st.latex(r"Z = -9.72 + 4.35\,B + 0.41\,N_{Ar} + 0.01\,\text{TPSA} + 0.04\,V_2 + 0.03\,V_{10}"
                 r"+ 0.23\,H_{Ar} - 1.67\,G - 0.16\,R - 0.03\,S - 0.12\,\chi")
    with c2:
        st.markdown("**Tier 2 — bRo5 / Macrocycles (MW ≥ 500 Da)**")
        st.latex(r"Z = -17.30 + 8.12\,B + 0.01\,\text{TPSA} + 0.04\,V_2 + 0.45\,H_{Ar} + 0.14\,N_{Ar}"
                 r"- 0.96\,G - 0.04\,V_{10} - 0.26\,R - 0.01\,S - 0.14\,\chi")

    st.latex(r"P = \frac{1}{1 + e^{-Z}}")

    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#9ca3af;
                line-height:1.8;margin-top:0.8rem;">
    B = BCUT2D_CHGHI &nbsp;·&nbsp; N<sub>Ar</sub> = fr_Ar_N &nbsp;·&nbsp;
    V<sub>2,10</sub> = EState_VSA2/10 &nbsp;·&nbsp; H<sub>Ar</sub> = NumArHetero &nbsp;·&nbsp;
    G = fr_guanido &nbsp;·&nbsp; R = RingCount &nbsp;·&nbsp;
    S = SMR_VSA6 &nbsp;·&nbsp; χ = Chi1n
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:3rem;padding-top:1.2rem;border-top:1px solid #e5e7eb;
            display:flex;justify-content:space-between;align-items:center;
            font-size:0.75rem;color:#9ca3af;">
    <span style="font-family:'JetBrains Mono',monospace;letter-spacing:0.06em;">
        DrugLikenessModel · Louisiana State University
    </span>
    <span>Trained on ChEMBL 34 · 209,269 compounds</span>
</div>
""", unsafe_allow_html=True)
