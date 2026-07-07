import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import json
import io
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
        'BCUT2D_CHGHI':   desc_dict.get('BCUT2D_CHGHI', 0.0),
        'fr_Ar_N':        desc_dict.get('fr_Ar_N', 0.0),
        'TPSA':           desc_dict.get('TPSA', 0.0),
        'EState_VSA2':    desc_dict.get('EState_VSA2', 0.0),
        'EState_VSA10':   desc_dict.get('EState_VSA10', 0.0),
        'NumArHetero':    num_ar_hetero,
        'fr_guanido':     desc_dict.get('fr_guanido', 0.0),
        'RingCount':      desc_dict.get('RingCount', 0.0),
        'SMR_VSA6':       desc_dict.get('SMR_VSA6', 0.0),
        'Chi1n':          desc_dict.get('Chi1n', 0.0),
        'BertzCT':        desc_dict.get('BertzCT', 0.0),
        'FractionCSP3':   desc_dict.get('FractionCSP3', 0.0),
        'NumHAcceptors':  desc_dict.get('NumHAcceptors', 0.0),
    }

    if mw < 500:
        Z = (-7.1577
             + 2.6147 * v['BCUT2D_CHGHI']
             + 0.0544 * v['fr_Ar_N']
             + 0.0208 * v['TPSA']
             + 0.1122 * v['EState_VSA2']
             + 0.0602 * v['EState_VSA10']
             + 0.3898 * v['RingCount']
             + 0.3284 * v['Chi1n']
             + 0.0026 * v['BertzCT']
             + 0.3701 * v['NumHAcceptors']
             - 1.8925 * v['FractionCSP3'])
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

def get_ro5_violations(mw, logp, hbd, hba):
    violations = []
    if mw > 500:
        violations.append(f"Molecular weight {mw:.1f} Da  (limit: 500 Da)")
    if logp > 5:
        violations.append(f"LogP {logp:.2f}  (limit: 5)")
    if hbd > 5:
        violations.append(f"H-bond donors {hbd}  (limit: 5)")
    if hba > 10:
        violations.append(f"H-bond acceptors {hba}  (limit: 10)")
    return violations

def evaluate_single(smiles, model, scaler, feature_order):
    X_tensor, raw_desc_dict, mol = process_molecule(smiles, scaler, feature_order)
    if X_tensor is None:
        return None
    with torch.no_grad():
        model_prob = model(X_tensor).item() * 100
    formula_prob = calculate_formula(mol, raw_desc_dict) * 100
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)
    rings = Descriptors.RingCount(mol)
    n_rot = Descriptors.NumRotatableBonds(mol)
    tier  = "Tier 2 — bRo5" if mw >= 500 else "Tier 1 — Standard"
    ro5_v = get_ro5_violations(mw, logp, hbd, hba)
    return {
        "mol":          mol,
        "model_prob":   model_prob,
        "formula_prob": formula_prob,
        "mw":           mw,
        "logp":         logp,
        "tpsa":         tpsa,
        "hbd":          hbd,
        "hba":          hba,
        "rings":        rings,
        "n_rot":        n_rot,
        "tier":         tier,
        "ro5_v":        ro5_v,
        "desc":         raw_desc_dict,
    }

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
.stApp { background-color: #f7f7f5; }
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
    margin-bottom: 0.5rem;
}
.result-verdict.high { color: #059669; }
.result-verdict.low  { color: #dc2626; }
.result-caption {
    font-size: 0.8rem;
    color: #9ca3af;
    font-weight: 300;
    line-height: 1.5;
}
.result-note {
    font-size: 0.78rem;
    color: #6b7280;
    background: #f9fafb;
    border-left: 3px solid #d1d5db;
    padding: 0.5rem 0.75rem;
    margin-top: 0.8rem;
    border-radius: 0 3px 3px 0;
    line-height: 1.5;
}
.result-note.primary {
    border-left-color: #1a1a2e;
    background: #f0f0ef;
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

.tier-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.25rem 0.65rem;
    border-radius: 2px;
    font-weight: 500;
    margin-bottom: 1rem;
}
.tier-badge.t1 {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
}
.tier-badge.t2 {
    background: #fdf4ff;
    color: #7e22ce;
    border: 1px solid #e9d5ff;
}

.prop-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-top: 1.2rem;
}
.prop-cell {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    padding: 1rem 1.2rem;
}
.prop-cell.tier1 { border-top: 3px solid #1d4ed8; }
.prop-cell.tier2 { border-top: 3px solid #7e22ce; }
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
.prop-value.mw-t1 { color: #1d4ed8; }
.prop-value.mw-t2 { color: #7e22ce; }
.prop-unit {
    font-size: 0.72rem;
    color: #9ca3af;
    margin-left: 3px;
}

.violation-item {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #991b1b;
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 3px;
    padding: 0.3rem 0.7rem;
    margin-bottom: 0.3rem;
}
.no-violation {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #065f46;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 3px;
    padding: 0.3rem 0.7rem;
}

.divider { border: none; border-top: 1px solid #e5e7eb; margin: 2rem 0; }

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
.stButton > button:hover { opacity: 0.82 !important; }

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

.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

from pathlib import Path
import base64

def load_logo(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_path = Path("logo_cropped.png")
if logo_path.exists():
    logo_b64 = load_logo(logo_path)
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="height:52px;margin-bottom:0.4rem;">'
else:
    logo_html = ""

masthead_html = """
<div class="masthead">
    <div class="masthead-label">Drug-Likeness Prediction</div>
    """ + logo_html + """
    <div class="masthead-title">DrugLikenessModel</div>
    <div class="masthead-sub">
        A bioactivity-grounded successor to Lipinski's Rule of Five.
        Trained on 209,269 ChEMBL compounds across standadr and beyond-Rule-of-Five chemical space,
        including macrocycles and bRo5 therapeutics.
    </div>
</div>
"""

st.markdown(masthead_html, unsafe_allow_html=True)

tab_single, tab_batch = st.tabs(["Single Molecule", "Batch / CSV"])

atorvastatin_smi = "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"

with tab_single:
    st.markdown('<div class="input-label">SMILES Input</div>', unsafe_allow_html=True)
    smiles_input = st.text_input(
        label="smiles",
        value=atorvastatin_smi,
        label_visibility="collapsed",
        placeholder="Enter canonical SMILES string...",
    )
    st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
    run = st.button("Run Analysis", key="single_run")

    if run:
        with st.spinner(""):
            model, scaler, feature_order = load_assets()
            result = evaluate_single(smiles_input, model, scaler, feature_order)

        if result is None:
            st.markdown("""
            <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:4px;
                        padding:0.9rem 1.2rem;font-size:0.85rem;color:#991b1b;margin-top:1rem;">
                Invalid SMILES string. Please verify the structure and try again.
            </div>
            """, unsafe_allow_html=True)
        else:
            mw          = result["mw"]
            logp        = result["logp"]
            tpsa        = result["tpsa"]
            hbd         = result["hbd"]
            hba         = result["hba"]
            rings       = result["rings"]
            n_rot       = result["n_rot"]
            tier        = result["tier"]
            ro5_v       = result["ro5_v"]
            model_prob  = result["model_prob"]
            formula_prob = result["formula_prob"]
            is_t2       = mw >= 500

            def sc(p): return "high" if p >= 50 else "low"
            def vd(p): return "Drug-like" if p >= 50 else "Non drug-like"

            st.markdown('<div class="section-head">Prediction Results</div>', unsafe_allow_html=True)

            tier_cls  = "t2" if is_t2 else "t1"
            tier_label = "Tier 2 — bRo5 / Macrocycle  ·  MW ≥ 500 Da" if is_t2 else "Tier 1 — Standard  ·  MW < 500 Da"

            col1, col2 = st.columns(2, gap="medium")

            with col1:
                st.markdown(f"""
                <div class="result-panel">
                    <div class="result-panel-title">Deep Model &nbsp;·&nbsp; 1,238 features</div>
                    <div class="tier-badge {tier_cls}">{tier_label}</div>
                    <div class="result-score {sc(model_prob)}">{model_prob:.1f}%</div>
                    <div class="result-verdict {sc(model_prob)}">{vd(model_prob)}</div>
                    <div class="result-note primary">
                        This is the primary score. The deep model evaluates all 1,024 structural
                        fingerprint bits and 214 physicochemical descriptors simultaneously.
                        Use this score for decision-making.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(int(model_prob))

            with col2:
                st.markdown(f"""
                <div class="result-panel">
                    <div class="result-panel-title">Piecewise Formula &nbsp;·&nbsp; Top 10 features</div>
                    <div class="tier-badge {tier_cls}">{tier_label}</div>
                    <div class="result-score {sc(formula_prob)}">{formula_prob:.1f}%</div>
                    <div class="result-verdict {sc(formula_prob)}">{vd(formula_prob)}</div>
                    <div class="result-note">
                        This score is for interpretability only. It uses 10 descriptors distilled
                        from the deep model and retains ~75% of its accuracy (Pearson R = 0.32).
                        Do not use in place of the model score.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(int(formula_prob))

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-head">Physicochemical Profile</div>', unsafe_allow_html=True)

            tc = "tier2" if is_t2 else "tier1"
            mw_cls = "mw-t2" if is_t2 else "mw-t1"

            st.markdown(f"""
            <div class="prop-grid">
                <div class="prop-cell {tc}">
                    <div class="prop-name">Mol. Weight</div>
                    <div class="prop-value {mw_cls}">{mw:.1f}<span class="prop-unit">Da</span></div>
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
                    <div class="prop-name">Rotatable Bonds</div>
                    <div class="prop-value">{n_rot}</div>
                </div>
                <div class="prop-cell {tc}">
                    <div class="prop-name">MW Tier</div>
                    <div class="prop-value" style="font-size:0.82rem;margin-top:3px;">{"bRo5" if is_t2 else "Ro5"}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-head">Lipinski Rule of Five Violations</div>', unsafe_allow_html=True)

            with st.expander(f"{len(ro5_v)} violation{'s' if len(ro5_v) != 1 else ''} detected — click to expand"):
                if ro5_v:
                    items = "".join(f'<div class="violation-item">✗ &nbsp;{v}</div>' for v in ro5_v)
                    st.markdown(items, unsafe_allow_html=True)
                else:
                    st.markdown('<div class="no-violation">✓ &nbsp;No Ro5 violations — all four thresholds satisfied</div>', unsafe_allow_html=True)
                st.markdown("""
                <div style="font-size:0.75rem;color:#9ca3af;margin-top:0.6rem;font-family:'JetBrains Mono',monospace;">
                Thresholds: MW ≤ 500 Da &nbsp;·&nbsp; LogP ≤ 5 &nbsp;·&nbsp; HBD ≤ 5 &nbsp;·&nbsp; HBA ≤ 10
                </div>
                """, unsafe_allow_html=True)

with tab_batch:
    st.markdown('<div class="input-label">Upload CSV</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.85rem;color:#6b7280;margin-bottom:1rem;line-height:1.6;">
    Upload a CSV file with a column named <code>smiles</code>. The tool will score every
    row and return a downloadable CSV with model score, formula score, physicochemical
    properties, tier classification, and Ro5 violation count.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        df_in = pd.read_csv(uploaded)

        if "smiles" not in df_in.columns:
            st.markdown("""
            <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:4px;
                        padding:0.9rem 1.2rem;font-size:0.85rem;color:#991b1b;">
                CSV must contain a column named <strong>smiles</strong>.
            </div>
            """, unsafe_allow_html=True)
        else:
            run_batch = st.button("Run Batch Analysis", key="batch_run")
            if run_batch:
                model, scaler, feature_order = load_assets()
                records = []
                progress = st.progress(0)
                status   = st.empty()
                total    = len(df_in)

                for i, row in df_in.iterrows():
                    smi = str(row["smiles"])
                    status.markdown(
                        f'<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;'
                        f'color:#6b7280;">Evaluating {i+1} / {total}</div>',
                        unsafe_allow_html=True
                    )
                    progress.progress(int((i + 1) / total * 100))
                    res = evaluate_single(smi, model, scaler, feature_order)

                    if res is None:
                        records.append({
                            "smiles":                  smi,
                            "model_score_%":           "INVALID",
                            "formula_score_%":         "INVALID",
                            "verdict_model":           "INVALID",
                            "tier":                    "INVALID",
                            "MW_Da":                   "",
                            "LogP":                    "",
                            "TPSA_A2":                 "",
                            "HB_donors":               "",
                            "HB_acceptors":            "",
                            "ring_count":              "",
                            "rotatable_bonds":         "",
                            "ro5_violation_count":     "",
                            "ro5_violations_detail":   "",
                        })
                    else:
                        records.append({
                            "smiles":                  smi,
                            "model_score_%":           f"{res['model_prob']:.2f}",
                            "formula_score_%":         f"{res['formula_prob']:.2f}",
                            "verdict_model":           "Drug-like" if res['model_prob'] >= 50 else "Non drug-like",
                            "tier":                    res["tier"],
                            "MW_Da":                   f"{res['mw']:.2f}",
                            "LogP":                    f"{res['logp']:.3f}",
                            "TPSA_A2":                 f"{res['tpsa']:.2f}",
                            "HB_donors":               res["hbd"],
                            "HB_acceptors":            res["hba"],
                            "ring_count":              res["rings"],
                            "rotatable_bonds":         res["n_rot"],
                            "ro5_violation_count":     len(res["ro5_v"]),
                            "ro5_violations_detail":   " | ".join(res["ro5_v"]) if res["ro5_v"] else "none",
                        })

                status.empty()
                progress.empty()

                df_out = pd.DataFrame(records)

                st.markdown('<div class="section-head">Batch Results</div>', unsafe_allow_html=True)
                st.dataframe(df_out, use_container_width=True, hide_index=True)

                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Results CSV",
                    data=csv_bytes,
                    file_name="druglikeness_results.csv",
                    mime="text/csv",
                )

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
    two tiers. Pearson R = 0.32 against full model; ~75% binary classification accuracy retained.
    The formula is provided for interpretability only — the deep model score is the authoritative prediction.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    st.markdown("**Tier 1 — Standard molecules (MW < 500 Da)**")
    st.latex(
        r"Z = -7.1577"
        r" + 2.6147\,(\text{BCUT2D\_CHGHI})"
        r" + 0.0544\,(\text{fr\_Ar\_N})"
        r" + 0.0208\,(\text{TPSA})"
        r" + 0.1122\,(\text{EState\_VSA2})"
        r" + 0.0602\,(\text{EState\_VSA10})"
        r" + 0.3898\,(\text{RingCount})"
        r" + 0.3284\,(\text{Chi1n})"
        r" + 0.0026\,(\text{BertzCT})"
        r" + 0.3701\,(\text{NumHAcceptors})"
        r" - 1.8925\,(\text{FractionCSP3})"
    )

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    st.markdown("**Tier 2 — bRo5 / Macrocycles (MW ≥ 500 Da)**")
    st.latex(
        r"Z = -17.2985"
        r" + 8.1213\,(\text{BCUT2D\_CHGHI})"
        r" + 0.1382\,(\text{fr\_Ar\_N})"
        r" + 0.0104\,(\text{TPSA})"
        r" + 0.0406\,(\text{EState\_VSA2})"
        r" - 0.0368\,(\text{EState\_VSA10})"
        r" + 0.4550\,(\text{NumArHetero})"
        r" - 0.9592\,(\text{fr\_guanido})"
        r" - 0.2622\,(\text{RingCount})"
        r" - 0.0095\,(\text{SMR\_VSA6})"
        r" - 0.1402\,(\text{Chi1n})"
    )

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.latex(r"P = \frac{1}{1 + e^{-Z}}")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-head">Descriptor Coefficients</div>', unsafe_allow_html=True)

    coef_df = pd.DataFrame({
        "Descriptor": [
            "BCUT2D_CHGHI", "fr_Ar_N", "TPSA", "EState_VSA2", "EState_VSA10",
            "RingCount", "Chi1n", "BertzCT", "NumHAcceptors", "FractionCSP3",
            "—",
            "SMR_VSA6", "NumArHetero", "fr_guanido",
        ],
        "Description": [
            "Highest generic charge — electrostatic distribution",
            "Number of aromatic nitrogens — hydrogen bonding capacity",
            "Topological polar surface area — membrane permeability",
            "Electrotopological state VSA (bin 2) — electron accessibility",
            "Electrotopological state VSA (bin 10) — steric bulk",
            "Total ring count — rigidity and complexity",
            "1st-order path connectivity index — structural branching",
            "Bertz complexity index — overall molecular complexity",
            "H-bond acceptor count — pharmacophoric anchor",
            "Fraction sp3 carbons — penalises inert saturated structures",
            "Tier 2 only",
            "Molar refractivity VSA (bin 6) — polarizability",
            "Number of aromatic heteroatoms",
            "Number of guanidine groups — high basicity",
        ],
        "Tier 1 coeff.": [
            "+2.6147", "+0.0544", "+0.0208", "+0.1122", "+0.0602",
            "+0.3898", "+0.3284", "+0.0026", "+0.3701", "−1.8925",
            "",
            "—", "—", "—",
        ],
        "Tier 2 coeff.": [
            "+8.1213", "+0.1382", "+0.0104", "+0.0406", "−0.0368",
            "−0.2622", "−0.1402", "—", "—", "—",
            "",
            "−0.0095", "+0.4550", "−0.9592",
        ],
    })

    st.dataframe(coef_df, use_container_width=True, hide_index=True)

st.markdown("""
<div style="margin-top:3rem;padding-top:1.2rem;border-top:1px solid #e5e7eb;
            display:flex;justify-content:space-between;align-items:center;
            font-size:0.75rem;color:#9ca3af;">
    <span style="font-family:'JetBrains Mono',monospace;letter-spacing:0.06em;">
        DrugLikenessModel
    </span>
    <span>Trained on ChEMBL 34 · 209,269 compounds</span>
</div>
""", unsafe_allow_html=True)
