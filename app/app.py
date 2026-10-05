"""Toxicity profiler: paste a molecule, get its predicted Tox21 profile.

Run from the project folder (after featurize.py and train_models.py):
    streamlit run app/app.py
"""
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from featurize import ASSAYS, descriptors, fingerprint, parse  # noqa: E402
from train_models import MODEL_FILE  # noqa: E402

st.set_page_config(page_title="Tox21 Toxicity Profiler", page_icon="🧪", layout="wide")

ASSAY_INFO = {
    "NR-AR": "Androgen receptor", "NR-AR-LBD": "Androgen receptor (binding domain)",
    "NR-AhR": "Aryl hydrocarbon receptor (dioxin-like)", "NR-Aromatase": "Aromatase inhibition",
    "NR-ER": "Oestrogen receptor", "NR-ER-LBD": "Oestrogen receptor (binding domain)",
    "NR-PPAR-gamma": "PPAR-γ (metabolism)", "SR-ARE": "Oxidative stress", "SR-ATAD5": "DNA damage (ATAD5)",
    "SR-HSE": "Heat-shock response", "SR-MMP": "Mitochondrial damage", "SR-p53": "DNA damage (p53)",
}
EXAMPLES = {
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Bisphenol A": "CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O",
    "Benzo[a]pyrene": "C1=CC=C2C3=C4C(=CC2=C1)C=CC5=C4C(=CC=C5)C=C3",
    "Triclosan": "C1=CC(=C(C=C1Cl)O)OC2=C(C=C(C=C2)Cl)Cl",
    "Tamoxifen": "CCC(=C(C1=CC=CC=C1)C2=CC=C(C=C2)OCCN(C)C)C3=CC=CC=C3",
}
FLAG_PERCENTILE = 90       # flag molecules ranked in the riskiest 10% of the training set
SIMILARITY_GOOD, SIMILARITY_LOW = 0.50, 0.35


@st.cache_resource(show_spinner="Loading models ...")
def load():
    if not MODEL_FILE.exists():
        st.error("Models not found. Run `python src/featurize.py` and `python src/train_models.py` first.")
        st.stop()
    bundle = joblib.load(MODEL_FILE)
    mols = pd.read_parquet(ROOT / "data" / "processed" / "molecules.parquet")
    fps = np.load(ROOT / "data" / "processed" / "fingerprints.npy")
    bitvectors = [DataStructs.CreateFromBitString("".join(map(str, row))) for row in fps]
    return bundle, mols, bitvectors


def featurise(mol, bundle):
    d = pd.Series(descriptors(mol))[bundle["descriptor_names"]]
    d = d.replace([np.inf, -np.inf], np.nan).clip(-1e6, 1e6).fillna(bundle["descriptor_medians"])
    return np.hstack([fingerprint(mol), d.to_numpy()]).astype(np.float32).reshape(1, -1)


bundle, mols, bitvectors = load()

st.title("🧪 Tox21 Toxicity Profiler")
st.caption("Predicts how a molecule would behave in 12 Tox21 toxicity assays, using random forests trained on "
           "~7,800 tested chemicals. Educational demo, not a safety assessment.")

left, right = st.columns([1, 2], gap="large")
with left:
    example = st.selectbox("Start from an example", list(EXAMPLES), index=3)
    smiles = st.text_input("…or paste a SMILES string", EXAMPLES[example])
    mol = parse(smiles) if smiles else None
    if mol is None:
        st.error("This SMILES string could not be read. Check it for typos.")
        st.stop()
    st.image(Draw.MolToImage(mol, size=(360, 280)), caption=Chem.MolToSmiles(mol))

    # Applicability domain: similarity to the nearest training molecule
    query_bits = DataStructs.CreateFromBitString("".join(map(str, fingerprint(mol))))
    sims = np.array(DataStructs.BulkTanimotoSimilarity(query_bits, bitvectors))
    nn = int(sims.argmax())
    sim = sims[nn]
    if sim == 1.0:
        st.info("This exact molecule was **tested in Tox21** and is part of the training data, so its real lab "
                "results are shown below. The predictions show what the model learned from it.")
    elif sim >= SIMILARITY_GOOD:
        st.success(f"**Familiar chemistry** (similarity {sim:.2f} to the nearest training molecule). Predictions are most reliable here.")
    elif sim >= SIMILARITY_LOW:
        st.warning(f"**Partly familiar** (similarity {sim:.2f}). Treat predictions with some caution.")
    else:
        st.error(f"**Unfamiliar chemistry** (similarity {sim:.2f}). The model has seen little like this; predictions are unreliable.")

with right:
    X = featurise(mol, bundle)
    rows = []
    for assay in ASSAYS:
        score = bundle["models"][assay].predict_proba(X)[0, 1]
        pct = np.searchsorted(bundle["oob_scores"][assay], score) / len(bundle["oob_scores"][assay]) * 100
        rows.append({"assay": assay, "description": ASSAY_INFO[assay], "percentile": pct,
                     "flag": "High" if pct >= FLAG_PERCENTILE else ("Medium" if pct >= 75 else "Low")})
    res = pd.DataFrame(rows)

    n_high = (res["flag"] == "High").sum()
    st.subheader(f"{n_high} of 12 assays flagged" if n_high else "No assays flagged")
    st.write("Each bar shows how this molecule ranks against the ~7,800 training chemicals for that assay. "
             f"A molecule in the riskiest {100 - FLAG_PERCENTILE}% is **flagged**.")
    chart = alt.Chart(res).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("description:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("percentile:Q", scale=alt.Scale(domain=[0, 100]), title="Risk rank vs training chemicals (percentile)"),
        color=alt.Color("flag:N", scale=alt.Scale(domain=["Low", "Medium", "High"],
                                                  range=["#b9bcc6", "#eda100", "#eb6834"]), title="Risk"),
        tooltip=["assay", "description", alt.Tooltip("percentile:Q", format=".0f"), "flag"],
    ).properties(height=380)
    rule = alt.Chart(pd.DataFrame({"x": [FLAG_PERCENTILE]})).mark_rule(strokeDash=[4, 4], color="#7a8091").encode(x="x:Q")
    st.altair_chart(chart + rule, use_container_width=True)

    st.markdown("**Nearest molecule in the training data** and its real lab results")
    nn_row = mols.iloc[nn]
    c1, c2 = st.columns([1, 2])
    c1.image(Draw.MolToImage(Chem.MolFromSmiles(nn_row["smiles_clean"]), size=(240, 180)),
             caption=f"{nn_row['mol_id']} · similarity {sim:.2f}")
    known = nn_row[ASSAYS].map({1.0: "active", 0.0: "inactive"}).fillna("not tested")
    c2.dataframe(pd.DataFrame({"assay": [ASSAY_INFO[a] for a in ASSAYS], "lab result": known.to_numpy()}),
                 hide_index=True, use_container_width=True, height=300)
