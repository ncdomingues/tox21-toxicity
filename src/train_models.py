"""Train the final models used by the app: one random forest per assay on all molecules.

Each model uses fingerprint + descriptor features (the best setup in notebook 02).
Out-of-bag predictions give every training molecule a score from trees that never saw it;
the app uses them to say how a new molecule ranks against the training set.

Run:  python src/train_models.py   (a few minutes; writes models/tox21_models.joblib)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from featurize import ASSAYS

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODEL_FILE = ROOT / "models" / "tox21_models.joblib"


def load_features():
    desc = pd.read_parquet(PROC / "descriptors.parquet")
    fps = np.load(PROC / "fingerprints.npy")
    return fps, desc


def main():
    mols = pd.read_parquet(PROC / "molecules.parquet")
    fps, desc = load_features()
    medians = desc.median()
    X = np.hstack([fps, desc.fillna(medians).to_numpy()]).astype(np.float32)

    models, oob_scores = {}, {}
    for assay in ASSAYS:
        y = mols[assay].to_numpy()
        tested = ~np.isnan(y)
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                    oob_score=True, n_jobs=-1, random_state=0)
        rf.fit(X[tested], y[tested])
        models[assay] = rf
        oob_scores[assay] = np.sort(rf.oob_decision_function_[:, 1])
        print(f"  {assay:14s} trained on {tested.sum():,} molecules")

    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump({"models": models, "oob_scores": oob_scores, "descriptor_medians": medians,
                 "descriptor_names": list(desc.columns)}, MODEL_FILE, compress=3)
    print(f"Saved {MODEL_FILE.relative_to(ROOT)} ({MODEL_FILE.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
