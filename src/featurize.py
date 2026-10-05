"""Download Tox21 and turn every molecule into model-ready features.

Outputs in data/processed/:
- molecules.parquet    one row per valid molecule: id, SMILES, scaffold, the 12 assay labels,
                       and which split (random / scaffold) it belongs to
- fingerprints.npy     Morgan fingerprints (radius 2, 2048 bits), uint8
- descriptors.parquet  ~200 RDKit 2D descriptors (molecular weight, logP, ring counts, ...)

Run once:  python src/featurize.py   (about one minute)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"

ASSAYS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
          "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
FP_BITS, FP_RADIUS = 2048, 2

_largest = rdMolStandardize.LargestFragmentChooser()
fingerprinter = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


def parse(smiles):
    """SMILES -> RDKit molecule, keeping only the largest fragment (drops salts and counter-ions)."""
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else _largest.choose(mol)


def fingerprint(mol):
    return fingerprinter.GetFingerprintAsNumPy(mol).astype(np.uint8)


def descriptors(mol):
    return {name: fn(mol) for name, fn in Descriptors.descList}


def scaffold(mol):
    """Bemis–Murcko scaffold: the ring systems and linkers, side chains removed. '' for acyclic molecules."""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(scaffolds, frac_train=0.8, frac_valid=0.1):
    """Deterministic scaffold split (as in MoleculeNet): the largest scaffold groups go to training,
    so validation and test contain only scaffolds the model has never seen."""
    groups = sorted(scaffolds.groupby(scaffolds).groups.values(), key=lambda idx: (len(idx), idx[0]), reverse=True)
    n = len(scaffolds)
    split = pd.Series("test", index=scaffolds.index)
    n_train = n_valid = 0
    for idx in groups:
        if n_train + len(idx) <= frac_train * n:
            split[idx] = "train"
            n_train += len(idx)
        elif n_valid + len(idx) <= frac_valid * n:
            split[idx] = "valid"
            n_valid += len(idx)
    return split


def random_split(n, seed=0, frac_train=0.8, frac_valid=0.1):
    order = np.random.RandomState(seed).permutation(n)
    split = np.array(["test"] * n, dtype=object)
    split[order[: int(frac_train * n)]] = "train"
    split[order[int(frac_train * n): int((frac_train + frac_valid) * n)]] = "valid"
    return split


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    raw_file = RAW / "tox21.csv.gz"
    if not raw_file.exists():
        print("Downloading Tox21 ...")
        resp = requests.get(URL, timeout=60)
        resp.raise_for_status()
        raw_file.write_bytes(resp.content)

    df = pd.read_csv(raw_file)
    mols = [parse(s) for s in df["smiles"]]
    valid = np.array([m is not None for m in mols])
    print(f"{len(df)} molecules, {(~valid).sum()} could not be parsed and are dropped")
    df = df[valid].reset_index(drop=True)
    mols = [m for m in mols if m is not None]

    df["smiles_clean"] = [Chem.MolToSmiles(m) for m in mols]
    df["scaffold"] = [scaffold(m) for m in mols]
    df["split_scaffold"] = scaffold_split(df["scaffold"])
    df["split_random"] = random_split(len(df))
    df[["mol_id", "smiles", "smiles_clean", "scaffold", "split_random", "split_scaffold"] + ASSAYS] \
        .to_parquet(OUT / "molecules.parquet", index=False)

    print("Computing fingerprints ...")
    np.save(OUT / "fingerprints.npy", np.vstack([fingerprint(m) for m in mols]))

    print("Computing descriptors (about a minute) ...")
    desc = pd.DataFrame([descriptors(m) for m in mols])
    desc = desc.replace([np.inf, -np.inf], np.nan).clip(-1e6, 1e6)
    desc.to_parquet(OUT / "descriptors.parquet", index=False)

    print("Split sizes (scaffold):", df["split_scaffold"].value_counts().to_dict())
    print("Done.")


if __name__ == "__main__":
    main()
