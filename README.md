**English** | [Português](README.pt.md)

# Predicting Chemical Toxicity from Molecular Structure (Tox21)

Can we tell whether a molecule is toxic just from its structure? This project builds and honestly evaluates machine-learning models that predict the results of **12 toxicity assays** for about **7,800 chemicals**. It also includes an app that profiles any molecule you paste in.

**Data:** [Tox21](https://tripod.nih.gov/tox21/challenge/) (US EPA, NIH, FDA) via [MoleculeNet](https://moleculenet.org/). It covers 7 nuclear-receptor assays (hormone disruption) and 5 stress-response assays (DNA damage, oxidative stress, mitochondrial damage). Each result is active or inactive. Classes are imbalanced (3–16% active) and some labels are missing.

**Plain-language summary:** [open the live dashboard](https://ncdomingues.github.io/tox21-toxicity/toxicity-at-a-glance.html) ([source](toxicity-at-a-glance.html)), a one-page overview for non-technical readers in English and Portuguese.

## Key findings
1. **Random splits flatter the model.** Under a *scaffold split*, where test molecules have ring systems never seen in training, mean ROC-AUC is **0.79**, compared with **0.83** under a random split. The scaffold split is the honest estimate for new chemistry.

   ![Random vs scaffold split](figures/08_random_vs_scaffold.png)

2. **Whole-molecule descriptors beat fingerprints** (0.78 vs 0.72). Combining both with a random forest was best (0.79). Gradient boosting and logistic regression did worse.
3. **Trust depends on similarity.** ROC-AUC is **0.86** for test molecules similar to the training data and **0.72** for the least familiar third, so every prediction should come with a similarity score.

   ![Applicability domain](figures/12_applicability_domain.png)

4. **The model learns real chemistry.**
   - Activity at the "dioxin receptor" (AhR) rises from 1% to 33% with the number of aromatic rings.
   - The top fragments for mitochondrial toxicity are lipophilic phenols such as pentachlorophenol, which are classic uncouplers.
5. **It is useful for prioritising compounds.** Testing only the top 10% of compounds by model rank finds actives at **4x** the random rate and recovers about **40%** of all toxic compounds.

## Toxicity profiler app
Paste a SMILES string, or pick an example such as bisphenol A, triclosan or benzo[a]pyrene. The app shows:
- the molecule;
- its predicted risk rank in each of the 12 assays;
- a reliability warning based on similarity to the training data;
- the nearest tested molecule, with its real lab results.

```bash
streamlit run app/app.py
```

## Project structure
```
src/featurize.py        downloads Tox21; cleans molecules; computes Morgan fingerprints, RDKit descriptors, scaffolds and splits
src/train_models.py     trains the final per-assay random forests used by the app
src/style.py            shared chart style
notebooks/01_explore_tox21.ipynb      assays, imbalance, assay correlations, structure–activity patterns, scaffolds
notebooks/02_predict_toxicity.ipynb   random vs scaffold split, model comparison, applicability domain, toxic fragments, enrichment
app/app.py              Streamlit toxicity profiler
figures/                exported charts
toxicity-at-a-glance.html   plain-language dashboard
```

## How to run
```bash
pip install -r requirements.txt
python src/featurize.py        # ~1 minute
python src/train_models.py     # a few minutes, only needed for the app
jupyter notebook notebooks/
```

## Limitations
- High-throughput assay results are noisy, and "inactive" in a screen is not proof of safety.
- 2D features ignore 3D shape, which matters for receptor binding.
- Next steps: graph neural networks, multi-task learning across related assays, and conformal prediction for per-molecule confidence.

*Educational project. The predictions are not a safety assessment.*
