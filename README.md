# CS5099 · Bayesian Unsupervised Learning with Missing Data for Mixture Modelling

**Author:** James Zhang (190015412)
**Date:** August 2025

A research codebase for clustering with missing data using probabilistic mixture models. It implements **fully Bayesian** inference (Gibbs Sampling, VBEM) and baselines (EM, KMeans with/without imputation), evaluates robustness under increasing missingness, and includes an **MNAR** extension where missingness depends on latent clusters.

---

## 0. Highlights

* **Models:** Bernoulli Mixture Models (BMM) for binary features; Gaussian Mixture Models (GMM) for continuous features.
* **Inference:** Gibbs, VBEM, EM; KMeans baselines.
* **Missingness:** MCAR experiments (0–90% in 10% steps); exploratory MNAR extension.
* **Metrics:** ARI (clustering), log-likelihood (fit), RMSE (imputation).
* **Reproducibility:** Scripts + notebooks to regenerate metrics and figures.

---

## 1. Repository Structure

| Name                         | Description                                                                                  |
| ---------------------------- | -------------------------------------------------------------------------------------------- |
| `datasets/`                  | `Dataset` classes and MCAR simulation utilities.                                             |
| `Datasets/`                  | Real and synthetic datasets (CSV/NumPy) used for evaluation.                                 |
| `mnar_results/`              | MNAR vs MCAR comparison outputs for Gibbs on BMM/GMM datasets.                               |
| `models/`                    | Core model/inference classes (Gibbs, VBEM, EM, KMeans).                                      |
| `notebooks/`                 | Workbench notebooks (`Week1/` … `Week8/`), derivations, synthetic generation, preprocessing. |
| `Results/`                   | Aggregated CSVs used to produce traceplots in the report (by dataset).                       |
| `utils/`                     | Helpers: imputations, evaluation runners, plotting, etc.                                     |
| `Evaluate.py`                | Main script to run all methods and **collect evaluation metrics** (saves CSVs).              |
| `Evaluation.ipynb`           | Regenerates quantitative evaluation figures in the report.                                   |
| `ImputationEvaluation.ipynb` | Builds imputation visualizations.                                                            |
| `Documentation.pdf`          | Documentation for all of the model classes                                                   |

> For CLI details to collect metrics with `Evaluate.py`, see **`EvaluationInstruction.md`**.

---

## 2. Installation (Unix)

Requires Python 3.10+.

```bash
# from root directory
python -m venv .venv
source .venv/bin/activate         
pip install -r requirements.txt
```

---

## 3. Datasets

Expected layout for each dataset (after preprocessing):

```
./../Datasets/{MODEL}/Processed/{NAME}/
  X_train_{NAME}.npy
  y_train_{NAME}.npy
  X_test_{NAME}.npy
  y_test_{NAME}.npy
```

* `{MODEL}` is `Bernoulli` or `Gaussian`.
---

## 4. Typical Workflow

1. **Prep data:** use notebooks in `notebooks/` to generate `Processed/{NAME}` splits.
2. **Collect metrics:** run `Evaluate.py` per dataset/model family.
3. **Plot results:** use `Evaluation.ipynb` to draw traceplots and bar charts.
4. *(Optional)* **MNAR runs:** ensure MNAR configs are enabled (see `models/` and utilities).


