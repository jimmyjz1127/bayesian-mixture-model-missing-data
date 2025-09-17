# Evaluation Script — Quick Usage Guide

This script runs all selected inference methods across a grid of missingness levels and saves the resulting metrics (used for traceplots) to CSV.

---

## What it does

* Loads pre-split data (`X_train`, `y_train`, `X_test`, `y_test`) from:

  ```
  ./../Datasets/{MODEL}/Processed/{NAME}/
  ```

  where `{MODEL} ∈ {Bernoulli, Gaussian}` and `{NAME}` is your dataset ID.
* Runs each method over missingness rates `[0.0, 0.1, ..., 0.9]`.
* Collects train/test metrics (e.g., ARI, log-likelihood, RMSE — whatever `run_full_evaluation` emits).
* Saves a single CSV per run to your chosen output directory.

---

## Required files (per dataset)

Inside `./../Datasets/{MODEL}/Processed/{NAME}/`:

* `X_train_{NAME}.npy`
* `y_train_{NAME}.npy`
* `X_test_{NAME}.npy`
* `y_test_{NAME}.npy`

> Shapes: `X_*` are `(N, D)` arrays, `y_*` are `(N,)` integer labels.  

> Example : `{MODEL}` = `Bernoulli`, `{NAME}` = `shapes`

---

## CLI

```
python eval.py --type {Gaussian|Bernoulli} --name DATASET_NAME -k K -d SAVE_DIR
```

**Arguments**

* `--type` (required): model family. Use `Bernoulli` for BMM, `Gaussian` for GMM.
* `--name` (required): dataset name (used in file paths and filenames) - e.g : `shapes`.
* `-k/--k` (required): number of mixture components K.
* `-d/--d` (required): output directory (CSV will be written here).

---

## Examples

### Bernoulli (BMM)

```bash
python eval.py --type Bernoulli --name shapes -k 3 -d ./metrics_out
# -> writes ./metrics_out/bmm_mp_voting.csv
```

### Gaussian (GMM)

```bash
python eval.py --type Gaussian --name digits5x5 -k 3 -d ./metrics_out
# -> writes ./metrics_out/gmm_blobs3.csv
```

---

## Output

A single CSV per run:

* `./{SAVE_DIR}/bmm_{NAME}.csv`  (for `--type Bernoulli`)
* `./{SAVE_DIR}/gmm_{NAME}.csv`  (for `--type Gaussian`)

**Columns** depend on `run_full_evaluation`, but typically include:

* `method`, `missing_rate`
* train/test metrics (e.g., `train_ARI`, `test_ARI`, `train_loglike`, `test_loglike`, `train_RMSE`, `test_RMSE`)
* optional dispersion columns (e.g., `*_std`) if computed over restarts/seeds

---

## Methods run (by family)

### Bernoulli

* `Gibbs+MNAR` (MNAR-aware Gibbs)
* `Gibbs` (standard Gibbs)
* `VBEM` (variational Bayes EM; multi-restart)
* `EM` (maximum likelihood EM; multi-restart)
* `EM+mean` (EM with mean imputation; multi-restart)
* `EM+CC` (EM on complete cases; multi-restart)
* `KMeans+CC`, `KMeans+mean` (baselines; multi-restart)

### Gaussian

* `Gibbs+MNAR`
* `Gibbs`
* `VBEM` (multi-restart)
* `EM` (multi-restart)
* `EM+CC`
* `EM+mean`
* `KMeans+CC`, `KMeans+mean`

> Methods commented out in the script are provided as references and can be enabled if needed.

---

## Notes & tips

* **Working directory:** The script uses relative paths; run it from the repo root (or adjust paths).
* **K (components):** Must match your data generation/assumptions.
* **Prior settings:** Priors come from `BMMPriorParameters` / `GMMPriorParameters` initialized on `X_train`.
* **Restarts:** Methods wrapped with `multi_restart(...)` will aggregate across random initializations.
* **Dependencies:** This script calls project classes/functions like `BMMGibbs`, `GMMVBEM`, `run_full_evaluation`, etc. Ensure they’re importable on `PYTHONPATH`.

---

NOTE : All Gaussian datasets take considerable time to run with (30+ minutes). The BMM `synthetic` dataset takes the least amount of time at roughly 15 minutes.