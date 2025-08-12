# CS5099 : Bayesian Unsupervised Learning with Missing Data for Mixture Modelling
#### Author : James Zhang (190015412)
#### Date : August 2025

## 1 Files and Directories
|Name|Description|
|---|---|
|`datasets/`|contains `Dataset` classes which are used for simulating MCAR missingness|
|`Datasets/`|contains the datasets used for evaluation such as realworld and synthetic datasets as csv files|
|`mnar_results/`|contains evaluation data from comparing MNAR gibbs model against MCAR gibbs model on MCAR datasets|
|`models/`|contains class definitions for all models implemented (Gibbs Sampling, VBEM, EM, KMeans)|
|`notebooks/`|Contains all notebooks used for workbenching. These are organized into directories ranging from `Week1/` to `Week8/` corresponding to the weeks of workbench development from the start to end of this research project. The notebooks contain detailed derivations in markdown for each approach used in this research. The directory also contains notebooks for synthetic data generation, as well as data-preprocessing notebooks for real-world datasets|
|`Results/`|Contains all evaluation results used  to produce traceplots in the report. These are organized by dataset, where each `csv` files contains performance metrics across all missingness percentages, for all inference approaches|
|`utils/`|Contains utility functions, notably imputation functions and methods used for collecting evaluation metrics|
|`Evaluate.py`|Main script for running routine to collect evaluataion metrics for all algorithms|
|`Evaluation.ipynb`|Notebook containing code used to produce all quantitative evaluation results presented in the main report|
|`ImputationEvaluation.ipynb`|Notebook used to produce visual imputation figures in the report.|

See `EvaluationInstruction.md` for instructions on how to collect run algorithms to colelction evaluation metrics using the `Evaluate.py` script