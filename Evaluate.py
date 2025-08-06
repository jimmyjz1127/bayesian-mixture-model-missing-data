import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
from sklearn.metrics import adjusted_rand_score
import argparse


from models.PriorParameters import GMMPriorParameters
from models.PriorParameters import BMMPriorParameters
from models.GMMGibbs import GMMGibbs
from models.BMMGibbs import BMMGibbs
from models.GMMVBEM import GMMVBEM
from models.BMMVBEM import BMMVBEM
from models.BMMEM import BMMEM
from models.GMMEM import GMMEM

from utils.ArbitraryImputer import mean_impute, median_impute, mode_impute
from utils.EvaluationUtilities import rmse, multi_restart, run_full_evaluation, get_full_results

from datasets.Dataset import Dataset

def eval(name,  model, K):
    dir_path = f"./../Datasets/{model}/Processed/{name}/"
    X_train = np.load(dir_path + f"X_train_{name}.npy")
    y_train = np.load(dir_path + f"y_train_{name}.npy")
    X_test = np.load(dir_path + f"X_test_{name}.npy")
    y_test = np.load(dir_path + f"y_test_{name}.npy")

    dataset_train = Dataset(X_train, y_train)
    dataset_test = Dataset(X_test, y_test)

    if model == "Bernoulli":
        params = BMMPriorParameters(X_train, K)
        bmm_methods = {
            'Gibbs'         : lambda X_train, X_test, X_train_true: [get_full_results(BMMGibbs(params), X_train, X_test, X_train_true)],  
            'VBEM'          : lambda X_train, X_test, X_train_true: multi_restart(lambda : BMMVBEM(params), X_train, X_test, X_train_true),
            'EM'            : lambda X_train, X_test, X_train_true: multi_restart(lambda :  BMMEM(K), X_train, X_test, X_train_true),
            'EM+mean'       : lambda X_train, X_test, X_train_true: multi_restart(lambda : BMMEM(K), mean_impute(X_train, True), mean_impute(X_test, True), X_train_true),
            # 'Gibbs+mean'    : lambda X_train, X_test: [get_full_results(BMMGibbs(params), mean_impute(X_train), X_test)],
            # 'VBEM+mean'     : lambda X_train, X_test: multi_restart(lambda : BMMVBEM(params), mean_impute(X_train), X_test),
            'EM+median'     : lambda X_train, X_test, X_train_true: multi_restart(lambda : BMMEM(K), median_impute(X_train, True), median_impute(X_test, True), X_train_true),
            # 'Gibbs+median'  : lambda X_train, X_test: [get_full_results(BMMGibbs(params), median_impute(X_train), X_test)],
            # 'VBEM+median'   : lambda X_train, X_test: multi_restart(lambda : BMMVBEM(params), median_impute(X_train), X_test),
            'EM+mode'       : lambda X_train, X_test, X_train_true: multi_restart(lambda : BMMEM(K), mode_impute(X_train,True), mode_impute(X_test,True), X_train_true),
            # 'Gibbs+mode'    : lambda X_train, X_test: [get_full_results(BMMGibbs(params), mode_impute(X_train), X_test)],
            # 'VBEM+mode'     : lambda X_train, X_test: multi_restart(lambda : BMMVBEM(params), mode_impute(X_train), X_test),
        }

        metrics_df = run_full_evaluation(
            dataset_train, dataset_test, bmm_methods, missing_rates=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], bernoulli=True
        )
        metrics_df.to_csv(f"./loglikes/bmm_{name}.csv")
        
    elif model=="Gaussian":
        params = GMMPriorParameters(X_train, K)
        gmm_methods = {
            'Gibbs'         : lambda X_train, X_test, X_train_true: [get_full_results(GMMGibbs(params), X_train, X_test, X_train_true)],  
            'VBEM'          : lambda X_train, X_test, X_train_true: multi_restart(lambda : GMMVBEM(params), X_train, X_test, X_train_true),
            'EM'            : lambda X_train, X_test, X_train_true: multi_restart(lambda :  GMMEM(K), X_train, X_test, X_train_true),
            'EM+mean'       : lambda X_train, X_test, X_train_true: multi_restart(lambda : GMMEM(K), mean_impute(X_train), mean_impute(X_test), X_train_true),
            # 'Gibbs+mean'    : lambda X_train, X_test: [get_full_results(GMMGibbs(params), mean_impute(X_train), X_test)],
            # 'VBEM+mean'     : lambda X_train, X_test: multi_restart(lambda : GMMVBEM(params), mean_impute(X_train), X_test),
            'EM+median'     : lambda X_train, X_test, X_train_true: multi_restart(lambda : GMMEM(K), median_impute(X_train), median_impute(X_test), X_train_true),
            # 'Gibbs+median'  : lambda X_train, X_test: [get_full_results(GMMGibbs(params), median_impute(X_train), X_test)],
            # 'VBEM+median'   : lambda X_train, X_test: multi_restart(lambda : GMMVBEM(params), median_impute(X_train), X_test),
            'EM+mode'       : lambda X_train, X_test, X_train_true: multi_restart(lambda : GMMEM(K), mode_impute(X_train), mode_impute(X_test), X_train_true),
            # 'Gibbs+mode'    : lambda X_train, X_test: [get_full_results(GMMGibbs(params), mode_impute(X_train), X_test)],
            # 'VBEM+mode'     : lambda X_train, X_test: multi_restart(lambda : GMMVBEM(params), mode_impute(X_train), X_test),
        }

        metrics_df = run_full_evaluation(
            dataset_train, dataset_test, gmm_methods, missing_rates=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], bernoulli=False
        )
        metrics_df.to_csv(f"./loglikes/gmm_{name}.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True, choices=['Gaussian', 'Bernoulli'])
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('-k', '--k', type=int, required=True)
    args = parser.parse_args()

    eval(args.name, args.type, args.k)

if __name__ == "__main__":
    main()
