import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


def multi_restart(model_factory, X_train, X_test, N=10):
    samples = []
    for n in range(N):
        model = model_factory()
        result = get_full_results(model, X_train, X_test)
        samples.append(result)

    return samples

def get_full_results(model, X_train, X_test, mnar=False):
    result = None
    if mnar:
        result = model.fit(X_train, mnar=True)
    else:
        result = model.fit(X_train)
    if X_test is not None : 
        result['test_z'] = model.predict(X_test)
        result['X_test_impute'] = model.posterior_predict(X_test)
    result['X_train_impute'] = model.posterior_predict(X_train)

    ll =  result['loglike']
    if isinstance(result['loglike'], list):
        ll = result['loglike'][-1]
    result['loglike'] = ll
    return result


def rmse(A,B):
    if len(B) == 0 or len(A) == 0: return 0.0
    return np.sqrt(mean_squared_error(A,B))


def evaluate_model(method_fn, X_missing_train, X_true_train, y_train,
                   X_missing_test, X_true_test, y_test, bernoulli=False):
    
    results = method_fn(X_missing_train, X_missing_test)
    if not isinstance(results, list):
        results = [results]

    all_metrics = []

    for result in results:
        metrics = {}

        # ARI
        if 'z' in result and y_train is not None:
            z_pred_train = result['z']
            metrics['train_ARI'] = adjusted_rand_score(y_train, z_pred_train)
        
        if 'test_z' in result and y_test is not None:
            z_pred_test = result['test_z']
            metrics['test_ARI'] = adjusted_rand_score(y_test, z_pred_test)

        # RMSE Impute 
        if 'X_train_impute' in result:
            missing_mask_train = np.isnan(X_missing_train)
            X_hat_train = result['X_train_impute'][missing_mask_train]
            X_true_train_vals = X_true_train[missing_mask_train]
            metrics['RMSE_train'] = rmse(X_hat_train, X_true_train_vals)

        if 'X_test_impute' in result:
            missing_mask_test = np.isnan(X_missing_test)
            X_hat_test = result['X_test_impute'][missing_mask_test]
            X_true_test_vals = X_true_test[missing_mask_test]
            metrics['RMSE_test'] = rmse(X_hat_test, X_true_test_vals)

        # Log-likelihood
        if 'loglike' in result:
            metrics['loglike'] = result['loglike']

        all_metrics.append(metrics)

    # Collect all keys across all metric dicts
    all_keys = set(k for m in all_metrics for k in m)

    summary = {
        k: np.mean([m[k] for m in all_metrics if k in m]) for k in all_keys
    }
    summary.update({
        k + '_std': np.std([m[k] for m in all_metrics if k in m]) for k in all_keys
    })

    return summary



def run_full_evaluation(dataset_train, dataset_test, methods, missing_rates, bernoulli=False):
    results = []

    for rate in missing_rates:
        X_missing_train, y_train = dataset_train.apply_missingness(missing_rate=rate)
        X_true_train = dataset_train.get_complete_data()

        X_missing_test, y_test = dataset_test.apply_missingness(missing_rate=rate)
        X_true_test = dataset_test.get_complete_data()

        N,D = X_missing_train.shape

        print("True Missing Percent:", np.sum(np.isnan(X_missing_train)) / (N * D))

        for name, method_fn in methods.items():
            print(name, rate, "Running Evaluation...")
            metrics = evaluate_model(method_fn, X_missing_train, X_true_train, y_train, 
                                     X_missing_test, X_true_test, y_test,
                                     bernoulli=bernoulli)
            metrics['method'] = name
            metrics['missing_rate'] = rate
            results.append(metrics)

    return pd.DataFrame(results)

def plot_ari_by_missingness_line(df, metric,title='ARI by Method and Missing Rate'):

    # Set style
    sns.set(style="whitegrid")

    # Create figure and axis
    plt.figure(figsize=(10, 6))

    # Unique methods
    methods = df['method'].unique()
    missing_rates = sorted(df['missing_rate'].unique())

    # Plot each method
    for method in methods:
        sub_df = df[df['method'] == method].sort_values(by='missing_rate')
        plt.errorbar(
            sub_df['missing_rate'],
            sub_df[metric] if metric in sub_df.columns else sub_df[metric],
            yerr=sub_df[f'{metric}_std'] if f'{metric}_std' in sub_df.columns else sub_df[f'{metric}_std'],
            label=method,
            capsize=4,
            marker='o',
            linestyle='-'
        )

    plt.xlabel('Missing Rate')
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

