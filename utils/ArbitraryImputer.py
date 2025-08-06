import numpy as np 
from sklearn.impute import SimpleImputer

global_imputer = SimpleImputer()

def fallback_imputer(X, strategy='mean', bernoulli=False, fallback_value=0.5):
    X = np.array(X)
    all_nan_cols = np.isnan(X).all(axis=0)

    imputer = SimpleImputer(strategy=strategy)
    try:
        X_partial = imputer.fit_transform(X[:, ~all_nan_cols])
    except ValueError:
        X_partial = X[:, ~all_nan_cols] 

    fallback = np.full((X.shape[0], all_nan_cols.sum()), fallback_value)


    X_imputed = np.empty_like(X, dtype=float)
    X_imputed[:, ~all_nan_cols] = X_partial
    X_imputed[:, all_nan_cols] = fallback

    if bernoulli:
        X_imputed = np.round(X_imputed).astype(int)

    return X_imputed

def mean_impute(X, bernoulli=False):
    return fallback_imputer(X, strategy='mean', bernoulli=bernoulli)

def median_impute(X, bernoulli=False):
    return fallback_imputer(X, strategy='median', bernoulli=bernoulli)

def mode_impute(X, bernoulli=False):
    return fallback_imputer(X, strategy='most_frequent', bernoulli=bernoulli)