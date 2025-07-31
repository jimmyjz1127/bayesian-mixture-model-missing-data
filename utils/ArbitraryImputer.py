import numpy as np 
from sklearn.impute import SimpleImputer

global_imputer = SimpleImputer()

def mean_impute(X, bernoulli=False):
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    if bernoulli:
        X_imputed = np.round(X_imputed).astype(int)
    return X_imputed

def median_impute(X, bernoulli=False):
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    if bernoulli:
        X_imputed = np.round(X_imputed).astype(int)

    return X_imputed

def mode_impute(X, bernoulli=False):
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = imputer.fit_transform(X)
    if bernoulli:
        X_imputed = np.round(X_imputed).astype(int)
    return X_imputed