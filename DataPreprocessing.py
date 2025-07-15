import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from argparse import Namespace
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config

import sys
import argparse

def gaussian_pipeline(X_train, X_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    X_train_prepared = pipeline.fit_transform(X_train)
    X_test_prepared = pipeline.transform(X_test)

    return X_train_prepared, X_test_prepared

def pipeline(args):
    df = pd.read_csv(args.input)

    y = pd.Categorical(df[args.yname]).codes
    X = df.drop(columns=args.yname)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    if args.type == "gaussian":
        X_train_prepared, X_test_prepared = gaussian_pipeline(X_train, X_test)
    else:
        pass

    prefix = args.output + "/" + args.name + "/"

    np.save(prefix + "X_train_" + args.name + ".npy", X_train_prepared)
    np.save(prefix + "X_test_" + args.name + ".npy", X_test_prepared)
    np.save(prefix + "y_train_" + args.name + ".npy", y_train)
    np.save(prefix + "y_test_" + args.name + ".npy", y_test)

    print("Processed data files saved to directory : ", args.output)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, required=True,
                        choices=['gaussian', 'bernoulli'],
                        help='Dataset type: "bernoulli" or "gaussian" (required)')

    parser.add_argument('-i', '--input', type=str, required=True,
                    help='Path to input file (default: output.csv)')

    parser.add_argument('-o', '--output', type=str, default='output',
                    help='Path to output directory (default: output)')
    
    parser.add_argument('-n', '--name', type=str, default='dataset',
                    help='Dataset output name prefix (Example: "bird_species")')
    
    parser.add_argument('-y', '--yname', type=str, required=True,
                    help='Dataset label column name (Example: "class")')
    
    args = parser.parse_args()

    pipeline(args)
    
    
    

if __name__ == "__main__":
    main()