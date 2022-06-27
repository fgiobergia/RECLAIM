import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from utils import train_test, do_runs

def read_adult():
    column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]

    ds_train_path = os.path.join("datasets", "adult.data")
    df_train = pd.read_csv(ds_train_path, header=None, skipinitialspace=True, names=column_names)
    ds_test_path = os.path.join("datasets", "adult.test")
    df_test = pd.read_csv(ds_test_path, header=None, names=column_names, skiprows=1)
    df_test["class"] = df_test["class"].map(lambda x: x[:-1])

    df = pd.concat([df_train, df_test])

    y = df["class"].values
    y = (y == ">50K").astype(int)
    df.drop(columns=["class"], inplace=True)
    
    df_no = df.select_dtypes(exclude=object)
    df_strip = pd.get_dummies(df.select_dtypes(include=object))

    df = pd.concat([df_no, df_strip], axis=1)

    return df.values, y

def trainer(ds, random_state):
    X, y = ds
    clf = RandomForestClassifier(100, random_state=42, n_jobs=-1)
    return train_test(X, y, .2, clf, random_state=random_state)

if __name__ == "__main__":
    n_runs = 10
    X, y = read_adult()
    do_runs((X,y), trainer, n_runs)