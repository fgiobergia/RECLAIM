import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from ip import solve_ip_problem
from utils import rebuild_metric_range, train_test

from prettytable import PrettyTable

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

def rounding_reconstruction(C, N_P, metrics, metrics_order):
    roundings = [4, 32]
    recs = np.zeros((len(metrics), len(metrics), len(roundings)))
    for j, m in enumerate(metrics_order):
        val = metrics[m]
        for p, i in enumerate(roundings):
            val1 = round(val, i) - 5 * 10 ** -(i + 1)
            val2 = round(val, i) + 5 * 10 ** -(i + 1)
            cm1, cm2 = solve_ip_problem(C=C, N_P=N_P, **{m: (val1, val2)})
            for k, m_r in enumerate(metrics_order):
                recs[j, k, p] = rebuild_metric_range(cm1, cm2, m_r, metrics[m_r])
    return recs

if __name__ == "__main__":
    metrics_order = ["A","P","R","Fb"]
    metric_names = {
        "A": "Accuracy",
        "P": "Precision", 
        "R": "Recall",
        "Fb": "F_1 score"
    }
    X, y = read_adult()
    

    repeats = 10
    reconstructions = []
    for i in range(repeats):
        clf = RandomForestClassifier(100, random_state=42, n_jobs=-1)
        C, N_P, metrics = train_test(X, y, .2, clf, random_state=42*i)

        reconstructions.append(rounding_reconstruction(C, N_P, metrics, metrics_order))
    rec = np.array(reconstructions)

    rec_mean = rec.mean(axis=0)
    rec_std = rec.std(axis=0)

    table = PrettyTable()

    table.field_names = [""] + [ metric_names[m] for m in metrics_order ]

    for i, m in enumerate(metrics_order):
        row = [metric_names[m]]
        for j, mr in enumerate(metrics_order):
            row.append(f"{round(rec_mean[i, j, 1],4)} +/- {round(rec_std[i,j,1],4)}")
        table.add_row(row)
        row = [f"{metric_names[m]} (rounded)"]
        for j, mr in enumerate(metrics_order):
            row.append(f"{round(rec_mean[i, j, 0],4)} +/- {round(rec_std[i,j,0],4)}")
        table.add_row(row)
    print(table)