# Experiment to generate the plots shown in the paper
from sklearn.datasets import make_classification
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.getcwd())
from reclaim import find_confusion_matrix
from reclaim.utils import make_linear, train_test, rebuild_metric_range

def gen_performance(variable, ds_type="classification", repeats=10, rounding=None): 
    test_size = .2
    
    defaults = {
        "ds_size": [ 25000 ],
        "frac": [ 0.5 ], 
        "m_d": [ 6 ]
    }
    # Some parameters (e.g. ranges of values) to be
    # used for the studies. Based on `variable`, one
    # of the 3 factors is allowed to change, the others
    # are kept fixed
    if variable == "C":
        min_test_size = 100
        max_test_size = 50000
        ds_size_list = list(map(int, np.logspace(np.log10(min_test_size/test_size), np.log10(max_test_size/test_size), 15)))
        frac_list = defaults["frac"]
        m_d_list = defaults["m_d"]
    elif variable == "N_P":
        ds_size_list = defaults["ds_size"]
        frac_list = np.linspace(0.01, 0.99, 15)
        m_d_list = defaults["m_d"]
    elif variable == "m_d":
        ds_size_list = defaults["ds_size"]
        frac_list = defaults["frac"]
        m_d_list = np.arange(1, 31, 2)
    
    # this dictionary contains, for each combination
    # of factors, the full description of all metrics
    info = {
        ds_size: { # define dataset size used
            frac: { # define fractiokn of positive samples used
                m_d: { # define depth of decision tree (model capacity, which varies the performance)
                    x: [] for x in ["A","P","R","Fb"] # define the metric that is used for the reconstruction (along with C and N_P)
                } for m_d in m_d_list
            } for frac in frac_list
        } for ds_size in ds_size_list
    }

    for ds_size in ds_size_list:
        for frac in frac_list:
            for m_d in m_d_list:
                with tqdm(range(repeats)) as bar:
                    for i in bar:
                        if ds_type == "classification":
                            X, y = make_classification(ds_size, random_state=42, weights=(1-frac,))
                        elif ds_type == "linear":
                            X, y = make_linear(n_pts=ds_size, random_state=42)
                        
                        clf = DecisionTreeClassifier(max_depth=m_d, random_state=42)
                        # 42*i to obtain `repeats` different splits
                        C, N_P, metrics = train_test(X, y, test_size, clf, random_state=42*i)
                        
                        for m in ["A", "Fb", "P", "R"]:
                            val = metrics[m]
                            if rounding is not None:
                                val1 = round(val, rounding) - 5 * 10 ** -(rounding + 1)
                                val2 = round(val, rounding) + 5 * 10 ** -(rounding + 1)
                                val = (val1, val2)

                            vmin, vmax = find_confusion_matrix(C=C, N_P=N_P, **{m: val})
                            info[ds_size][frac][m_d][m].append((vmin, vmax, C, N_P, metrics))
    return info

def plot_performance(info, variable, outdir):
    # fig, ax = plt.subplots(2, 2, figsize=(20,15))
    fig, ax = plt.subplots(1, 4, figsize=(23,4))

    metric_names = {
        "A": "accuracy",
        "P": "precision",
        "R": "recall", 
        "Fb": "$F_1$ score"
    }

    for i, computed_metric in enumerate(["A", "P", "R", "Fb"]):

        x = { m: [] for m in ["A", "P", "R", "Fb"] }
        y = { m: [] for m in ["A", "P", "R", "Fb"] }
        z = { m: [] for m in ["A", "P", "R", "Fb"] }

        if variable == "C":
            frac = 0.5
            m_d = 6
            iter_var = info
        elif variable == "N_P":
            m_d = 6
            ds_size = 25000
            iter_var = info[ds_size]
        elif variable == "m_d":
            frac = 0.5
            ds_size = 25000
            iter_var = info[ds_size][frac]
        
        for metr in ["A", "P", "R", "Fb"]:
            for var in iter_var:
                if variable == "C":
                    ds_size = var
                elif variable == "N_P":
                    frac = var
                elif variable == "m_d":
                    m_d = var
                lst = []
                for cm1, cm2, C, N_P, met in info[ds_size][frac][m_d][metr]:  
                    rng = rebuild_metric_range(cm1, cm2, computed_metric, met[computed_metric])
                    lst.append(rng)
                x[metr].append(var)
                y[metr].append(np.mean(lst))
                z[metr].append(np.std(lst))

        colors = {
            "A": "r",
            "P": "y",
            "R": "b",
            "Fb": "g"
        }
        markers = {
            "A": "s",
            "P": "v",
            "R": "x",
            "Fb": "o"
        }
        for j, metr in enumerate(["P", "A", "R", "Fb"]):
            if metr == computed_metric:
                continue
            ax[i].errorbar(x[metr],y[metr],z[metr], label=metr, capsize=3, color=colors[metr], marker=markers[metr], zorder=j)
            if variable == "C":
                ax[i].set_xscale('log', base=10)
        ax[i].grid()
        ax[i].set_title(f"Reconstructing {metric_names[computed_metric]}")
        ax[i].set_ylim([0, 1])
        if i == 0:
            ax[i].set_ylabel("reconstructed range width")
        ax[i].set_xlabel("dataset size" if variable == "C" else "fraction of positive samples" if variable == "N_P" else "maximum tree depth" if variable == "m_d" else "!!!")
        if variable == "N_P":
            ax[i].set_xlim([0, 1])
        
    fig.legend(["Precision", "Recall", "$F_1$ score", "Accuracy"], loc="center right")

    fname = {
        "C": "ds-size-plots.pdf",
        "N_P": "n-p-plots.pdf",
        "m_d": "m-d-plots.pdf"
    }
    fig.savefig(os.path.join(outdir, fname[variable]), bbox_inches="tight")

if __name__ == "__main__":
    outdir = "plots"
    plt.rcParams.update({'font.size': 15})
    
    # generate plots for the 3 experiments
    # (fixing C, N_P and m_d respectively)
    for variable in ["m_d", "N_P", "C"]:
        ds_type = "linear" if variable == "m_d" else "classification"
        info = gen_performance(variable, ds_type, 4)
        plot_performance(info, variable, outdir)