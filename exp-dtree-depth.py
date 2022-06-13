import matplotlib.pyplot as plt
import numpy as np
from utils import make_linear, train_test
import os

if __name__ == "__main__":
    n_pts = 25000
    repeats = 10
    test_size = .2
    metrics_order = ["A", "P", "R", "Fb"]
    outdir = "plots"
    
    max_depths = list(range(1, 31))
    # keeping track of all scores
    scores = np.zeros((len(metrics_order), repeats, len(max_depths)))
    for i in range(repeats):
        X, y = make_linear(n_pts=n_pts, eps=.25, random_state=42*i)
        for md_pos, m_d in enumerate(max_depths):
            C, N_P, metrics = train_test(X, y, test_size, m_d)
            scores[:, i, md_pos] = [ metrics[v] for v in metrics_order ]

    fig, ax = plt.subplots(figsize=(6,4))
    a_ndx = metrics_order.index("A")
    ax.errorbar(max_depths, scores[a_ndx].mean(axis=0), scores[a_ndx].std(axis=0), capsize=2, color='b')
    ax.set_ylim(0,1)
    ax.set_xlabel("maximum tree depth")
    ax.set_ylabel("accuracy")
    ax.grid()
    outfile = os.path.join(outdir, "depth-vs-accuracy.pdf")
    fig.savefig(outfile, bbox_inches="tight")