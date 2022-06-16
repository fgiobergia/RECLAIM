from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import warnings


def get_coef_intercept(value, metric, beta=1):
    """Return coefficients and intercept for a given metric.

    Parameters:
        value (float): actual value of the metric
        metric (str): name of the metric passed
        beta (int): if metric="Fb", this is the beta value of the Fb score (typically, beta=1)a

    Returns:
    A tuple (w, b), where:
        w (list[float]): coefficients of a row of the W matrix (Wv = b, where v= [tp,tn,fp,fn])
        b float: one of the intercepts in b, where in Wv = b
    """
    coefs = {
        "C": ([1,1,1,1], value),
        "N_P": ([1,0,0,1], value),
        "A": ([value-1, value-1, value, value], 0),
        "P": ([value-1,0,value,0], 0),
        "R": ([value-1,0,0,value], 0),
        "Fb": ([ (value-1)*(1+beta**2), 0, value, value*beta**2 ], 0)
    }
    return coefs[metric]

def rebuild_metric_range(cm1, cm2, metric, correct_value=None):
    """
    Given an upper and lower confusion matrices and a metric
    to be reconstructed, build the upper and lower bounds for that
    metric and return the range. Optionally make sure that the
    "ground truth value" (i.e. the correct metric) is actually contained
    in the predicted range.

    Parameters:
        cm1, cm2 (list[int]): 
    """
    tp_min, tn_min, fp_min, fn_min = cm1
    tp_max, tn_max, fp_max, fn_max = cm2
    
    if metric == "A":
        v1 = (tp_min + tn_min) / (tp_min + tn_min + fp_min + fn_min)
        v2 = (tp_max + tn_max) / (tp_max + tn_max + fp_max + fn_max)
    elif metric == "P":
        v1 = tp_min / (tp_min + fp_min) if tp_min + fp_min > 0 else 0
        v2 = tp_max / (tp_max + fp_max) if tp_max + fp_max > 0 else 0
    elif metric == "R":
        v1 = tp_min / (tp_min + fn_min) if tp_min + fn_min > 0 else 0
        v2 = tp_max / (tp_max + fn_max) if tp_max + fn_max > 0 else 0
    elif metric == "Fb": # computing F1 here!
        v1 = tp_min / (tp_min + 0.5 * (fp_min + fn_min))
        v2 = tp_max / (tp_max + 0.5 * (fp_max + fn_max))
    else:
        raise Exception(f"Unknown metric {metric}")
    
    if correct_value is not None:
        # if a correct value is passed, we make sure that it
        # is contained within the min & max values computed. 
        # using a small epsilon to avoid false positives due to
        # python rounding problems (e.g. 0.9 ==> 0.900000000005)
        eps = 1e-4
        # assert min(v1,v2) - eps <= correct_value <= max(v1, v2) + eps, f"{v1} {v2} {correct_value}"
        if not (min(v1,v2) - eps <= correct_value <= max(v1, v2) + eps, f"{v1} {v2} {correct_value}"):
            warnings.warn(f"Range found [{v1}, {v2}] does not contain correct value {correct_value}!")
    return abs(v1-v2) # returns range


def make_linear(n_pts=5000, eps=.25, random_state=42):
    """Build a binary dataset that uses the identity
    function as the decision boundary. This is useful because
    decision trees struggle to split such a dataset (increasing
    model capacity (i.e. max depth reached) has progressively
    better performance.
    
    Parameters:
        n_pts (int): number of points to be generated
        eps (float): the value that modulates the additive random noise (the smaller,
        the closer the points will be to the identity function)
        random_state (int): random state to set for reproducibility
    
    Returns:
        X (ndarray[float]): array of points, with shape (n_pts, 2)
        y (ndarray[int]): array of labels (0 or 1), with shape (n_pts,). The points
                          will belong to the two classes in an approximately balanced way
    """
    np.random.seed(random_state)
    x = np.random.random(n_pts)
    X = np.vstack([x,x]).T + np.random.random((n_pts, 2)) * eps
    y = (X[:,0] - X[:,1] <= 0).astype(int)
    return X, y

def train_test(X, y, test_size, clf, random_state=42):
    """Given a binary-labelled dataset, build a classifier (decision tree) and
    return a bunch of metrics (which will be used for reconstruction.
    
    Parameters:
        X: either the feature matrix, or a tuple containing (X_train, X_test), the feature matrices 
           for the (already split) training and test sets
        y: similarly to X, either the target labels or a tuple containing (y_train, y_test), the
           target labels for the (already split) training and test sets.
        test_size (float): the test set size. If None, X and y are assumed to already contain the splits.
        clf (model): model to be used for the training
        random_state (int): random state to set for reproducibility
    
    Returns:
        C (int): test set size
        N_P (int): number of positive samples in the test set
        metrics (dict): a dictionary of key-value pairs. Keys are metrics ('A', 'R', 'P', 'Fb')
                        and values are the performance obtained by the decision tree for each metric
    """
    
    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        X_train, X_test = X
        y_train, y_test = y

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    metrics = {
        "A": accuracy_score(y_test, y_pred),
        "Fb": f1_score(y_test, y_pred, zero_division=0),
        "P": precision_score(y_test, y_pred, zero_division=0),
        "R": recall_score(y_test, y_pred, zero_division=0)
    }
    C = len(y_test)
    N_P = y_test.sum()
    return C, N_P, metrics

