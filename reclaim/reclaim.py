import numpy as np
from .ip import solve_ip_problem, get_coef_intercept

def find_confusion_matrix(**kwargs):
    """Given some constraints as input, find the confusion matrix (or its boundaries). 

    This function automatically decides the policy to be used to solve the problem, based
    on the number of available constraints. 

    Parameters:
        All parameters are passed as keyword arguments.
        C, N_P (int): the dataset size and the number of positive samples, respectively. C > 0, 0 < N_P <= C
        N_N: May be passed instead of N_P, represents the number of negative samples. Because this function handles
             a binary classification problem, N_P is computed as C - N_P
        A, P, R, Fb (float): accuracy, precision, recall and F-beta score for the positive class, respectively. All metrics are expected to be in [0,1]
        beta (float): beta parameter for the F-beta score. Defaults to 1 (F1 score)
    
    Returns:
        cm1 (list[int]): lower bound for the confusion matrix (w.r.t. one of the values). The values are returned as [tp, tn, fp, fn]
        cm2 (list[int]): upper bound for the confusion matrix (w.r.t. one of the values). The values are returned as [tp, tn, fp, fn]. Only returned if the problem cannot be solved exactly
    """

    keys = ["C", "N_P", "P", "R", "Fb", "A"]
    
    if "N_P" not in kwargs and all([x in kwargs for x in ["N_N","C"]]):
        # always work with N_P
        kwargs["N_P"] = kwargs["C"] - kwargs["N_N"]
    
    vals = { k: kwargs[k] for k in keys if k in kwargs }
    
    if len(vals) < 3:
        # not enough constraints to be meaningful -- raise exception
        raise Exception("Not enough constraints available")
    
    if len(vals) < 4:
        # solve as IP problem
        # TODO: add accuracy-based solution w/o ip problem?
        return solve_ip_problem(beta=kwargs.get("beta", 1), **vals)
    
    elif len(vals) >= 4:
        # 4+ constraints available -- extract v = A^(-1)b
        # (if > 4 constraints, use pseudoinverse)
                
        A = np.zeros((len(vals),4))
        b = np.zeros(len(vals))
        for i, v in enumerate(vals):
            a_, b_ = get_coef_intercept(vals[v], v, kwargs.get("beta", 1))
            A[i] = a_
            b[i] = b_
        
        if len(vals) == 4:
            return np.linalg.solve(A, b)
        else:
            return np.dot(np.linalg.pinv(A), b)
