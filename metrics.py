import numpy as np
from utils import get_coef_intercept
from ip import solve_ip_problem



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
        # solve either as IP problem, or as accuracy-based
        
        if set(vals) == { "A", "N_P", "C"}:
            # accuracy-based
            pass
        else:
            # IP problem
            return solve_ip(beta=kwargs.get("beta", 1), **vals)
    
    else:
        # 4+ constraints available -- extract v = A^(-1)b
        # build A with 4 available constraints -- if other constraints
        # are available, use those for an additional check
        
        # if C and N_P are available, prefer them over others
        pref_set = {"C", "N_P"}
        A_vals = pref_set & set(vals)
        A_vals |= set(list(set(vals) - pref_set)[:4-len(A_vals)])
        A_vals = list(A_vals)
        
        A = np.zeros((4,4))
        b = np.zeros(4)
        for i, v in enumerate(A_vals):
            a_, b_ = get_coef_intercept(vals[v], v, kwargs.get("beta", 1))
            A[i] = a_
            b[i] = b_
        print(A, b)
        print(A_vals)
        
        return np.linalg.solve(A, b)
