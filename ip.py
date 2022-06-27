import cvxpy as cp
from mip_cvxpy import PYTHON_MIP
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


def solve_ip_problem(**kwargs):
    """Create and solve IP problem, given the metrics as inputs (same format
    as find_confusion_matrix()).

    The constraints can be expressed in either of two ways:

    metric=value: in this case, an equality constraint is introduced
    metric=(min_value, max_value): in this case, two inequality constraints are introduced, 
                                   so as to guarantee that the solution produces the metric
                                   that is contained in the specified range. This is useful, 
                                   for example, when the available value for the metric has
                                   been rounded.
    """

    # vector representing confusion matrix, as [tp, tn, fp, fn]
    # enforcing integer constraint
    v = cp.Variable(4, integer=True)
    
    # the 1st constraint is on all values, which are expected to be positive
    constraints = [
        v >= 0
    ]
    
    # W is used to keep track of the constraints currently in place -- used
    # to determine d (i.e. which element of the confusion matrix) should be
    # used for the maximiziation & minimization problems.
    W = np.zeros((4,4))
    i = 0
    if "C" in kwargs:
        C = kwargs["C"]
        if isinstance(C, tuple):
            C_min, C_max = C
            constraints.append(sum(v) >= C_min)
            constraints.append(sum(v) <= C_max)
            C = C_min
        else:
            constraints.append(sum(v) == kwargs["C"])
        w, _ = get_coef_intercept(C, "C")
        W[i] = w
        i += 1
    if "N_P" in kwargs:
        N_P = kwargs["N_P"]
        if isinstance(N_P, tuple):
            N_P_min, N_P_max = N_P
            constraints.append(v[0] + v[3] >= N_P_min)
            constraints.append(v[0] + v[3] <= N_P_max)
            N_P = N_P_min
        else:
            constraints.append(v[0] + v[3] == kwargs["N_P"])
        w, _ = get_coef_intercept(N_P, "N_P")
        W[i] = w
        i += 1
    if "P" in kwargs:
        P = kwargs["P"]
        if isinstance(P, tuple):
            P_min, P_max = P
            constraints.append((P_min-1)*v[0] + P_min*v[2] <= 0)
            constraints.append((P_max-1)*v[0] + P_max*v[2] >= 0)
            P = P_min
        else:
            constraints.append((P-1)*v[0] + P*v[2] == 0)
        if P > 0:
            constraints.append(v[0] >= 1)
        w, _ = get_coef_intercept(P, "P")
        W[i] = w
        i += 1
    if "R" in kwargs:
        R = kwargs["R"]
        if isinstance(R, tuple):
            R_min, R_max = R
            constraints.append((R_min-1)*v[0] + R_min * v[3] <= 0)
            constraints.append((R_max-1)*v[0] + R_max * v[3] >= 0)
            R = R_min
        else:
            constraints.append((R-1)*v[0] + R * v[3] == 0)
        if R > 0:
            constraints.append(v[0] >= 1)
        w, _ = get_coef_intercept(R, "R")
        W[i] = w
        i += 1
    if "A" in kwargs:
        A = kwargs["A"]
        if isinstance(A, tuple):
            A_min, A_max = A
            constraints.append((A_min-1)*v[0] + (A_min-1)*v[1] + A_min*v[2] + A_min*v[3] <= 0)
            constraints.append((A_max-1)*v[0] + (A_max-1)*v[1] + A_max*v[2] + A_max*v[3] >= 0)
            A = A_min
        else:
            constraints.append((A-1)*v[0] + (A-1)*v[1] + A*v[2] + A*v[3] == 0)
        w, _ = get_coef_intercept(A, "A")
        W[i] = w
        i += 1
    if "Fb" in kwargs:
        Fb = kwargs["Fb"]
        beta = kwargs.get("beta", 1)
        if isinstance(Fb, tuple):
            Fb_min, Fb_max = Fb
            constraints.append(v[0] * (Fb_min-1)*(1+beta**2) + Fb_min * v[2] + Fb_min * beta ** 2 * v[3] <= 0)
            constraints.append(v[0] * (Fb_max-1)*(1+beta**2) + Fb_max * v[2] + Fb_max * beta ** 2 * v[3] >= 0)
            Fb = Fb_min
        else:
            constraints.append(v[0] * (Fb-1)*(1+beta**2) + Fb * v[2] + Fb * beta ** 2 * v[3] == 0)
        w, _ = get_coef_intercept(Fb, "Fb", beta)
        W[i] = w
        i += 1
    
    d = 0
    if i == 3:

        # To decide the 4th constraint to be introduced (which
        # fixes one of tp, tn, fp, fn), we need to make sure that
        # the introduction of the 4th constraint results in a 
        # matrix of coefficients W that has det(W) > 0. 
        # If that is not the case, we are still leaving at least 
        # one degree of freedom to the problem. 
        while d < 4:
            W[i] = np.eye(4)[d]
            if np.linalg.det(W) != 0:
                # not singular -- use this!
                break
            d += 1

        if d > 3:
            # there is no other suitable solution. This means that some boundaries
            # will not be found. Raise a warning, but continue. 
            # This may occur, for example, when the constraints passed are Fb, P, R.
            warnings.warn("No suitable variable found for optimization. No upper bound may be found")
            d = 0
    elif i > 4:
        warnings.warn(f"{i} constraints found. The problem may be overly constrained")

    obj_min = cp.Minimize(v[d])
    obj_max = cp.Maximize(v[d])
    
    prob_min = cp.Problem(obj_min, constraints=constraints)
    prob_max = cp.Problem(obj_max, constraints=constraints)
    
    # solve minimization and a maximization problems. Return 
    # the obtained matrices as lower and upper bounds (w.r.t. some element
    # of the confusion matrix)
    # NOTE: previously using the "SCIP" solver. Either works fine.
    prob_min.solve(solver=PYTHON_MIP())
    min_val = v.value
    prob_max.solve(solver=PYTHON_MIP())
    max_val = v.value
    
    return min_val, max_val