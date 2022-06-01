
from utils import get_coef_intercept
import cvxpy as cp
from mip_cvxpy import PYTHON_MIP
import numpy as np
import warnings


def solve_ip_problem(**kwargs):
    v = cp.Variable(4, integer=True)
    
    constraints = [
        v >= 0
    ]
    
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
        while d < 4:
            W[i] = np.eye(4)[d]
            if np.linalg.det(W) != 0:
                # not singular -- use this!
                break
            d += 1

        if d > 3:
            # there is no other suitable solution. This means that some boundaries
            # will not be found. Raise a warning, but continue
            warnings.warn("No suitable variable found for optimization. No upper bound may be found")
            d = 0
    elif i > 4:
        warnings.warn(f"{i} constraints found. The problem may be overly constrained")

    obj_min = cp.Minimize(v[d])
    obj_max = cp.Maximize(v[d])
    
    prob_min = cp.Problem(obj_min, constraints=constraints)
    prob_max = cp.Problem(obj_max, constraints=constraints)
    
    # previously: "SCIP"
    prob_min.solve(solver=PYTHON_MIP())
    min_val = v.value
    prob_max.solve(solver=PYTHON_MIP())
    max_val = v.value
    
    return min_val, max_val