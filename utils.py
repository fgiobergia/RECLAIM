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