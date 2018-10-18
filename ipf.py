import numpy as np
import pandas as pd
from patsy import dmatrices
import statsmodels.api as sm

def ipf_2D(row_total, col_total, x = None, maxitr = 3, tol = 10e-5, debug = False):
    """Standard implementation of ipf
    Inputs:
        row_total: marginal distribution for variable 1
        col_total: marginal distribution for variable 2
        x: initial estimates for joint distribution
        maxitr: maximum number of iterations
        tol: error tolerance
    Output:
        mle estimates for joint distribution
    """
    if(x is None):
        x = np.ones((len(row_total),len(col_total)))
    xnew = x = np.copy(x*1.0)
    r = 1
    converged = False
    while(converged is False):
        xnew = (x*row_total[:,np.newaxis])/np.sum(x,axis = 1)[:,np.newaxis]
        xnew = (xnew*col_total)/np.sum(xnew,axis = 0)
        r += 1
        x = np.copy(xnew)
        if(np.amax(np.absolute(np.subtract(xnew,x)))<tol):
            converged = True
        if(r>maxitr):
            converged = True
        if(debug):
            print('Iteration: {}, x: {}'.format(r,x))
    return(x)

def ipf_2D_poisson(row_total, col_total, x = None):
    """Standard implementation of ipf
    Inputs:
        row_total: marginal distribution for variable 1
        col_total: marginal distribution for variable 2
        x: initial estimates for joint distribution
    Output:
        estimates for joint distribution using poisson regression
    """
    if(x is None):
        x = np.ones((len(row_total),len(col_total)))
    data = pd.DataFrame()
    data['y'] = x.reshape(-1)
    data['i'] = np.repeat(row_total, len(col_total))
    data['j'] = np.tile(col_total, len(row_total))
    formula = "y ~ i + j"
    response, predictors = dmatrices(formula, data, return_type='dataframe')
    po_results = sm.GLM(response, predictors, family=sm.families.Poisson()).fit()
    x = po_results.predict(predictors.values).reshape(x.shape)
    return(x)
