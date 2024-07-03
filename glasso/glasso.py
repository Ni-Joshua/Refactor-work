"""
Code obtained from Ignavier's another ongoing project on missing data
"""
import numpy as np
from sklearn.covariance import graphical_lasso

from glasso.em_utils import get_miss_group_info, E_step


def glasso(X, glasso_lambda=0.01, glasso_iter=1000):
    cov_emp = np.cov(X.T, bias=True)
    _, inv_cov_est = graphical_lasso(cov_emp, alpha=glasso_lambda, max_iter=glasso_iter)
    return inv_cov_est

def miss_glasso(X, mask, em_iter=20, glasso_lambda=0.01, glasso_iter=1000):
    """
    - Reference: https://arxiv.org/abs/0903.5463
    - X corresponds to the observed samples with missing entries (i.e., NaN)
    - mask has the same shape as X; they both have a shape of (n, d)
    - If an entry in mask is False, the corresponding entry in X is a missing value (i.e., NaN)
    """
    print("Start fitting MissGLasso Model")
    n, d = X.shape
    miss_group_df = get_miss_group_info(mask)
    # print(miss_group_df)

    # Initial values
    mu_init = np.zeros((d))
    K_init = np.eye(d)

    # Variables for EM algorithms
    mu_m = mu_init
    K_m = K_init

    X_m = None

    for m in range(1, em_iter + 1):
        ###################### E-step ######################
        T1_m, T2_m, X_m = E_step(X, miss_group_df, mu_m, K_m)
        ####################################################

        ###################### M-step ######################
        mu_m = T1_m / n
        S_m = T2_m / n - np.outer(mu_m, mu_m)
        S_m, K_m = graphical_lasso(S_m, alpha=glasso_lambda, max_iter=glasso_iter, tol=1e-4)
        ####################################################
    return S_m, K_m, X_m    # Imputed data