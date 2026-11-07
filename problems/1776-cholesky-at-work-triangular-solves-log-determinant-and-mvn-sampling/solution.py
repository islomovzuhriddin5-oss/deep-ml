import numpy as np

def forward_sub(L, b):
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    y = np.zeros(n, dtype=float)
    
    for i in range(n):
        s = np.dot(L[i, :i], y[:i])
        y[i] = (b[i] - s) / L[i, i]
        
    return y.tolist()

def back_sub(U, y):
    U = np.asarray(U, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    x = np.zeros(n, dtype=float)
    
    for i in range(n - 1, -1, -1):
        s = np.dot(U[i, i+1:], x[i+1:])
        x[i] = (y[i] - s) / U[i, i]
        
    return x.tolist()

def chol_solve(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    
    L = np.linalg.cholesky(A)
    y = forward_sub(L, b)
    x = back_sub(L.T, np.array(y))
    
    return np.asarray(x, dtype=float)

def logdet_chol(A):
    A = np.asarray(A, dtype=float)
    L = np.linalg.cholesky(A)
    # log |A| = log |LL^T| = log(|L| * |L^T|) = 2 * log |L| = 2 * sum(log(L_ii))
    diag_L = np.diagonal(L)
    return float(2.0 * np.sum(np.log(diag_L)))

def mvn_sample(mu, Sigma, eps):
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    eps = np.asarray(eps, dtype=float)
    
    L = np.linalg.cholesky(Sigma)
    # eps has shape (m, d), rows are standard normal draws.
    # We want row i: L @ eps[i] + mu -> (eps[i] @ L.T) + mu
    samples = eps @ L.T + mu
    return samples