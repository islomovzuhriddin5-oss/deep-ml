import numpy as np

def min_norm_solution(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    
    AAT = A @ A.T
    y = np.linalg.solve(AAT, b)
    return A.T @ y

def min_norm_via_kkt(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    
    AAT = A @ A.T
    lam = np.linalg.solve(AAT, -2.0 * b)
    x = -0.5 * (A.T @ lam)
    
    return x, lam

def min_norm_from_particular(A, x_p):
    A = np.asarray(A, dtype=float)
    x_p = np.asarray(x_p, dtype=float)
    
    Ax_p = A @ x_p
    AAT = A @ A.T
    y = np.linalg.solve(AAT, Ax_p)
    return A.T @ y