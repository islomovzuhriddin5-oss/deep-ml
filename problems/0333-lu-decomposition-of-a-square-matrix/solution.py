import numpy as np

def lu_decomposition(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    
    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)
    
    for i in range(n):
        # Compute U entries for row i
        for j in range(i, n):
            s = sum(L[i, k] * U[k, j] for k in range(i))
            U[i, j] = A[i, j] - s
            
        # Compute L entries for column i
        for j in range(i + 1, n):
            s = sum(L[j, k] * U[k, i] for k in range(i))
            L[j, i] = (A[j, i] - s) / U[i, i]
            
    return (L, U)