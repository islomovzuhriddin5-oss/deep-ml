import numpy as np

def qr_decomposition(A):
    A = np.array(A, dtype=float)
    m, n = A.shape
    
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)
    
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v = v - R[i, j] * Q[:, i]
        
        norm_v = np.linalg.norm(v)
        R[j, j] = norm_v
        if norm_v > 1e-12:
            Q[:, j] = v / norm_v
        else:
            Q[:, j] = v
            
    return (Q, R)