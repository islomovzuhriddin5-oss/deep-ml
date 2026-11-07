import numpy as np

def cholesky_decomposition(A):
    if A is None:
        return -1
    
    try:
        A = np.array(A, dtype=float)
    except Exception:
        return -1
    
    # Check if empty or non-2D
    if A.ndim != 2 or A.size == 0:
        return -1
    
    n, m = A.shape
    # Check if non-square or asymmetric
    if n != m or not np.allclose(A, A.T, atol=1e-8):
        return -1
        
    L = np.zeros((n, n), dtype=float)
    
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            
            if i == j:
                val = A[i, i] - s
                if val <= 0:
                    return -1
                L[i, j] = np.sqrt(val)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
                
    return L.tolist()