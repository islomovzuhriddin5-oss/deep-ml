import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    if x0 is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x = np.array(x0, dtype=float)
        
    r = b - A @ x
    p = r.copy()
    rs_old = r @ r
    
    for _ in range(n):
        if np.sqrt(rs_old) < tol:
            break
            
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        
        rs_new = r @ r
        if np.sqrt(rs_new) < tol:
            break
            
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
        
    return x.tolist()