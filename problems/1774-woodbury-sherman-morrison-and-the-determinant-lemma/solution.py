import numpy as np

def woodbury_solve(sigma_diag, X, B):
    sigma_diag = np.asarray(sigma_diag, dtype=float)
    X = np.asarray(X, dtype=float)
    B = np.asarray(B, dtype=float)
    
    # \Sigma^{-1} B and \Sigma^{-1} X via elementwise broadcast: O(N) and O(ND)
    inv_sigma_B = B / sigma_diag[:, None] if B.ndim > 1 else B / sigma_diag
    inv_sigma_X = X / sigma_diag[:, None]
    
    # Form D x D matrix: I_D + X^T \Sigma^{-1} X
    D = X.shape[1]
    M = np.eye(D) + X.T @ inv_sigma_X
    
    # Solve M Y = X^T \Sigma^{-1} B for Y in O(D^3 + D^2 K) or O(D^3 + D^2)
    XT_inv_sigma_B = X.T @ inv_sigma_B
    Y = np.linalg.solve(M, XT_inv_sigma_B)
    
    # Final result: \Sigma^{-1} B - \Sigma^{-1} X Y
    return inv_sigma_B - inv_sigma_X @ Y

def sherman_morrison(A_inv, u, v):
    A_inv = np.asarray(A_inv, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    
    # Compute A^{-1} u and v^T A^{-1}
    A_inv_u = A_inv @ u
    vT_A_inv = v @ A_inv
    
    # Scalar denominator: 1 + v^T A^{-1} u
    denom = 1.0 + np.dot(v, A_inv_u)
    
    if np.abs(denom) < 1e-12:
        raise ValueError("Matrix update is singular (1 + v^T A^{-1} u is zero).")
        
    return A_inv - np.outer(A_inv_u, vT_A_inv) / denom

def logdet_rank_one_update(logdet_A, A_inv, u, v):
    A_inv = np.asarray(A_inv, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    
    # Compute 1 + v^T A^{-1} u
    factor = 1.0 + float(v @ (A_inv @ u))
    
    if factor <= 0:
        raise ValueError("Updated determinant is non-positive.")
        
    return float(logdet_A + np.log(factor))