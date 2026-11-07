import numpy as np

def partitioned_inverse(M, k):
    M = np.array(M, dtype=float)
    E = M[:k, :k]
    F = M[:k, k:]
    G = M[k:, :k]
    H = M[k:, k:]
    
    H_inv = np.linalg.inv(H)
    schur_H = E - F @ H_inv @ G
    schur_H_inv = np.linalg.inv(schur_H)
    
    top_left = schur_H_inv
    top_right = -schur_H_inv @ F @ H_inv
    bottom_left = -H_inv @ G @ schur_H_inv
    bottom_right = H_inv + H_inv @ G @ schur_H_inv @ F @ H_inv
    
    return np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])

def block_slogdet(M, k):
    M = np.array(M, dtype=float)
    E = M[:k, :k]
    F = M[:k, k:]
    G = M[k:, :k]
    H = M[k:, k:]
    
    H_inv = np.linalg.inv(H)
    schur_H = E - F @ H_inv @ G
    
    sign_H, logabs_H = np.linalg.slogdet(H)
    sign_schur, logabs_schur = np.linalg.slogdet(schur_H)
    
    sign = float(sign_schur * sign_H)
    logabsdet = float(logabs_schur + logabs_H)
    
    return (sign, logabsdet)

def bordered_inverse(A_inv, b, c):
    A_inv = np.array(A_inv, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    
    s = c - (b.T @ A_inv @ b)[0, 0]
    if np.abs(s) < 1e-12:
        raise ValueError("Schur complement s is too close to zero.")
    
    top_left = A_inv + (1.0 / s) * (A_inv @ b @ b.T @ A_inv)
    top_right = -(1.0 / s) * (A_inv @ b)
    bottom_left = -(1.0 / s) * (b.T @ A_inv)
    bottom_right = np.array([[1.0 / s]])
    
    return np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])