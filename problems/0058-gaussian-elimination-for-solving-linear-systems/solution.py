import numpy as np

def gaussian_elimination(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    # Create augmented matrix [A | b]
    Ab = np.column_stack((A, b))
    
    for i in range(n):
        # Partial pivoting: find the row with the largest absolute value in the current column
        max_row = i + np.argmax(np.abs(Ab[i:, i]))
        
        # Swap the current row with the max_row
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]
            
        pivot = Ab[i, i]
        if np.abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular.")
            
        # Normalize the pivot row
        Ab[i] = Ab[i] / pivot
        
        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            factor = Ab[j, i]
            Ab[j] -= factor * Ab[i]
            
    # Backward substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])
        
    return x.tolist()