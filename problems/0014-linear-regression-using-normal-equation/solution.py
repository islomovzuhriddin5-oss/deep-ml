import numpy as np

def linear_regression_normal_equation(X, y):
    """
    Computes the linear regression coefficients using the normal equation.
    Returns a list of coefficients rounded to four decimal places.
    """
    X = np.array(X)
    y = np.array(y)
    
    # Calculate theta using the normal equation: (X^T * X)^(-1) * X^T * y
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # Convert to list and round each coefficient to 4 decimal places
    return [round(float(val), 4) for val in theta]