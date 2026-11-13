import numpy as np

def online_simple_regression(xs, ys):
    n = len(xs)
    results = np.zeros((n, 2))
    
    mean_x = 0.0
    mean_y = 0.0
    M_xx = 0.0
    M_xy = 0.0
    
    for i in range(n):
        k = i + 1
        x = xs[i]
        y = ys[i]
        
        if k == 1:
            mean_x = x
            mean_y = y
            M_xx = 0.0
            M_xy = 0.0
        else:
            dx = x - mean_x
            dy = y - mean_y
            mean_x += dx / k
            mean_y += dy / k
            M_xx += dx * (x - mean_x)
            M_xy += dx * (y - mean_y)
            
        C_xx = M_xx / k
        
        if C_xx == 0.0 or M_xx == 0.0:
            w1 = 0.0
            w0 = mean_y
        else:
            w1 = M_xy / M_xx
            w0 = mean_y - w1 * mean_x
            
        results[i] = [w0, w1]
        
    return results