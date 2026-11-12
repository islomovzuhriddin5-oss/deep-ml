import numpy as np

def power_grid_forecast(data):
    """
    Computes power consumption prediction for day 15 given 10 days of raw readings.
    """
    # Step 1: Remove the known fluctuation f_i = 10 * sin(2 * pi * i / 10)
    n = len(data)
    days = np.arange(1, n + 1, dtype=float)
    fluctuations = 10 * np.sin(2 * np.pi * days / n)
    
    detrended_y = np.array(data, dtype=float) - fluctuations
    
    # Step 2: Fit a linear regression model (y = m * day + c) to the detrended data
    X = np.vstack([days, np.ones(n)]).T
    m, c = np.linalg.lstsq(X, detrended_y, rcond=None)[0]
    
    # Step 3: Predict day 15's base consumption and add back the day 15 fluctuation
    day_15 = 15.0
    base_15 = m * day_15 + c
    
    fluctuation_15 = 10 * np.sin(2 * np.pi * day_15 / n)
    pred_15 = base_15 + fluctuation_15
    
    # Step 4: Round prediction to the nearest integer, multiply by 1.05, and ceiling
    rounded_pred = round(pred_15)
    with_margin = rounded_pred * 1.05
    final_answer = int(np.ceil(with_margin))
    
    return final_answer