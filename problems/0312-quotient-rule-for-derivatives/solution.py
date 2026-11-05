import numpy as np
def quotient_rule_derivative(g_coeffs, h_coeffs, x, decimals=None):
    def eval_poly(coeffs, val):
        res = 0
        for c in coeffs:
            res = res * val + c
        return res

    def derivative_coeffs(coeffs):
        n = len(coeffs) - 1
        if n <= 0:
            return [0]
        return [coeffs[i] * (n - i) for i in range(n)]

    g_val = eval_poly(g_coeffs, x)
    h_val = eval_poly(h_coeffs, x)
    g_prime_val = eval_poly(derivative_coeffs(g_coeffs), x)
    h_prime_val = eval_poly(derivative_coeffs(h_coeffs), x)

    numerator = (g_prime_val * h_val) - (g_val * h_prime_val)
    denominator = h_val ** 2
    
    result = numerator / denominator
    return round(result, decimals) if decimals is not None else result