def poly_term_derivative(c: float, x: float, n: float) -> float:
    if n == 0:
        return 0.0
    return c * n * (x ** (n - 1))