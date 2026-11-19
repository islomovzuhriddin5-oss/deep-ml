import math


def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    a, b = matrix[0]
    c, d = matrix[1]

    trace = a + d
    det = a * d - b * c
    discriminant = max(0.0, trace**2 - 4 * det)
    sqrt_disc = math.sqrt(discriminant)

    lambda_1 = (trace + sqrt_disc) / 2.0
    lambda_2 = (trace - sqrt_disc) / 2.0

    eigenvalues = sorted([float(lambda_1), float(lambda_2)], reverse=True)
    return eigenvalues