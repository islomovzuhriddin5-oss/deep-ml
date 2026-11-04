import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    arr = np.array(a)
    if arr.size != np.prod(new_shape):
        return []
    try:
        return arr.reshape(new_shape).tolist()
    except Exception:
        return []