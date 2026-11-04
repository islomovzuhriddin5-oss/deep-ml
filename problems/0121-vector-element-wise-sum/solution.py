def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float] | int:
    if len(a) != len(b):
        return -1
    return [x + y for x, y in zip(a, b)]