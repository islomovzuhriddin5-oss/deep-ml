def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if not matrix or not matrix[0]:
        return []
    
    rows = len(matrix)
    cols = len(matrix[0])
    
    if mode == 'row':
        # Har bir qatorning o'rtacha qiymatini hisoblash
        return [sum(row) / cols for row in matrix]
    
    elif mode == 'column':
        # Har bir ustunning o'rtacha qiymatini hisoblash
        means = []
        for c in range(cols):
            col_sum = sum(matrix[r][c] for r in range(rows))
            means.append(col_sum / rows)
        return means
        
    return []