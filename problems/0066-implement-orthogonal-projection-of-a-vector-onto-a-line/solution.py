def orthogonal_projection(v, L):
    """
    Computes the orthogonal projection of vector v onto vector L.
    Returns a list representing the projection vector rounded to three decimal places.
    """
    dot_v_L = sum(v_i * l_i for v_i, l_i in zip(v, L))
    dot_L_L = sum(l_i * l_i for l_i in L)
    
    if dot_L_L == 0:
        raise ValueError("The line vector L cannot be a zero vector.")
        
    scalar = dot_v_L / dot_L_L
    
    # Scale vector L and round each component to three decimal places
    return [round(scalar * l_i, 3) for l_i in L]