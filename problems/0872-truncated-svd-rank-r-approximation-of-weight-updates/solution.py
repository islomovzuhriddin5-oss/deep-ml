import numpy as np

def low_rank_approximation(delta_W, r):
    delta_W = np.array(delta_W)
    # SVD ni hisoblaymiz: delta_W = U @ S_matrix @ Vt
    U, S, Vt = np.linalg.svd(delta_W, full_matrices=False)
    
    # Faqat dastlabki r ta komponentni olamiz
    U_r = U[:, :r]
    S_r = np.diag(S[:r])
    Vt_r = Vt[:r, :]
    
    # Rank-r yaqinlashmani hisoblaymiz
    approx = U_r @ S_r @ Vt_r
    
    # Python ro'yxatiga o'tkazib qaytaramiz
    return approx.tolist()