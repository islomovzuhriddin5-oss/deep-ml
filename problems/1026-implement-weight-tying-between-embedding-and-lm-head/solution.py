import numpy as np

def tie_weights(token_embedding, lm_head_weight):
    if lm_head_weight is None:
        return token_embedding
    
    token_embedding = np.asarray(token_embedding)
    lm_head_weight = np.asarray(lm_head_weight)
    
    if token_embedding.shape != lm_head_weight.shape:
        raise ValueError("Shape of lm_head_weight does not match token_embedding.")
        
    return lm_head_weight