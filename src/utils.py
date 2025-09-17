import numpy as np

def _norm(vec):
    """Safe normalization of a 2D vector."""
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-12 else vec