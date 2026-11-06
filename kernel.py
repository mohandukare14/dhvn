"""
4D Euclidean Kernel module for Dynamic Hydro-Void Network (DHVN).
Implements the 4-Dimensional Quintic Spline SPH Kernel and its analytical gradients in R^4.
"""

import math
from typing import Tuple, List

# Normalization constant for 4D Quintic Spline: \alpha_4 = 1 / (64 * pi^2)
ALPHA_4 = 1.0 / (64.0 * (math.pi ** 2))

def euclidean_dist_4d(x1: List[float], x2: List[float]) -> float:
    """Computes standard Euclidean distance in R^4."""
    return math.sqrt(
        (x1[0] - x2[0]) ** 2 +
        (x1[1] - x2[1]) ** 2 +
        (x1[2] - x2[2]) ** 2 +
        (x1[3] - x2[3]) ** 2
    )

def quintic_kernel_4d(r: float, h: float) -> float:
    """
    Evaluates the 4D Quintic Spline smoothing kernel W(r, h).
    r: Euclidean distance between particles.
    h: Smoothing length (support radius is 3h).
    """
    q = r / h
    if q >= 3.0:
        return 0.0

    factor = ALPHA_4 / (h ** 4)
    if q < 1.0:
        val = (3.0 - q)**5 - 6.0 * (2.0 - q)**5 + 15.0 * (1.0 - q)**5
    elif q < 2.0:
        val = (3.0 - q)**5 - 6.0 * (2.0 - q)**5
    else:  # 2.0 <= q < 3.0
        val = (3.0 - q)**5

    return factor * val

def grad_quintic_kernel_4d(x_i: List[float], x_j: List[float], h: float) -> Tuple[List[float], float]:
    """
    Computes analytical spatial gradient of the 4D kernel: \nabla_{x_i} W(r_ij, h).
    Returns (grad_vector_4d, r_ij).
    """
    dx = [x_i[k] - x_j[k] for k in range(4)]
    r = math.sqrt(sum(d * d for d in dx))
    
    if r < 1e-7 or r >= 3.0 * h:
        return [0.0, 0.0, 0.0, 0.0], r

    q = r / h
    factor = ALPHA_4 / (h ** 5)

    if q < 1.0:
        # d/dq [ (3-q)^5 - 6(2-q)^5 + 15(1-q)^5 ]
        dq = -5.0 * (3.0 - q)**4 + 30.0 * (2.0 - q)**4 - 75.0 * (1.0 - q)**4
    elif q < 2.0:
        # d/dq [ (3-q)^5 - 6(2-q)^5 ]
        dq = -5.0 * (3.0 - q)**4 + 30.0 * (2.0 - q)**4
    else:
        # d/dq [ (3-q)^5 ]
        dq = -5.0 * (3.0 - q)**4

    dW_dr = factor * dq
    # \nabla_{x_i} W = (dW/dr) * (dx / r)
    grad = [dW_dr * (dx[k] / r) for k in range(4)]
    return grad, r

