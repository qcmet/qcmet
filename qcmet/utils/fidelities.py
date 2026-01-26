"""fidelity_metrics.py.

This module provides functions for computing fidelity metrics.
"""

import numpy as np


def fidelity(P1, P2):
    """Compute the fidelity between two probability distributions.
     
    Fidelity is defined as the square of the sum of the square roots of the product of
    corresponding probabilities in the two distributions.

    Args:
        P1 (list | np.ndarray): First probability distribution.
        P2 (list | np.ndarray): Second probability distribution. Must be the same length as P1.

    Returns:
        float: Fidelity value between the two distributions.

    Raises:
        AssertionError: If the input distributions are not of the same length.

    """
    assert len(P1) == len(P2)

    fidelity = 0.
    for i in range(len(P1)):
        fidelity += np.sqrt(P1[i] * P2[i])
    fidelity = fidelity**2
    return fidelity


def normalized_fidelity(exact_probs, device_probs) -> float:
    """Normalize fidelity against a fully depolarized output distribution.

    This metric compares how close the hardware
    distribution is to the exact one, normalized against a fully depolarized
    output distribution.

    Args:
        exact_probs (list or np.ndarray): Ideal or exact probability distribution.
        device_probs (list or np.ndarray): Measured or hardware probability distribution.

    Returns:
        float: Normalized fidelity score in the range [0, 1].

    """
    uniform_dist = np.array([1 for _ in exact_probs]) / len(exact_probs)
    f_ideal_and_uniform = fidelity(exact_probs, uniform_dist)
    f_ideal_and_hardware = fidelity(exact_probs, device_probs)
    return max(
        (f_ideal_and_hardware - f_ideal_and_uniform) / (1 - f_ideal_and_uniform), 0
    )
