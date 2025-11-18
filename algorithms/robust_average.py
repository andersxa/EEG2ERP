"""
Robust averaging methods for ERP data.

Implements three robust averaging approaches from the literature:
- Leonowicz et al.: Weighted averaging based on amplitude ranking
- Kotowski et al.: Iterative fuzzy clustering approach
- Molina et al.: Dynamic Time Warping (DTW) based averaging
"""

import torch
import numpy as np
from dtaidistance import dtw
from joblib import Parallel, delayed


def leonowicz_predict_fn(x: torch.Tensor):
    """
    Robust averaging using amplitude-based weighting.

    Reference: Leonowicz et al. method

    Args:
        x: Input tensor of shape (repeats, trials, timepoints)

    Returns:
        Averaged signal of shape (repeats, timepoints)
    """
    k = 0.1
    v = 0
    N = x.size(1)

    # Sort by amplitude and create weights
    sorted_indices = torch.argsort(x, dim=1)
    inverse_indices = torch.argsort(sorted_indices, dim=1)
    first_half = (inverse_indices < N//2)
    second_half = (inverse_indices >= N//2)
    inverse_indices = inverse_indices.float()

    weights = (first_half * (torch.tanh(k * (inverse_indices + 1)) - v) +
               second_half * (-torch.tanh(k * (inverse_indices - N)) - v))
    weights /= weights.sum(dim=1, keepdim=True)

    x_pred = (x * weights).sum(dim=1)
    return x_pred


def kotowski_predict_fn(x: torch.Tensor):
    """
    Robust averaging using iterative fuzzy clustering.

    Reference: Kotowski et al. method

    Args:
        x: Input tensor of shape (repeats, trials, timepoints)

    Returns:
        Averaged signal of shape (repeats, timepoints)
    """
    # Hyperparameters
    tol = 1e-8
    m = 2
    c = 100
    norm_p = 2
    exponent = 1 / (1-m)
    N = x.size(1)
    zero_val = 1 / (c * N)
    iterations = 1000

    # Initialize with traditional average
    v = x.mean(dim=1, keepdim=True)

    # Initialize weights using eq. 3
    w_i = (torch.norm(v-x, p=norm_p, dim=-1, keepdim=True)+tol) ** exponent
    w_i = w_i / w_i.sum(dim=1, keepdim=True)
    zero_w = w_i < zero_val
    w_i = torch.where(zero_w, torch.tensor(0.0, device=w_i.device), w_i)
    w_i = w_i / w_i.sum(dim=1, keepdim=True)

    # Update averaged signal using eq. 5
    w_e = w_i ** m
    v = (x * w_e).sum(dim=1, keepdim=True) / w_e.sum(dim=1, keepdim=True)

    for i in range(iterations):
        # Calculate Pearson correlations and rescale to (0, 1)
        v_mean = v.mean(dim=-1, keepdim=True)
        x_mean = x.mean(dim=-1, keepdim=True)

        v_centered = v - v_mean
        x_centered = x - x_mean

        v_ss = (v_centered ** 2).sum(dim=-1, keepdim=True)
        x_ss = (x_centered ** 2).sum(dim=-1, keepdim=True)

        dot_product = (v_centered * x_centered).sum(dim=-1, keepdim=True)
        u = dot_product / (torch.sqrt(v_ss * x_ss))
        u_scaled = (u + 1) / 2

        # Calculate weights using eq. 3
        w_i = (torch.norm(v-x, p=norm_p, dim=-1, keepdim=True)+tol) ** exponent
        w_i = w_i / w_i.sum(dim=1, keepdim=True)

        # Update weights by multiplying by correlation
        w_i = w_i * u_scaled
        w_i = w_i / w_i.sum(dim=1, keepdim=True)

        # Apply zero threshold
        zero_w = w_i < zero_val
        w_i = torch.where(zero_w, torch.tensor(0.0, device=w_i.device), w_i)

        # Update averaged signal using eq. 5
        v_new = (x * w_i).sum(dim=1, keepdim=True) / w_i.sum(dim=1, keepdim=True)

        # Check convergence
        change = torch.norm(v_new - v)
        v = v_new
        if change < tol:
            break

    x_pred = v.squeeze(1)
    return x_pred


def _process_repeat(s_r, s_i, K):
    """Helper function for DTW-based averaging (parallelized)."""
    s_warped = []
    for i in range(K):
        _, paths = dtw.warping_paths_fast(s_r, s_i[i], use_c=True)
        path = dtw.best_path(paths)
        s_warp, path = dtw.warp(s_r, s_i[i], path, use_c=True)
        s_warped.append(s_warp)
    return s_warped


def molina_fast_predict_fn(x: torch.Tensor):
    """
    Robust averaging using Dynamic Time Warping.

    Reference: Molina et al. method

    Args:
        x: Input tensor of shape (repeats, trials, channels, timepoints)

    Returns:
        Averaged signal of shape (repeats, channels, timepoints)
    """
    r = x.mean(dim=1).cpu().numpy().astype(float)
    x_n = x.cpu().numpy().astype(float)

    reps = Parallel(n_jobs=-1)(
        delayed(_process_repeat)(r[repeat].copy(), x_n[repeat].copy(), x.size(1))
        for repeat in range(x.size(0))
    )

    x_pred = torch.stack([
        torch.tensor(v, device=x.device, dtype=x.dtype).mean(dim=0)
        for v in reps
    ])
    return x_pred


def traditional_predict_fn(x: torch.Tensor):
    """
    Traditional averaging (simple mean).

    Args:
        x: Input tensor of shape (repeats, trials, ...)

    Returns:
        Averaged signal of shape (repeats, ...)
    """
    v = x.mean(dim=1)
    return v
