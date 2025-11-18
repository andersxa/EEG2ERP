"""
Woody's adaptive filter for ERP latency estimation.

Reference: Woody, C. D. (1967). Characterization of an adaptive filter for the
analysis of variable latency neuroelectric signals. Medical and Biological
Engineering, 5(6), 539-553.
"""

import numpy as np
from scipy.signal import correlate


def woody_latency_estimation(
    data: np.ndarray,
    template: np.ndarray,
    time_window: tuple,
    search_duration_samples: int,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-6
):
    """
    Estimate single-trial latencies using Woody's adaptive filter method.

    The algorithm iteratively:
    1. Cross-correlates each trial with a template to find the peak
    2. Aligns trials based on estimated latencies
    3. Updates the template as the average of aligned trials
    4. Repeats until convergence

    Args:
        data: EEG data of shape (n_samples, n_trials)
        template: Initial template waveform of shape (n_samples,)
        time_window: Tuple (start_sample, end_sample) defining the search window
        search_duration_samples: Half-width of the search range in samples
        max_iterations: Maximum number of iterations
        convergence_threshold: Threshold for convergence (norm of latency changes)

    Returns:
        latencies: Array of shape (n_trials,) with estimated latencies in samples
        aligned_template: Final template after alignment
    """
    n_samples, n_trials = data.shape
    latencies = np.zeros(n_trials, dtype=int)

    win_start, win_end = time_window
    win = slice(win_start, win_end)

    for iteration in range(max_iterations):
        latencies_old = latencies.copy()
        template_segment = template[win]

        # Estimate latencies for each trial
        for trial_idx in range(n_trials):
            trial_segment = data[win, trial_idx]

            # Cross-correlate trial with template (remove means for robustness)
            xcorr = correlate(
                trial_segment - trial_segment.mean(),
                template_segment - template_segment.mean(),
                mode='same'
            )

            # Find peak within search range
            center = len(xcorr) // 2
            search_start = max(0, center - search_duration_samples)
            search_end = min(len(xcorr), center + search_duration_samples)
            search_range = xcorr[search_start:search_end]

            peak_idx = np.argmax(search_range)
            latencies[trial_idx] = peak_idx - search_duration_samples

        # Center latencies around zero (subtract median)
        latencies -= int(np.median(latencies))

        # Align trials and update template
        aligned_data = np.zeros_like(data)
        for trial_idx in range(n_trials):
            aligned_data[:, trial_idx] = np.roll(data[:, trial_idx], -latencies[trial_idx])

        template = np.mean(aligned_data, axis=1)

        # Check for convergence
        if np.linalg.norm(latencies - latencies_old) < convergence_threshold:
            break

    return latencies, template


def woody_single_pass(
    data: np.ndarray,
    template: np.ndarray,
    time_window: tuple,
    search_duration_samples: int
):
    """
    Single-pass latency estimation (no iterative refinement).

    This is the basic cross-correlation step used in Woody's method
    and also as initialization in RIDE.

    Args:
        data: EEG data of shape (n_samples, n_trials)
        template: Template waveform of shape (n_samples,)
        time_window: Tuple (start_sample, end_sample) defining the search window
        search_duration_samples: Half-width of the search range in samples

    Returns:
        latencies: Array of shape (n_trials,) with estimated latencies in samples
    """
    n_samples, n_trials = data.shape
    latencies = np.zeros(n_trials, dtype=int)

    win_start, win_end = time_window
    win = slice(win_start, win_end)
    template_segment = template[win]

    for trial_idx in range(n_trials):
        trial_segment = data[win, trial_idx]

        # Cross-correlate trial with template
        xcorr = correlate(
            trial_segment - trial_segment.mean(),
            template_segment - template_segment.mean(),
            mode='same'
        )

        # Find peak within search range
        center = len(xcorr) // 2
        search_start = center - search_duration_samples
        search_end = center + search_duration_samples
        peak_idx = np.argmax(xcorr[search_start:search_end])
        latencies[trial_idx] = peak_idx - search_duration_samples

    # Center latencies around zero
    latencies -= int(np.median(latencies))

    return latencies
