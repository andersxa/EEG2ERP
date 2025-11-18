"""
Common preprocessing utilities for EEG/ERP data.

This module provides shared functionality for preprocessing different datasets.
"""

import numpy as np
import random


def create_epochs(data, event_indices, tmin, srate, n_samples_epoch):
    """
    Create epochs from continuous data based on event indices.

    Args:
        data: Continuous EEG data of shape (n_channels, n_samples)
        event_indices: Array of event sample indices
        tmin: Start time of epoch in seconds (typically negative for pre-stimulus)
        srate: Sampling rate in Hz
        n_samples_epoch: Number of samples per epoch

    Returns:
        epochs: Array of shape (n_epochs, n_channels, n_samples_epoch)
    """
    epochs = []
    for ev in event_indices:
        start = int(ev + tmin * srate)
        stop = start + n_samples_epoch
        if start < 0 or stop > data.shape[1]:
            continue  # skip epochs outside data range
        epochs.append(data[:, start:stop])
    return np.array(epochs)


def baseline_correction(epochs, baseline, srate):
    """
    Apply baseline correction to epochs.

    Args:
        epochs: Epochs array of shape (n_epochs, n_channels, n_samples)
        baseline: Tuple (start_ms, end_ms) defining baseline period
        srate: Sampling rate in Hz

    Returns:
        Baseline-corrected epochs
    """
    baseline_start = 0
    baseline_end = int((baseline[1] - baseline[0]) / 1000.0 * srate)
    baseline_mean = epochs[:, :, baseline_start:baseline_end].mean(axis=2, keepdims=True)
    return epochs - baseline_mean


def split_subjects(subjects, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15, seed=0):
    """
    Split subjects into train/dev/test sets.

    Args:
        subjects: List of unique subject IDs
        train_ratio: Proportion for training set
        dev_ratio: Proportion for development set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_subjects, dev_subjects, test_subjects)
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    n = len(subjects)
    n_train = int(train_ratio * n)
    n_dev = int(dev_ratio * n)

    # Shuffle subjects
    random.seed(seed)
    subjects_shuffled = subjects.copy()
    random.shuffle(subjects_shuffled)

    train_subjects = sorted(subjects_shuffled[:n_train])
    dev_subjects = sorted(subjects_shuffled[n_train:n_train + n_dev])
    test_subjects = sorted(subjects_shuffled[n_train + n_dev:])

    return train_subjects, dev_subjects, test_subjects


def normalize_trials(data, method='trial_wise'):
    """
    Normalize EEG data.

    Args:
        data: Data tensor of shape (n_trials, n_channels, n_timepoints)
        method: Normalization method
            - 'trial_wise': Normalize each trial independently across time
            - 'channel_wise': Normalize each trial-channel pair independently

    Returns:
        Normalized data tensor
    """
    if method == 'trial_wise':
        # Normalize across time for each trial
        mean = data.mean(dim=-1, keepdim=True)
        std = data.std(dim=-1, keepdim=True)
    elif method == 'channel_wise':
        # Normalize across channels and time for each trial
        mean = data.mean(dim=(-2, -1), keepdim=True)
        std = data.std(dim=(-2, -1), keepdim=True)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return (data - mean) / (std + 1e-8)
