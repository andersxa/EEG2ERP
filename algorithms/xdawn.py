"""
xDAWN spatial filtering for ERP enhancement.

This module provides a wrapper around MNE's xDAWN implementation
for enhancing event-related potentials through spatial filtering.

Reference: Rivet et al. (2009) "xDAWN Algorithm to Enhance Evoked Potentials:
Application to Brain-Computer Interface"
"""

import numpy as np
import mne
from mne.preprocessing import Xdawn


def apply_xdawn_filter(epochs_train, epochs_test, n_components=2, correct_overlap=False):
    """
    Train xDAWN spatial filter on training epochs and apply to test epochs.

    Args:
        epochs_train: MNE Epochs object for training
        epochs_test: MNE Epochs object for testing
        n_components: Number of xDAWN components to extract
        correct_overlap: Whether to correct for overlap in xDAWN

    Returns:
        epochs_denoised: Spatially filtered test epochs
        xd: Trained xDAWN object
    """
    xd = Xdawn(n_components=n_components, correct_overlap=correct_overlap)
    xd.fit(epochs_train)
    epochs_denoised = xd.apply(epochs_test)
    return epochs_denoised, xd


def create_epochs_from_data(data, labels, ch_names=None, sfreq=256., tmin=-0.2):
    """
    Create MNE Epochs object from numpy arrays.

    Args:
        data: Numpy array of shape (n_trials, n_channels, n_timepoints)
        labels: Array of event labels for each trial
        ch_names: List of channel names (default: BioSemi32 montage)
        sfreq: Sampling frequency in Hz
        tmin: Start time of epochs relative to event

    Returns:
        epochs: MNE Epochs object
    """
    if ch_names is None:
        biosemi_montage = mne.channels.make_standard_montage('biosemi32')
        ch_names = biosemi_montage.ch_names

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    # Create events array
    n_trials = data.shape[0]
    events = np.zeros((n_trials, 3), dtype=int)
    events[:, 0] = np.arange(n_trials)
    events[:, 2] = labels

    epochs = mne.EpochsArray(data, info, events, tmin=tmin, verbose=False)
    return epochs
