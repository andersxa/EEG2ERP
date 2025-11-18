"""
EEG2ERP: Deep Learning for Event-Related Potential Prediction from EEG Data

This package provides tools for training and evaluating neural network models
that predict ERPs from single-trial EEG data.
"""

from .data import ERPDataset, ERPBootstrapTargets
from .models import ERPUNet, ERPAE, CSLPAE, ERPVAE, ERPVQVAE, ERPGCVAE
from .utils import load_model, plot_bootstrap_curve, get_measures

__version__ = "1.0.0"

__all__ = [
    "ERPDataset",
    "ERPBootstrapTargets",
    "ERPUNet",
    "ERPAE",
    "CSLPAE",
    "ERPVAE",
    "ERPVQVAE",
    "ERPGCVAE",
    "load_model",
    "plot_bootstrap_curve",
    "get_measures",
]
