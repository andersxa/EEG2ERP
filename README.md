# EEG2ERP: Deep Learning for Event-Related Potential Prediction

This repository contains code for training and evaluating EEG2ERP, a model that predicts Event-Related Potentials (ERPs) from few-trial EEG data.

## Project Structure

```
eeg2erp/
├── eeg2erp/                     # Main code
│   ├── data.py                  # Dataset classes and data loaders
│   ├── models.py                # Neural network architectures
│   ├── training.py              # Training loop and optimization
│   ├── evaluation.py            # Model evaluation and testing
│   └── utils.py                 # Utility functions
├── algorithms/                  # Reference algorithm implementations
│   ├── ride.py                  # RIDE decomposition
│   ├── woody.py                 # Woody's adaptive filter
│   ├── robust_average.py        # Robust averaging methods
│   └── xdawn.py                 # xDAWN spatial filtering
├── preprocessing/               # Dataset preprocessing scripts
│   ├── bcispeller.py            # BCISpeller P300 dataset
│   ├── wakeman_henson_eeg.py    # Wakeman-Henson EEG dataset
│   ├── wakeman_henson_meg.py    # Wakeman-Henson MEG dataset
│   └── utils.py                 # Preprocessing utilities
├── scripts/                     # Utility scripts for getting targets
│   ├── create_bootstrap_targets.py
│   ├── create_weighting_scheme.py
│   └── ...
└── tests/                       # Test and validation scripts
    ├── test_woody.py
    ├── test_robust_average.py
    ├── test_x_dawn.py
    └── test_all_models.py
```

## Installation

### Requirements

  - Python 3.12+
  - PyTorch 2.7+ with CUDA support
  - CUDA 12.4+
  - NVIDIA GPU
  - mne 1.10.1+

## Preprocessing

Preprocessing scripts are located in `preprocessing/` and handle:

- Epoch extraction from continuous EEG
- Baseline correction
- Resampling and filtering
- Normalization
- Train/dev/test splitting

### Example: Preprocess BCISpeller Data

```python
# Run the preprocessing script
python preprocessing/bcispeller.py

# This will create:
# - data/BCISpeller/simple_data.pt (preprocessed epochs)
```

## Data Format

Preprocessed data is stored as PyTorch `.pt` files with the following structure:

```python
{
    'dataset': 'BCISpeller',
    'data': torch.Tensor,        # (n_trials, n_channels, n_timepoints)
    'subjects': torch.Tensor,    # (n_trials,)
    'tasks': torch.Tensor,       # (n_trials,)
    'labels': dict,              # Task label mapping
    'train_subjects': list,      # Subject IDs for training
    'dev_subjects': list,        # Subject IDs for development
    'test_subjects': list,       # Subject IDs for testing
}
```

## Citation

If you use this code in your research, please cite:

```
Anders Vestergaard Nørskov, Kasper Jørgensen, Alexander Neergaard Zahid, and Morten Mørup. 2025. Estimating the Event-Related Potential from Few EEG Trials. Transactions on Machine Learning Research. https://openreview.net/forum?id=c6LgqDhpH0
```