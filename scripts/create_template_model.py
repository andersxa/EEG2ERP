#%%
from eeg2erp.data import ERPDataset
import torch
path = 'data/ERP Core/'
processing = 'simple'
dataset = ERPDataset(path, split='train', processing=processing)
template_dict = {}
for t in dataset.unique_tasks:
    indices = dataset.tasks == t
    data = dataset.data[indices][:, dataset.task_to_channel[t]]
    mean = data.mean(dim=0)
    template_dict[t] = mean
torch.save(template_dict, f"{path}/{processing}_{dataset.split}_templates.pt")
#%%
import numpy as np
import mne
from mne.preprocessing import Xdawn
import pickle

# ----- Parameters (modify these as needed) -----
sfreq = 250  # Sampling frequency in Hz (adjust to your dataset)
tmin = -0.2  # Start time of each epoch (in seconds)
n_components = 2  # Number of xDAWN components

# ----- Create the MNE Info object -----
n_channels = dataset.data.shape[1]
biosemi_montage = mne.channels.make_standard_montage('biosemi32')
info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=256., ch_types='eeg')

# ----- Create an event_id dictionary from dataset.tasks -----
# Each unique task label is mapped to an integer ID.
unique_tasks = np.unique(dataset.tasks)
task_to_label = dataset.task_to_label
event_id = {task_to_label[task]: task for task in unique_tasks}

# ----- Build the events array -----
# MNE expects an events array of shape (n_epochs, 3): 
#  - first column: arbitrary sample index (here we use the epoch index)
#  - second column: a placeholder (set to 0)
#  - third column: event ID according to event_id
n_epochs = dataset.data.shape[0]
events = np.zeros((n_epochs, 3), dtype=int)
events[:, 0] = np.arange(n_epochs)  # using epoch index as the "sample" index
events[:, 2] = [task for task in dataset.tasks]

# ----- Create the Epochs object -----
epochs = mne.EpochsArray(dataset.data, info, events=events, event_id=event_id, tmin=tmin)

# ----- Initialize and train the xDAWN model -----
# Here we let xDAWN compute the signal covariance from the epochs automatically (signal_cov=None).
xd = Xdawn(n_components=n_components, correct_overlap=False)
xd.fit(epochs)

#%%
# Optionally, you can apply the spatial filtering:
epochs_denoised = xd.apply(epochs, event_id=['target'])

#%%
# ----- Save the trained xDAWN object -----
# This saves the fitted model to disk (using pickle in this example).
with open(f"{path}/xdawn_model.pkl", "wb") as f:
    pickle.dump(xd, f)
# %%
# Plot image epoch after Xdawn
mne.viz.plot_epochs_image(epochs_denoised["target"]["target"][:100], picks='Cz')
#%%