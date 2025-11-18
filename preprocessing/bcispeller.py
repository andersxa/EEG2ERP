#%%
import mne
mne.set_log_level('critical')
from mne.preprocessing import Xdawn
import mat73
import numpy as np
from scipy.signal import resample
import torch
import random
from glob import glob
from tqdm import tqdm

def create_epochs(data, event_indices, tmin, srate, n_samples_epoch):
    epochs = []
    for ev in event_indices:
        start = int(ev + tmin * srate)  # note: tmin is negative
        stop = start + n_samples_epoch
        if start < 0 or stop > data.shape[1]:
            continue  # skip epochs that fall outside the data range
        epochs.append(data[:, start:stop])
    return np.array(epochs)  # shape: (n_epochs, n_channels, n_samples_epoch)

def baseline_correction(epochs, baseline, srate):
    #Baseline is negative, so we need to add the baseline to the epochs
    # E.g.: index = int(round((t - t_start) / dt))
    # So: baseline starts at 0 and ends at t=0
    # So: baseline_start = 0
    # So: baseline_end = int(round((0 - t_start) / dt)) = int(-t_start / dt)
    baseline_start = 0
    baseline_end = int((baseline[1]-baseline[0]) / 1000.0 * srate)
    baseline_mean = epochs[:, :, baseline_start:baseline_end].mean(axis=2, keepdims=True)
    return epochs - baseline_mean

mat_files = list(glob("data/BCISpeller/data/*.mat"))

# Parameters
baseline = [-200, 0]  # ms
frame = [0, 1000]  # ms
tmin = baseline[0] / 1000.0  # -0.2 s
tmax = frame[1] / 1000.0  # 0.8 s
sfreq = srate = 512
n_samples_epoch = int((frame[1] - frame[0]) / 1000.0 * srate)  # 800ms worth of samples
file_name = 'data/BCISpeller/simple_data.pt'

data_list = []
subjects_list = []
tasks_list = []
metadata_list = []
label_names = {1: 'target', 2: 'non_target'}
xdawn_dict = {}
# #%%

for subject_id, mat_file in enumerate(tqdm(mat_files)):
    eeg = mat73.loadmat(mat_file)


    # pre-processing for training data
    for n_calib in range(len(eeg['train'])):
        data = np.asarray(eeg['train'][n_calib]['data'])
        srate = eeg['train'][n_calib]['srate']
        markers = eeg['train'][n_calib]['markers_target']
    
        # Save relevant metadata
        metadata = dict(
            subject=subject_id,
            markers_target=np.array(eeg['train'][n_calib]['markers_target']),
            markers_seq=np.array(eeg['train'][n_calib]['markers_seq']),
            text_to_spell=eeg['train'][n_calib]['text_to_spell'],
            nbTrials=eeg['train'][n_calib]['nbTrials'],
            srate=eeg['train'][n_calib]['srate'],
        )

        metadata_list.append(metadata)

        targetID = np.where(markers == 1)[0]
        nontargetID = np.where(markers == 2)[0]

        target_epochs = create_epochs(data, targetID, tmin, srate, n_samples_epoch)  # (n_epochs, n_channels, n_samples)
        nontarget_epochs = create_epochs(data, nontargetID, tmin, srate, n_samples_epoch)

        # Baseline correction
        target_epochs = baseline_correction(target_epochs, baseline, srate)
        nontarget_epochs = baseline_correction(nontarget_epochs, baseline, srate)

        # Apply downsampling
        target_epochs = resample(target_epochs, sfreq // 2, axis=2)
        nontarget_epochs = resample(nontarget_epochs, sfreq // 2, axis=2)

        # Transpose to match original shape: (channels, samples, epochs)
        tmp_targetEEG = np.transpose(target_epochs, (1, 2, 0))
        tmp_nontargetEEG = np.transpose(nontarget_epochs, (1, 2, 0))

        if n_calib == 0:
            targetEEG = tmp_targetEEG
            nontargetEEG = tmp_nontargetEEG
        else:
            targetEEG = np.dstack((targetEEG, tmp_targetEEG))
            nontargetEEG = np.dstack((nontargetEEG, tmp_nontargetEEG))

    targetEEG = np.transpose(targetEEG, (2, 0, 1))  # [trials, channels, time]
    nontargetEEG = np.transpose(nontargetEEG, (2, 0, 1))

    all_epochs = np.concatenate((targetEEG, nontargetEEG), axis=0)
    tasks = np.concatenate((np.ones(targetEEG.shape[0], dtype=int), 2 * np.ones(nontargetEEG.shape[0], dtype=int)))
    subjects = np.full(all_epochs.shape[0], subject_id, dtype=int)

    # Create MNE Epochs object
    # events = np.column_stack((
    #     np.arange(len(tasks)),
    #     np.zeros(len(tasks), dtype=int),
    #     tasks
    # ))

    # biosemi_montage = mne.channels.make_standard_montage('biosemi32')
    # info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=sfreq // 2, ch_types='eeg')

    # event_id = {'target': 1, 'non_target': 2}
    # epochs = mne.EpochsArray(all_epochs, info, events=events, tmin=tmin, event_id=event_id)

    # # xDAWN
    # xd = Xdawn(n_components=2, correct_overlap=False)
    # xd.fit(epochs)
    # xdawn_dict[subject_id] = xd  # Save Xdawn per subject

    # Store data
    data_list.append(all_epochs)
    subjects_list.append(subjects)
    tasks_list.append(tasks)

# Stack all
data = torch.from_numpy(np.concatenate(data_list, axis=0))

# Normalize data
data = (data - data.mean(dim=-1, keepdim=True)) / data.std(dim=-1, keepdim=True)
# or else.
# data = (data - data.mean(dim=(-2, -1), keepdim=True)) / data.std(dim=(-2,-1), keepdim=True)

subjects = torch.from_numpy(np.concatenate(subjects_list))
tasks = torch.from_numpy(np.concatenate(tasks_list))

# Split subjects
unique_subjects = list(set(subjects.numpy()))
n_train_subjects = int(len(unique_subjects) * 0.7)
n_dev_subjects = 5

random.seed(0)
random.shuffle(unique_subjects)

dev_subjects = [int(x) for x in sorted(unique_subjects[:n_dev_subjects])]
train_subjects = [int(x) for x in sorted(unique_subjects[n_dev_subjects:n_dev_subjects + n_train_subjects])]
test_subjects = [int(x) for x in sorted(unique_subjects[n_dev_subjects + n_train_subjects:])]
#%%
# Save the full dataset
file_name = "data/BCISpeller/simple_data.pt"
torch.save(
    dict(
        dataset='BCISpeller',
        data=data,
        subjects=subjects,
        tasks=tasks,
        labels=label_names,
        train_subjects=train_subjects,
        test_subjects=test_subjects,
        dev_subjects=dev_subjects,
        metadata=metadata_list
    ),
    file_name
)

# # Save xDAWN objects separately
# torch.save(xdawn_dict, "data/BCISpeller/xdawn_models.pt")
#%%
from data import ERPDataset

path = 'data/BCISpeller/'
processing = 'simple'
dataset = ERPDataset(path, split='train', processing=processing)
dataset.save_mean_std()
#%%
from data import ERPDataset
import torch
path = 'data/BCISpeller/'
processing = 'simple'
dataset = ERPDataset(path, split='dev', processing=processing)
bootstrap_targets = torch.load(f"data/BCISpeller/simple_dev_200_half_half_bootstrap.pt", weights_only=False)
#%%
task = 1
subject = dataset.unique_subjects[2]
indices = dataset.get_indices(subject, task, True)
targets = dataset.data[indices]
indices = dataset.get_indices(subject, task, False)
inputs = dataset.data[indices]

from matplotlib import pyplot as plt
# Mean targets and inputs over first dimension
mean_targets = targets.mean(dim=0)
mean_inputs = inputs[:15].mean(dim=0)
ch = 31
plt.plot(mean_targets[ch].numpy(), label='target')
plt.plot(mean_inputs[ch].numpy(), label='input')
plt.legend()

tss = torch.sum(mean_targets.square())
rss = torch.sum((mean_targets - mean_inputs).square())
r2 = (1 - rss / tss).item()
print('rw2:', r2)
# %%
