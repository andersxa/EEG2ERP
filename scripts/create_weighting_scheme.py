#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from eeg2erp.data import ERPDataset
from tqdm import tqdm

dataset = ERPDataset(path='data/', split='train', processing='simple', num_samples=585, no_leakage=True, restricted=False).to('cuda')

REPEATS = 200
dataset.deterministic()
subjects = dataset.unique_subjects
tasks = dataset.unique_tasks
NUM_WEIGHTS = 586
print("NUM_WEIGHTS", NUM_WEIGHTS)
# For each pair of s and t, split the full indices into 2 equal parts (randomly)
half1_indices = dataset.half_targets
half2_indices = dataset.half_inputs
#%%

data = {}
with torch.inference_mode():
    s_t_weights = torch.zeros((len(subjects), len(tasks), NUM_WEIGHTS), device='cuda')
    total = len(subjects)*len(tasks)*REPEATS
    t_bar = tqdm(total=total)
    for i, s in enumerate(subjects):
        for j, t in enumerate(tasks):
            for r in range(REPEATS):
                # Sample targets from half1
                target_indices = dataset.rng.choice(half1_indices[s][t], dataset.get_num_samples(s, t, target=True))
                target_data = dataset.data[target_indices].mean(dim=0)

                bootstrap_indices = dataset.rng.choice(half2_indices[s][t], NUM_WEIGHTS)
                bootstrap_data = dataset.data[bootstrap_indices]
                bootstrap_data = bootstrap_data.cumsum(dim=0) / torch.arange(1, NUM_WEIGHTS+1, device='cuda').unsqueeze(1).unsqueeze(2)
                s_t_weights[i, j] += (target_data.unsqueeze(0).expand_as(bootstrap_data) - bootstrap_data).square().sum(dim=(1, 2))
                t_bar.update(1)
# %%
# Create task_to_task_index
weight_task_to_task_index = {t: i for i, t in enumerate(tasks)}
# Save weights
if dataset.restricted:
    save_path = f'data/{dataset.processing}_num_trials_weights.pt'
else:
    save_path = f'data/{dataset.processing}_unrestricted_num_trials_weights.pt'
torch.save({'s_t_weights': s_t_weights.cpu().numpy(), 'weight_task_to_task_index': weight_task_to_task_index}, save_path)
#%%
save_path1 = f'data/{dataset.processing}_num_trials_weights.pt'
save_path2 = f'data/{dataset.processing}_unrestricted_num_trials_weights.pt'
data1 = torch.load(save_path1)
data2 = torch.load(save_path2)
#%%
from matplotlib import pyplot as plt
fig, ax = plt.subplots(1, len(tasks), figsize=(4*len(tasks), 5), dpi=80)
for i, t in enumerate(tasks):
    ax[i].plot(data1['s_t_weights'][:, i].sum(0), label='restricted')
    ax[i].plot(data2['s_t_weights'][:, i].sum(0), label='unrestricted')
    ax[i].set_title(t)
    ax[i].legend()
plt.show()
# %%
from eeg2erp.data import ERPCoreData
dataset = ERPCoreData(path='data/', split='dev', processing='simple', sample_method='weighted', num_samples=585, no_leakage=True, restricted=False)
#%%
s = dataset.unique_subjects[0]
t = dataset.unique_tasks[0]
NUM_SAMPLES = 10000
sampled_inputs = []
sampled_targets = []
for n in range(NUM_SAMPLES):
    input_index = dataset.sample_num_trials(1, s, t, target=False)
    target_index = dataset.sample_num_trials(input_index, s, t, target=True)
    sampled_inputs.append(input_index)
    sampled_targets.append(target_index)
# %%
from matplotlib import pyplot as plt
# Make a histogram of sampled_indices
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].hist(sampled_inputs, bins=50)
axs[1].hist(sampled_targets, bins=50)
plt.show()
# %%


import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_st = 100  # Total number of trials

# Sampling schemes
K = np.arange(1, N_st + 1)  # Trial numbers
uniform_weights = np.ones_like(K) / N_st  # Uniform weights
weighted_weights = (1 / K)
weighted_weights /= weighted_weights.sum()  # Normalize weights

weighted_weights2 = (1 / np.sqrt(K))
weighted_weights2 /= weighted_weights2.sum()  # Normalize weights

# Plot
plt.figure(figsize=(6.5, 4.5))
plt.plot(K, uniform_weights, label='Uniform Sampling', linestyle='--', linewidth=2)
plt.plot(K, weighted_weights, label='Weighted Sampling', linewidth=2)
plt.plot(K, weighted_weights2, label='Weighted Sampling 2', linewidth=2)
plt.title('Sampling Schemes for Trial Selection', fontsize=16)
plt.xlabel('Number of Trials (K)', fontsize=14)
plt.ylabel('Sampling Probability', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
# Remove padding
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.savefig("figures/sampling_schemes.png", dpi=180, bbox_inches='tight', pad_inches=0)
# %%
