#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from eeg2erp.data import ERPDataset
from tqdm import tqdm

dataset = ERPDataset(path='data/', split='train', processing='simple', num_samples=288, no_leakage=True, restricted=True).to('cuda')

REPEATS = 1000
dataset.deterministic()
subjects = dataset.unique_subjects
tasks = dataset.unique_tasks
NUM_WEIGHTS = 289
print("NUM_WEIGHTS", NUM_WEIGHTS)
# For each pair of s and t, split the full indices into 2 equal parts (randomly)
half1_indices = dataset.half_targets
half2_indices = dataset.half_inputs


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
from matplotlib import pyplot as plt
fig, ax = plt.subplots(1, len(tasks), figsize=(4*len(tasks), 5), dpi=80)
for i, t in enumerate(tasks):
    ax[i].plot(s_t_weights[:, i].sum(0).cpu().numpy())
    ax[i].set_title(t)
    ax[i].legend()
plt.show()
# %%
